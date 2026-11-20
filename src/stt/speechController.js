/**
 * Real-Time Streaming Speech-to-Text Controller (100% In-Browser WebAI).
 *
 * Engines:
 *   1. "moonshine" — Ultra-fast offline WebAI (onnx-community/moonshine-tiny-ONNX) on WebGPU/WASM.
 *                    Variable-length audio (no 30s Whisper padding) -> ~100-250ms per 1.4s live burst.
 *   2. "webspeech" — Instant streaming browser Web Speech API with live interim prefix commit
 *                    (emits words while you are still talking — zero waiting for end of paragraph).
 *   3. "whisper"   — Offline WebAI Whisper Tiny.en (English-only, fixed generation params).
 */

const TARGET_SAMPLE_RATE = 16000;
const BASE_VAD_THRESHOLD = 0.014;
const MICRO_PAUSE_MS = 260;     // Flush quickly on natural micro-pause between phrases
const STREAM_BURST_SEC = 1.45;  // Force flush every ~1.45s during continuous speech (real-time!)
const MIN_BURST_SEC = 0.32;     // Ignore tiny clicks/pops
const PRE_ROLL_CHUNKS = 2;      // Keep ~170ms pre-roll so first syllable isn't clipped

function downsampleTo16k(float32Audio, inputRate) {
  if (inputRate === TARGET_SAMPLE_RATE) return float32Audio;
  const ratio = inputRate / TARGET_SAMPLE_RATE;
  const newLen = Math.round(float32Audio.length / ratio);
  const out = new Float32Array(newLen);
  for (let i = 0; i < newLen; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.min(float32Audio.length, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = start; j < end; j++) sum += float32Audio[j];
    out[i] = sum / Math.max(1, end - start);
  }
  return out;
}

function concatFloat32(chunks) {
  const total = chunks.reduce((acc, c) => acc + c.length, 0);
  const merged = new Float32Array(total);
  let offset = 0;
  for (const c of chunks) {
    merged.set(c, offset);
    offset += c.length;
  }
  return merged;
}

export class SpeechController {
  constructor(callbacks = {}) {
    this.onTranscript = callbacks.onTranscript || (() => {});
    this.onInterim = callbacks.onInterim || (() => {});
    this.onStatus = callbacks.onStatus || (() => {});
    this.onLevel = callbacks.onLevel || (() => {});

    this.engine = "moonshine"; // "moonshine" | "webspeech" | "whisper"
    this.isListening = false;

    // WebAI Worker + Streaming VAD state
    this.worker = null;
    this.loadedModelKey = null;
    this.workerBusy = false;
    this.pendingAudioChunks = [];
    this.lastTailWord = "";

    this.mediaStream = null;
    this.audioCtx = null;
    this.processorNode = null;
    this.preRoll = [];
    this.speechChunks = [];
    this.isSpeaking = false;
    this.noiseFloor = 0.006;
    this.silenceStartMs = 0;
    this.burstStartMs = 0;

    // Web Speech API streaming prefix tracker + crash-proof auto-restart
    this.recognition = null;
    this.emittedWordsByResult = new Map();
    this.interimFlushTimer = null;
    this.webSpeechRestartTimer = null;
    this.webSpeechSessionId = 0;
    this.webSpeechErrorCount = 0;
  }

  static hasWebSpeech() {
    return (
      typeof window !== "undefined" &&
      Boolean(window.SpeechRecognition || window.webkitSpeechRecognition)
    );
  }

  setEngine(engine) {
    if (this.engine === engine) return;
    const wasListening = this.isListening;
    if (wasListening) this.stop();
    this.engine = engine;
    if (wasListening) this.start();
  }

  ensureWorker() {
    if (this.worker) return;
    this.worker = new Worker(new URL("./whisper.worker.js", import.meta.url), {
      type: "module",
    });

    this.worker.onmessage = (event) => {
      const msg = event.data || {};
      if (msg.type === "STATUS") {
        this.onStatus(msg.text, "loading");
      } else if (msg.type === "PROGRESS") {
        const pct = Math.round(msg.progress);
        this.onStatus(`Downloading ${msg.label || "WebAI STT"} (${pct}%)…`, "loading");
      } else if (msg.type === "READY") {
        this.loadedModelKey = msg.modelKey;
        const name = msg.modelKey === "whisper" ? "Whisper" : "Moonshine";
        this.onStatus(
          this.isListening
            ? `Live · ${name} (${msg.device.toUpperCase()})`
            : `${name} Ready (${msg.device.toUpperCase()})`,
          "ready"
        );
      } else if (msg.type === "TRANSCRIPT") {
        this.workerBusy = false;
        const cleaned = this.dedupeBoundary(msg.text);
        if (cleaned) {
          this.onTranscript(cleaned, {
            engine: msg.modelKey || this.engine,
            latencyMs: msg.latencyMs,
          });
        }
        this.flushPendingAudio();
      } else if (msg.type === "IGNORED") {
        this.workerBusy = false;
        this.flushPendingAudio();
      } else if (msg.type === "ERROR") {
        this.workerBusy = false;
        this.onStatus(`STT error: ${msg.message}`, "error");
        this.flushPendingAudio();
      }
    };
  }

  /**
   * Avoids repeating a word split right across two consecutive 1.45s streaming bursts.
   */
  dedupeBoundary(text) {
    const words = text.trim().split(/\s+/).filter(Boolean);
    if (words.length === 0) return "";
    const firstNorm = words[0].toLowerCase().replace(/[^a-z0-9]/g, "");
    if (this.lastTailWord && firstNorm === this.lastTailWord && words.length > 1) {
      words.shift();
    }
    this.lastTailWord = words[words.length - 1].toLowerCase().replace(/[^a-z0-9]/g, "");
    return words.join(" ");
  }

  async start() {
    if (this.isListening) return;
    this.lastTailWord = "";
    this.webSpeechErrorCount = 0;
    if (this.engine === "webspeech") {
      this.isListening = true;
      this.startWebSpeechStreaming();
    } else {
      await this.startLocalWebAIMic(this.engine);
    }
  }

  stop() {
    this.isListening = false;
    this.isSpeaking = false;
    this.speechChunks = [];
    this.preRoll = [];
    this.pendingAudioChunks = [];
    this.webSpeechSessionId++;
    clearTimeout(this.interimFlushTimer);
    clearTimeout(this.webSpeechRestartTimer);
    this.emittedWordsByResult.clear();
    this.onLevel(0);

    if (this.processorNode) {
      this.processorNode.disconnect();
      this.processorNode = null;
    }
    if (this.audioCtx) {
      this.audioCtx.close().catch(() => {});
      this.audioCtx = null;
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((t) => t.stop());
      this.mediaStream = null;
    }
    if (this.recognition) {
      try {
        this.recognition.onresult = null;
        this.recognition.onerror = null;
        this.recognition.onend = null;
        this.recognition.onstart = null;
        this.recognition.abort();
      } catch {
        // Ignore
      }
      this.recognition = null;
    }
    this.onStatus("Mic Off", "idle");
  }

  async startLocalWebAIMic(modelKey) {
    this.ensureWorker();
    if (this.loadedModelKey !== modelKey) {
      this.worker.postMessage({ type: "INIT", modelKey });
    }

    this.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
      video: false,
    });

    this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const source = this.audioCtx.createMediaStreamSource(this.mediaStream);
    const processor = this.audioCtx.createScriptProcessor(2048, 1, 1);
    this.processorNode = processor;

    const sampleRate = this.audioCtx.sampleRate;
    this.isListening = true;
    this.speechChunks = [];
    this.preRoll = [];
    this.isSpeaking = false;
    this.noiseFloor = 0.006;

    const name = modelKey === "whisper" ? "Whisper" : "Moonshine";
    this.onStatus(
      this.loadedModelKey === modelKey ? `Live · ${name} WebAI` : `Loading ${name} WebAI…`,
      this.loadedModelKey === modelKey ? "ready" : "loading"
    );

    processor.onaudioprocess = (e) => {
      if (!this.isListening) return;
      const input = e.inputBuffer.getChannelData(0);

      let sumSq = 0;
      for (let i = 0; i < input.length; i++) sumSq += input[i] * input[i];
      const rms = Math.sqrt(sumSq / input.length);
      this.onLevel(Math.min(1, rms * 10));

      const now = performance.now();
      const chunk16k = downsampleTo16k(new Float32Array(input), sampleRate);

      // Adaptive noise floor so PC fans/hum never lock VAD open for 6 seconds
      const threshold = Math.max(BASE_VAD_THRESHOLD, this.noiseFloor * 2.5);

      if (rms >= threshold) {
        if (!this.isSpeaking) {
          this.isSpeaking = true;
          this.burstStartMs = now;
          this.speechChunks = [...this.preRoll];
          this.onInterim("Listening…");
        }
        this.silenceStartMs = 0;
        this.speechChunks.push(chunk16k);

        // Real-time burst flush every ~1.45s while speaking continuously
        if ((now - this.burstStartMs) / 1000 >= STREAM_BURST_SEC) {
          this.dispatchBurst(modelKey, true);
        }
      } else {
        // Slowly adapt ambient noise floor during silence
        this.noiseFloor = this.noiseFloor * 0.96 + rms * 0.04;

        this.preRoll.push(chunk16k);
        if (this.preRoll.length > PRE_ROLL_CHUNKS) this.preRoll.shift();

        if (this.isSpeaking) {
          this.speechChunks.push(chunk16k);
          if (!this.silenceStartMs) {
            this.silenceStartMs = now;
          } else if (now - this.silenceStartMs >= MICRO_PAUSE_MS) {
            this.dispatchBurst(modelKey, false);
          }
        }
      }
    };

    source.connect(processor);
    processor.connect(this.audioCtx.destination);
  }

  dispatchBurst(modelKey, stillSpeaking) {
    const merged = concatFloat32(this.speechChunks);
    this.speechChunks = [];
    this.silenceStartMs = 0;
    if (stillSpeaking) {
      this.burstStartMs = performance.now();
    } else {
      this.isSpeaking = false;
    }

    if (merged.length < TARGET_SAMPLE_RATE * MIN_BURST_SEC) {
      if (!stillSpeaking) this.onInterim("");
      return;
    }

    this.onInterim("Translating live speech…");
    if (this.workerBusy) {
      // Queue at most 2 recent bursts so we stay real-time instead of building lag
      this.pendingAudioChunks.push(merged);
      if (this.pendingAudioChunks.length > 2) {
        this.pendingAudioChunks.shift();
      }
      return;
    }

    this.workerBusy = true;
    this.worker.postMessage(
      { type: "TRANSCRIBE", audio: merged, modelKey },
      [merged.buffer]
    );
  }

  flushPendingAudio() {
    if (this.pendingAudioChunks.length === 0 || this.workerBusy || !this.isListening) return;
    const merged = concatFloat32(this.pendingAudioChunks);
    this.pendingAudioChunks = [];
    this.workerBusy = true;
    this.worker.postMessage(
      { type: "TRANSCRIBE", audio: merged, modelKey: this.engine },
      [merged.buffer]
    );
  }

  /**
   * True real-time Web Speech streaming with crash-proof auto-recovery:
   * - Commits stable interim words mid-sentence (never waits 5s for isFinal).
   * - Creates a fresh SpeechRecognition instance with backoff on Chrome session end/errors
   *   so InvalidStateError or network hiccups can never kill the recognizer or freeze the tab.
   */
  startWebSpeechStreaming() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      this.isListening = false;
      this.onStatus("Web Speech API unavailable — switch to Moonshine WebAI.", "error");
      return;
    }

    clearTimeout(this.interimFlushTimer);
    clearTimeout(this.webSpeechRestartTimer);

    if (this.recognition) {
      try {
        this.recognition.onresult = null;
        this.recognition.onerror = null;
        this.recognition.onend = null;
        this.recognition.onstart = null;
        this.recognition.abort();
      } catch {
        // Ignore
      }
      this.recognition = null;
    }

    const sessionId = ++this.webSpeechSessionId;
    this.emittedWordsByResult.clear();

    const rec = new SR();
    rec.lang = "en-US";
    rec.continuous = true;
    rec.interimResults = true;
    rec.maxAlternatives = 1;

    const scheduleRestart = (delayMs, statusText) => {
      if (!this.isListening || sessionId !== this.webSpeechSessionId) return;
      clearTimeout(this.webSpeechRestartTimer);
      if (statusText) this.onStatus(statusText, "loading");
      this.webSpeechRestartTimer = setTimeout(() => {
        if (this.isListening && sessionId === this.webSpeechSessionId) {
          this.startWebSpeechStreaming();
        }
      }, delayMs);
    };

    rec.onstart = () => {
      if (sessionId !== this.webSpeechSessionId) return;
      this.onStatus("Live Streaming · Web Speech", "ready");
    };

    const emitNewSlice = (resultIdx, words, forceAll) => {
      if (!this.isListening || sessionId !== this.webSpeechSessionId) return;
      const alreadyEmitted = this.emittedWordsByResult.get(resultIdx) || 0;
      const commitUpTo = forceAll ? words.length : Math.max(alreadyEmitted, words.length - 1);
      const countToEmit = commitUpTo - alreadyEmitted;

      if (countToEmit <= 0) return;
      if (!forceAll && countToEmit < 2) return;

      const slice = words.slice(alreadyEmitted, commitUpTo).join(" ").trim();
      this.emittedWordsByResult.set(resultIdx, commitUpTo);
      if (slice) {
        try {
          this.onTranscript(slice, { engine: "webspeech", latencyMs: 0 });
        } catch (err) {
          console.error("[OpenASL STT] Transcript callback error:", err);
        }
      }
    };

    rec.onresult = (event) => {
      if (sessionId !== this.webSpeechSessionId) return;
      this.webSpeechErrorCount = 0; // Reset backoff on any healthy speech event
      clearTimeout(this.interimFlushTimer);

      let latestInterimIdx = -1;
      let latestWords = [];

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const res = event.results[i];
        const txt = res[0]?.transcript?.trim() || "";
        if (!txt) continue;

        const words = txt.split(/\s+/).filter(Boolean);
        if (res.isFinal) {
          emitNewSlice(i, words, true);
          this.onInterim("");
        } else {
          this.onInterim(txt);
          emitNewSlice(i, words, false);
          latestInterimIdx = i;
          latestWords = words;
        }
      }

      if (latestInterimIdx >= 0 && latestWords.length > 0) {
        this.interimFlushTimer = setTimeout(() => {
          emitNewSlice(latestInterimIdx, latestWords, true);
        }, 350);
      }
    };

    rec.onerror = (e) => {
      if (sessionId !== this.webSpeechSessionId) return;
      if (e.error === "not-allowed" || e.error === "service-not-allowed") {
        this.isListening = false;
        this.onStatus("Mic permission blocked by browser.", "error");
        return;
      }
      if (e.error !== "no-speech" && e.error !== "aborted") {
        this.webSpeechErrorCount++;
        this.onStatus(`Web Speech reconnecting (${e.error})…`, "loading");
      }
    };

    rec.onend = () => {
      if (sessionId !== this.webSpeechSessionId) return;
      clearTimeout(this.interimFlushTimer);
      this.emittedWordsByResult.clear();
      if (this.isListening) {
        // Gentle backoff prevents Chrome InvalidStateError & tight crash loops
        const delay = Math.min(2000, 260 * Math.pow(1.6, this.webSpeechErrorCount));
        scheduleRestart(delay, this.webSpeechErrorCount > 0 ? "Reconnecting Web Speech…" : null);
      }
    };

    this.recognition = rec;
    try {
      rec.start();
    } catch (err) {
      this.webSpeechErrorCount++;
      scheduleRestart(Math.min(2000, 350 * this.webSpeechErrorCount), "Restarting Web Speech…");
    }
  }
}
