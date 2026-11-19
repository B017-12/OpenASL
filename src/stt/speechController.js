/**
 * Real-Time Speech-to-Text Controller (100% In-Browser WebAI).
 * Supports:
 *   1. "whisper"    — Local offline Transformers.js Whisper (WebGPU/WASM) in a Web Worker,
 *                     gated by RMS Voice Activity Detection (450ms pause flushes clause).
 *   2. "webspeech"  — Instant browser Web Speech API fallback (zero model download).
 */

const TARGET_SAMPLE_RATE = 16000;
const VAD_RMS_THRESHOLD = 0.016;
const SILENCE_FLUSH_MS = 450;
const MAX_CLAUSE_SECONDS = 6.0;
const MIN_CLAUSE_SECONDS = 0.45;

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

    this.engine = "whisper"; // "whisper" | "webspeech"
    this.isListening = false;

    // Whisper Worker + Audio VAD state
    this.worker = null;
    this.whisperReady = false;
    this.whisperBusy = false;
    this.pendingAudio = null;

    this.mediaStream = null;
    this.audioCtx = null;
    this.processorNode = null;
    this.speechChunks = [];
    this.isSpeaking = false;
    this.silenceStartMs = 0;
    this.clauseStartMs = 0;

    // Web Speech API state
    this.recognition = null;
  }

  static hasWebSpeech() {
    return typeof window !== "undefined" && Boolean(window.SpeechRecognition || window.webkitSpeechRecognition);
  }

  setEngine(engine) {
    if (this.engine === engine) return;
    const wasListening = this.isListening;
    if (wasListening) this.stop();
    this.engine = engine;
    if (wasListening) this.start();
  }

  ensureWhisperWorker() {
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
        this.onStatus(`Downloading Whisper (${pct}%)…`, "loading");
      } else if (msg.type === "READY") {
        this.whisperReady = true;
        this.onStatus(
          this.isListening
            ? `Listening · Whisper (${msg.device.toUpperCase()})`
            : `Whisper Ready (${msg.device.toUpperCase()})`,
          "ready"
        );
      } else if (msg.type === "TRANSCRIPT") {
        this.whisperBusy = false;
        this.onTranscript(msg.text, { engine: "whisper", latencyMs: msg.latencyMs });
        this.flushPendingAudio();
      } else if (msg.type === "IGNORED") {
        this.whisperBusy = false;
        this.flushPendingAudio();
      } else if (msg.type === "ERROR") {
        this.whisperBusy = false;
        this.onStatus(`Whisper error: ${msg.message}`, "error");
      }
    };
  }

  preloadWhisper() {
    this.ensureWhisperWorker();
    this.worker.postMessage({ type: "INIT" });
  }

  async start() {
    if (this.isListening) return;
    if (this.engine === "webspeech") {
      this.startWebSpeech();
    } else {
      await this.startWhisperMic();
    }
  }

  stop() {
    this.isListening = false;
    this.isSpeaking = false;
    this.speechChunks = [];
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
        this.recognition.onend = null;
        this.recognition.stop();
      } catch {
        // Ignore
      }
      this.recognition = null;
    }
    this.onStatus("Mic Off", "idle");
  }

  async startWhisperMic() {
    this.ensureWhisperWorker();
    if (!this.whisperReady) {
      this.worker.postMessage({ type: "INIT" });
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
    // ScriptProcessorNode works cleanly across all Chromium/Electron versions without external worklet URLs
    const processor = this.audioCtx.createScriptProcessor(4096, 1, 1);
    this.processorNode = processor;

    const sampleRate = this.audioCtx.sampleRate;
    this.isListening = true;
    this.speechChunks = [];
    this.isSpeaking = false;

    this.onStatus(
      this.whisperReady ? "Listening · Whisper WebAI" : "Loading Whisper WebAI…",
      this.whisperReady ? "ready" : "loading"
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

      if (rms >= VAD_RMS_THRESHOLD) {
        if (!this.isSpeaking) {
          this.isSpeaking = true;
          this.clauseStartMs = now;
          this.onInterim("Listening…");
        }
        this.silenceStartMs = 0;
        this.speechChunks.push(chunk16k);

        if ((now - this.clauseStartMs) / 1000 >= MAX_CLAUSE_SECONDS) {
          this.dispatchClause();
        }
      } else if (this.isSpeaking) {
        this.speechChunks.push(chunk16k);
        if (!this.silenceStartMs) {
          this.silenceStartMs = now;
        } else if (now - this.silenceStartMs >= SILENCE_FLUSH_MS) {
          this.dispatchClause();
        }
      }
    };

    source.connect(processor);
    processor.connect(this.audioCtx.destination);
  }

  dispatchClause() {
    const merged = concatFloat32(this.speechChunks);
    this.speechChunks = [];
    this.isSpeaking = false;
    this.silenceStartMs = 0;

    if (merged.length < TARGET_SAMPLE_RATE * MIN_CLAUSE_SECONDS) {
      this.onInterim("");
      return;
    }

    this.onInterim("Transcribing clause…");
    if (this.whisperBusy) {
      this.pendingAudio = merged;
      return;
    }
    this.whisperBusy = true;
    this.worker.postMessage({ type: "TRANSCRIBE", audio: merged }, [merged.buffer]);
  }

  flushPendingAudio() {
    if (!this.pendingAudio || this.whisperBusy) return;
    const next = this.pendingAudio;
    this.pendingAudio = null;
    this.whisperBusy = true;
    this.worker.postMessage({ type: "TRANSCRIBE", audio: next }, [next.buffer]);
  }

  startWebSpeech() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      this.onStatus("Web Speech API unavailable in this browser — use Whisper WebAI.", "error");
      return;
    }

    const rec = new SR();
    rec.lang = "en-US";
    rec.continuous = true;
    rec.interimResults = true;

    rec.onstart = () => {
      this.isListening = true;
      this.onStatus("Listening · Web Speech API", "ready");
    };

    rec.onresult = (event) => {
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const res = event.results[i];
        const txt = res[0]?.transcript?.trim() || "";
        if (res.isFinal && txt) {
          this.onInterim("");
          this.onTranscript(txt, { engine: "webspeech", latencyMs: 0 });
        } else if (txt) {
          interim += (interim ? " " : "") + txt;
        }
      }
      if (interim) this.onInterim(interim);
    };

    rec.onerror = (e) => {
      if (e.error !== "no-speech") {
        this.onStatus(`Web Speech error: ${e.error}`, "error");
      }
    };

    rec.onend = () => {
      if (this.isListening) {
        try {
          rec.start();
        } catch {
          // Ignore rapid restart
        }
      }
    };

    this.recognition = rec;
    rec.start();
  }
}
