/**
 * Real-Time Streaming Speech-to-Text Controller (100% In-Browser WebAI + Optional Web Speech Toggle).
 *
 * Engines:
 *   1. "moonshine" (Default) — Offline WebAI Streaming (onnx-community/moonshine-tiny-ONNX)
 *                              running in a dedicated Web Worker with AudioWorklet PCM capture,
 *                              sliding-window adaptive VAD, and LocalAgreement-2 word commit.
 *   2. "webspeech" (Optional) — Browser Web Speech API emitting cumulative streaming hypotheses
 *                              with monotonic utteranceId & estimated speech timestamps.
 *   3. "whisper"   (Offline)  — Offline WebAI Whisper Tiny.en in the same sliding-window worker.
 */

import { createPcmCapture } from "./pcmCapture.js";

const VALID_ENGINES = new Set(["moonshine", "whisper", "webspeech"]);
const WEBSPEECH_PAUSE_FLUSH_MS = 160;

export class SpeechController {
  constructor(callbacks = {}, deps = {}) {
    this.onTranscript = callbacks.onTranscript || (() => {});
    this.onInterim = callbacks.onInterim || (() => {});
    this.onStatus = callbacks.onStatus || (() => {});
    this.onLevel = callbacks.onLevel || (() => {});
    this.onListeningChange = callbacks.onListeningChange || (() => {});

    // Injectable platform hooks for deterministic unit testing in Node
    this._createWorker =
      deps.createWorker ||
      (() =>
        new Worker(new URL("./whisper.worker.js", import.meta.url), {
          type: "module",
        }));
    this._createPcmCapture = deps.createPcmCapture || createPcmCapture;
    this._getUserMedia =
      deps.getUserMedia ||
      ((constraints) => navigator.mediaDevices.getUserMedia(constraints));
    this._createAudioContext =
      deps.createAudioContext ||
      (() => new (window.AudioContext || window.webkitAudioContext)());
    this._getSpeechRecognition =
      deps.getSpeechRecognition ||
      (() =>
        typeof window !== "undefined"
          ? window.SpeechRecognition || window.webkitSpeechRecognition || null
          : null);
    this._now = deps.now || (() => performance.now());

    this.engine = SpeechController.hasWebSpeech() ? "webspeech" : "moonshine";
    this.isListening = false;
    this.sessionId = 0;
    this._engineSwitchSeq = 0;

    // WebAI Worker + Audio capture state
    this.worker = null;
    this.loadedModelKey = null;
    this.activeDevice = "wasm";
    this.mediaStream = null;
    this.audioCtx = null;
    this.pcmCapture = null;
    this._utteranceCounter = 1;

    // Web Speech API state
    this.recognition = null;
    this.interimFlushTimer = null;
    this.webSpeechRestartTimer = null;
    this.webSpeechErrorCount = 0;
    this._webSpeechNetworkErrors = 0;
    this._webSpeechUtteranceBase = 0;
  }

  static hasWebSpeech() {
    return (
      typeof window !== "undefined" &&
      Boolean(window.SpeechRecognition || window.webkitSpeechRecognition)
    );
  }

  _setListening(next) {
    const changed = this.isListening !== next;
    this.isListening = next;
    if (changed) {
      this.onListeningChange(next);
    }
  }

  async setEngine(engine) {
    const target = VALID_ENGINES.has(engine) ? engine : "moonshine";
    if (this.engine === target) return;

    const mySwitchSeq = ++this._engineSwitchSeq;
    const wasListening = this.isListening;
    if (wasListening) {
      this.stop({ silentStatus: true });
    }
    this.engine = target;

    if (wasListening) {
      const expectedSessionId = this.sessionId + 1;
      try {
        await this.start();
      } catch (err) {
        if (
          mySwitchSeq === this._engineSwitchSeq &&
          this.engine === target &&
          this.sessionId === expectedSessionId
        ) {
          this._setListening(false);
          this.onStatus(`Mic error: ${err?.message || String(err)}`, "error");
        }
        throw err;
      }
    }
  }

  ensureWorker() {
    if (this.worker) return;
    this.worker = this._createWorker();

    this.worker.onmessage = (event) => {
      const msg = event.data || {};

      if (msg.type === "STATUS") {
        this.onStatus(msg.text, msg.state || "loading");
        return;
      }

      if (msg.type === "PROGRESS") {
        const pct = Math.round(msg.progress);
        this.onStatus(`Downloading ${msg.label || "WebAI STT"} (${pct}%)…`, "loading");
        return;
      }

      if (msg.type === "READY") {
        if (msg.modelKey && msg.modelKey !== this.engine) {
          return;
        }
        this.loadedModelKey = msg.modelKey;
        this.activeDevice = msg.device || "wasm";
        const name = msg.modelKey === "whisper" ? "Whisper" : "Moonshine";
        if (this.engine !== "webspeech") {
          this.onStatus(
            this.isListening
              ? `Live Stream · ${name} (${this.activeDevice.toUpperCase()})`
              : `${name} Ready (${this.activeDevice.toUpperCase()})`,
            "ready"
          );
        }
        return;
      }

      if (msg.type === "INTERIM") {
        if (
          !this.isListening ||
          this.engine === "webspeech" ||
          msg.sessionId !== this.sessionId
        ) {
          return;
        }
        const emitTimestampMs = this._now();
        const isFinal = Boolean(msg.isFinal);
        const utteranceId = this._utteranceCounter;
        if (isFinal) {
          this._utteranceCounter++;
        }
        this.onInterim(msg.text || "", {
          engine: this.engine,
          device: this.activeDevice,
          isFinal,
          isCumulative: false,
          utteranceId,
          speechTimestampMs: msg.speechTimestampMs ?? emitTimestampMs,
          emitTimestampMs,
          sttEstimated: false,
        });
        return;
      }

      if (msg.type === "TRANSCRIPT") {
        if (
          !this.isListening ||
          this.engine === "webspeech" ||
          msg.sessionId !== this.sessionId
        ) {
          return;
        }
        const text = (msg.text || "").trim();
        if (!text) return;

        const emitTimestampMs = this._now();
        const speechTimestampMs =
          typeof msg.speechTimestampMs === "number" && Number.isFinite(msg.speechTimestampMs)
            ? msg.speechTimestampMs
            : emitTimestampMs;
        const sttLatencyMs = Math.max(0, emitTimestampMs - speechTimestampMs);
        const utteranceId = this._utteranceCounter;
        if (msg.isFinal) {
          this._utteranceCounter++;
        }

        const meta = {
          engine: msg.modelKey || this.engine,
          device: msg.device || this.activeDevice,
          isFinal: Boolean(msg.isFinal),
          isCumulative: false,
          utteranceId,
          speechTimestampMs,
          emitTimestampMs,
          sttLatencyMs,
          sttEstimated: false,
          inferenceMs: Number(msg.inferenceMs) || 0,
          latencyMs: Math.round(sttLatencyMs),
          audioDurationMs: Number(msg.audioDurationMs) || 0,
        };

        try {
          this.onTranscript(text, meta);
        } catch (err) {
          console.error("[OpenASL STT] Transcript callback error:", err);
        }
        return;
      }

      if (msg.type === "ERROR") {
        if (typeof msg.sessionId === "number" && msg.sessionId !== this.sessionId) {
          return;
        }
        this.onStatus(`STT error: ${msg.message}`, "error");
      }
    };
  }

  async start() {
    if (this.isListening) return;
    const sessionId = ++this.sessionId;
    this.webSpeechErrorCount = 0;
    this._webSpeechNetworkErrors = 0;
    this._setListening(true);

    try {
      if (this.engine === "webspeech") {
        this.startWebSpeechStreaming(sessionId);
      } else {
        await this.startLocalWebAIMic(this.engine, sessionId);
      }
    } catch (err) {
      if (sessionId === this.sessionId) {
        this._setListening(false);
        this.onStatus("Mic Off", "idle");
      }
      throw err;
    }
  }

  stop({ silentStatus = false } = {}) {
    const nextSessionId = ++this.sessionId;
    this._setListening(false);

    clearTimeout(this.interimFlushTimer);
    clearTimeout(this.webSpeechRestartTimer);
    this.interimFlushTimer = null;
    this.webSpeechRestartTimer = null;
    this.onLevel(0);

    if (this.worker) {
      try {
        this.worker.postMessage({ type: "RESET", sessionId: nextSessionId });
      } catch {
        // Ignore
      }
    }

    if (this.pcmCapture) {
      try {
        this.pcmCapture.stop();
      } catch {
        // Ignore
      }
      this.pcmCapture = null;
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
        this.recognition.onspeechstart = null;
        this.recognition.abort();
      } catch {
        // Ignore
      }
      this.recognition = null;
    }

    if (!silentStatus) {
      this.onStatus("Mic Off", "idle");
    }
  }

  async startLocalWebAIMic(modelKey, sessionId) {
    this.ensureWorker();
    this.worker.postMessage({ type: "RESET", sessionId });
    if (this.loadedModelKey !== modelKey) {
      this.worker.postMessage({ type: "INIT", modelKey, sessionId });
    }

    const stream = await this._getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
      video: false,
    });

    // Guard against stop() or setEngine() occurring while awaiting mic permission
    if (sessionId !== this.sessionId || !this.isListening) {
      stream.getTracks().forEach((t) => t.stop());
      return;
    }

    let audioCtx = null;
    let capture = null;
    try {
      audioCtx = this._createAudioContext();
      if (audioCtx && audioCtx.state === "suspended" && typeof audioCtx.resume === "function") {
        await audioCtx.resume();
      }

      if (sessionId !== this.sessionId || !this.isListening) {
        if (audioCtx && typeof audioCtx.close === "function") {
          audioCtx.close().catch(() => {});
        }
        stream.getTracks().forEach((t) => t.stop());
        return;
      }

      capture = await this._createPcmCapture({
        audioCtx,
        mediaStream: stream,
        chunkMs: 80,
        onChunk: ({ pcm, rms, captureTimestampMs }) => {
          if (sessionId !== this.sessionId || !this.isListening) return;
          const safeRms = Number.isFinite(rms) && rms >= 0 ? rms : 0;
          this.onLevel(Math.min(1, safeRms * 10));
          this.worker.postMessage(
            {
              type: "AUDIO_CHUNK",
              sessionId,
              pcm,
              rms: safeRms,
              captureTimestampMs,
            },
            [pcm.buffer]
          );
        },
      });
    } catch (err) {
      if (audioCtx && typeof audioCtx.close === "function") {
        audioCtx.close().catch(() => {});
      }
      stream.getTracks().forEach((t) => t.stop());
      throw err;
    }

    // Re-check sessionId after async AudioWorklet module registration
    if (sessionId !== this.sessionId || !this.isListening) {
      capture.stop();
      if (audioCtx && typeof audioCtx.close === "function") {
        audioCtx.close().catch(() => {});
      }
      stream.getTracks().forEach((t) => t.stop());
      return;
    }

    this.mediaStream = stream;
    this.audioCtx = audioCtx;
    this.pcmCapture = capture;

    const name = modelKey === "whisper" ? "Whisper" : "Moonshine";
    this.onStatus(
      this.loadedModelKey === modelKey
        ? `Live Stream · ${name} (${this.activeDevice.toUpperCase()})`
        : `Loading ${name} WebAI…`,
      this.loadedModelKey === modelKey ? "ready" : "loading"
    );
  }

  /**
   * Optional Web Speech API (Instant Stream) with unified sessionId guard,
   * cumulative hypothesis emission (diffed & stabilized by StreamingRuleGloss),
   * and fast 160ms pause flush.
   */
  startWebSpeechStreaming(sessionId) {
    const SR = this._getSpeechRecognition();
    if (!SR) {
      this._setListening(false);
      this.onStatus("Web Speech API unavailable — switch to Moonshine Offline WebAI.", "error");
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
        this.recognition.onspeechstart = null;
        this.recognition.abort();
      } catch {
        // Ignore
      }
      this.recognition = null;
    }

    this._webSpeechUtteranceBase += 1000;
    const utteranceBase = this._webSpeechUtteranceBase;
    const speechStartByResult = new Map();
    let lastSpeechStartMs = 0;
    let lastPending = null; // { uttId, text, speechTimestampMs }

    const rec = new SR();
    rec.lang = "en-US";
    rec.continuous = true;
    rec.interimResults = true;
    rec.maxAlternatives = 1;

    const scheduleRestart = (delayMs, statusText) => {
      if (!this.isListening || sessionId !== this.sessionId) return;
      clearTimeout(this.webSpeechRestartTimer);
      if (statusText) this.onStatus(statusText, "loading");
      this.webSpeechRestartTimer = setTimeout(() => {
        if (this.isListening && sessionId === this.sessionId) {
          this.startWebSpeechStreaming(sessionId);
        }
      }, delayMs);
    };

    const buildMeta = (uttId, isFinal, speechTimestampMs) => {
      const emitTimestampMs = this._now();
      const anchorMs = speechTimestampMs || emitTimestampMs;
      const sttLatencyMs = Math.max(0, emitTimestampMs - anchorMs);
      return {
        engine: "webspeech",
        device: "browser",
        isFinal,
        isCumulative: true,
        utteranceId: uttId,
        speechTimestampMs: anchorMs,
        emitTimestampMs,
        sttLatencyMs,
        sttEstimated: true,
        inferenceMs: 0,
        latencyMs: Math.round(sttLatencyMs),
      };
    };

    const emitFinal = (uttId, text, speechTimestampMs) => {
      if (!this.isListening || sessionId !== this.sessionId || !text) return;
      const meta = buildMeta(uttId, true, speechTimestampMs);
      try {
        this.onTranscript(text, meta);
      } catch (err) {
        console.error("[OpenASL STT] Transcript callback error:", err);
      }
    };

    rec.onstart = () => {
      if (sessionId !== this.sessionId) return;
      this.onStatus("Live Stream · Web Speech (Browser)", "ready");
    };

    rec.onspeechstart = () => {
      if (sessionId !== this.sessionId) return;
      lastSpeechStartMs = this._now();
    };

    rec.onresult = (event) => {
      if (sessionId !== this.sessionId || !this.isListening) return;
      this.webSpeechErrorCount = 0;
      this._webSpeechNetworkErrors = 0;
      clearTimeout(this.interimFlushTimer);

      const now = this._now();

      // If browser jumped resultIndex past a previous pending interim, finalize it first
      if (lastPending && utteranceBase + event.resultIndex > lastPending.uttId) {
        emitFinal(lastPending.uttId, lastPending.text, lastPending.speechTimestampMs);
        lastPending = null;
      }

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const res = event.results[i];
        const txt = res[0]?.transcript?.trim() || "";
        if (!txt) continue;

        const uttId = utteranceBase + i;
        if (!speechStartByResult.has(uttId)) {
          const estStart =
            lastSpeechStartMs > 0 && now - lastSpeechStartMs < 4000
              ? lastSpeechStartMs
              : Math.max(0, now - 120);
          speechStartByResult.set(uttId, estStart);
        }
        const speechAnchorMs = speechStartByResult.get(uttId);

        if (res.isFinal) {
          emitFinal(uttId, txt, speechAnchorMs);
          if (lastPending && lastPending.uttId === uttId) lastPending = null;
          this.onInterim("", buildMeta(uttId, true, speechAnchorMs));
        } else {
          // Skip duplicate identical interim string for the same utterance
          if (lastPending && lastPending.uttId === uttId && lastPending.text === txt) {
            continue;
          }
          const meta = buildMeta(uttId, false, speechAnchorMs);
          try {
            this.onInterim(txt, meta);
          } catch (err) {
            console.error("[OpenASL STT] Interim callback error:", err);
          }
          lastPending = { uttId, text: txt, speechTimestampMs: speechAnchorMs };
        }
      }

      if (lastPending && lastPending.text) {
        const snapshot = lastPending;
        this.interimFlushTimer = setTimeout(() => {
          if (sessionId !== this.sessionId || !this.isListening) return;
          emitFinal(snapshot.uttId, snapshot.text, snapshot.speechTimestampMs);
          if (lastPending === snapshot) lastPending = null;
        }, WEBSPEECH_PAUSE_FLUSH_MS);
      }
    };

    rec.onerror = (e) => {
      if (sessionId !== this.sessionId) return;
      if (e.error === "not-allowed") {
        this._setListening(false);
        this.onStatus("Mic permission blocked by browser.", "error");
        return;
      }
      if (e.error === "service-not-allowed") {
        this._setListening(false);
        this.onStatus(
          "Web Speech unavailable (cloud service blocked) — switch to Moonshine Offline WebAI.",
          "error"
        );
        return;
      }
      if (e.error === "network") {
        this._webSpeechNetworkErrors++;
        this.webSpeechErrorCount++;
        if (this._webSpeechNetworkErrors >= 2) {
          clearTimeout(this.webSpeechRestartTimer);
          this.webSpeechRestartTimer = null;
          this._setListening(false);
          this.onStatus(
            "Web Speech unavailable (cloud service blocked) — switch to Moonshine Offline WebAI.",
            "error"
          );
          return;
        }
        this.onStatus("Web Speech reconnecting (network)…", "loading");
        return;
      }
      if (e.error !== "no-speech" && e.error !== "aborted") {
        this.webSpeechErrorCount++;
        this.onStatus(`Web Speech reconnecting (${e.error})…`, "loading");
      }
    };

    rec.onend = () => {
      if (sessionId !== this.sessionId) return;
      clearTimeout(this.interimFlushTimer);
      if (lastPending) {
        emitFinal(lastPending.uttId, lastPending.text, lastPending.speechTimestampMs);
        lastPending = null;
      }
      if (this.isListening) {
        const delay = Math.min(2000, 260 * Math.pow(1.6, this.webSpeechErrorCount));
        scheduleRestart(delay, this.webSpeechErrorCount > 0 ? "Reconnecting Web Speech…" : null);
      }
    };

    this.recognition = rec;
    try {
      rec.start();
    } catch {
      this.webSpeechErrorCount++;
      scheduleRestart(Math.min(2000, 350 * this.webSpeechErrorCount), "Restarting Web Speech…");
    }
  }
}
