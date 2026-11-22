import { env, pipeline } from "@huggingface/transformers";
import { StreamingAudioBuffer, TARGET_SAMPLE_RATE } from "./audioRingBuffer.js";
import { StreamingStabilizer, computeMaxNewTokens } from "./streamingStabilizer.js";

// Run 100% locally in-browser via HuggingFace CDN cache (no custom backend server)
env.allowLocalModels = false;

const MODEL_CONFIGS = {
  moonshine: {
    id: "onnx-community/moonshine-tiny-ONNX",
    label: "Moonshine Tiny",
    webgpuDtype: { encoder_model: "fp32", decoder_model_merged: "q4" },
    wasmDtype: "q8",
    minTokens: 12,
  },
  whisper: {
    id: "onnx-community/whisper-tiny.en",
    label: "Whisper Tiny.en",
    webgpuDtype: { encoder_model: "fp32", decoder_model_merged: "q4" },
    wasmDtype: "q8",
    minTokens: 16,
  },
};

const MIN_FIRST_DECODE_MS = 380;
const STEP_CADENCE_MS = 260;
const MICRO_PAUSE_MS = 280;
const MAX_WINDOW_SEC = 4.5;
const COMPACT_KEEP_SEC = 1.4;
const MIN_NEW_SAMPLES = Math.round(TARGET_SAMPLE_RATE * 0.16); // 160ms new audio

const audioBuffer = new StreamingAudioBuffer({
  sampleRate: TARGET_SAMPLE_RATE,
  preRollMs: 180,
  maxWindowSec: MAX_WINDOW_SEC,
  compactKeepSec: COMPACT_KEEP_SEC,
});
const stabilizer = new StreamingStabilizer();

let transcriber = null;
let loadedKey = null;
let activeDevice = "wasm";
let webgpuFailed = false;

let initPromise = null;
let initTargetKey = null;
let initGeneration = 0;

let activeSessionId = 0;
let isDecoding = false;
let flushRequested = false;
let decodeTimer = null;
let lastDecodeEndMs = 0;
let lastInferenceMs = 0;
let consecutiveDecodeErrors = 0;

async function hasWebGPU() {
  if (webgpuFailed) return false;
  if (typeof navigator !== "undefined" && "gpu" in navigator) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      return Boolean(adapter);
    } catch {
      return false;
    }
  }
  return false;
}

async function disposeTranscriber() {
  if (!transcriber) return;
  const current = transcriber;
  transcriber = null;
  loadedKey = null;
  try {
    if (typeof current.dispose === "function") {
      await current.dispose();
    }
  } catch {
    // Best-effort GPU/WASM session release
  }
}

async function loadPipelineForDevice(key, device, myGen) {
  const cfg = MODEL_CONFIGS[key];
  const progress_callback = (info) => {
    if (myGen !== initGeneration) return;
    if (info && typeof info.progress === "number") {
      self.postMessage({
        type: "PROGRESS",
        progress: info.progress,
        file: info.file || "",
        label: cfg.label,
        modelKey: key,
      });
    }
  };

  if (myGen !== initGeneration) return false;
  self.postMessage({
    type: "STATUS",
    state: "loading",
    modelKey: key,
    text: `Loading ${cfg.label} on ${device.toUpperCase()}…`,
  });

  const pipe = await pipeline("automatic-speech-recognition", cfg.id, {
    device,
    dtype: device === "webgpu" ? cfg.webgpuDtype : cfg.wasmDtype,
    progress_callback,
  });

  if (myGen !== initGeneration) {
    if (pipe && typeof pipe.dispose === "function") {
      await pipe.dispose().catch(() => {});
    }
    return false;
  }

  await disposeTranscriber();
  if (myGen !== initGeneration) {
    if (pipe && typeof pipe.dispose === "function") {
      await pipe.dispose().catch(() => {});
    }
    return false;
  }

  transcriber = pipe;
  loadedKey = key;
  activeDevice = device;
  return true;
}

async function initModel(modelKey = "moonshine", forceWasm = false, fallbackReason = "") {
  const key = MODEL_CONFIGS[modelKey] ? modelKey : "moonshine";

  if (!forceWasm && transcriber && loadedKey === key && !initPromise) {
    self.postMessage({ type: "READY", device: activeDevice, modelKey: key });
    return;
  }

  // Coalesce concurrent INIT requests targeting the same model
  if (!forceWasm && initPromise && initTargetKey === key) {
    return initPromise;
  }

  const myGen = ++initGeneration;
  initTargetKey = key;
  const currentPromise = (async () => {
    await disposeTranscriber();
    if (myGen !== initGeneration) return;

    let reason = fallbackReason;
    if (!forceWasm && (await hasWebGPU())) {
      if (myGen !== initGeneration) return;
      try {
        const applied = await loadPipelineForDevice(key, "webgpu", myGen);
        if (!applied || myGen !== initGeneration) return;
        self.postMessage({ type: "READY", device: "webgpu", modelKey: key });
        return;
      } catch (gpuErr) {
        if (myGen !== initGeneration) return;
        reason = gpuErr?.message || "WebGPU unsupported op";
        webgpuFailed = true;
        self.postMessage({
          type: "STATUS",
          state: "loading",
          modelKey: key,
          text: `WebGPU fallback to WASM (${reason})…`,
        });
      }
    }

    if (myGen !== initGeneration) return;
    const applied = await loadPipelineForDevice(key, "wasm", myGen);
    if (!applied || myGen !== initGeneration) return;
    self.postMessage({
      type: "READY",
      device: "wasm",
      modelKey: key,
      ...(reason ? { fallbackReason: reason } : {}),
    });
  })();

  initPromise = currentPromise;
  try {
    await currentPromise;
  } finally {
    if (myGen === initGeneration) {
      initPromise = null;
      initTargetKey = null;
    }
  }
}

async function runInferenceWithFallback(audio) {
  const cfg = MODEL_CONFIGS[loadedKey] || MODEL_CONFIGS.moonshine;
  // CRITICAL: Explicitly pass max_new_tokens to fix @huggingface/transformers@4.3.2
  // `_call_moonshine` `<1.0s` 0-token bug (where Math.floor(len/16000)*6 === 0).
  const opts = {
    max_new_tokens: computeMaxNewTokens(audio.length, TARGET_SAMPLE_RATE, cfg.minTokens),
  };

  const t0 = performance.now();
  try {
    const result = await transcriber(audio, opts);
    return {
      text: (result?.text || "").trim(),
      inferenceMs: performance.now() - t0,
    };
  } catch (err) {
    // Automatic runtime WebGPU -> WASM fallback on first-inference shader/op failure
    if (activeDevice === "webgpu") {
      const reason = err?.message || "WebGPU runtime error";
      webgpuFailed = true;
      const retryKey = loadedKey || "moonshine";
      self.postMessage({
        type: "STATUS",
        state: "loading",
        modelKey: retryKey,
        text: `WebGPU runtime fallback to WASM (${reason})…`,
      });
      await initModel(retryKey, true, reason);
      if (!transcriber) throw err;
      const tRetry = performance.now();
      const result = await transcriber(audio, opts);
      return {
        text: (result?.text || "").trim(),
        inferenceMs: performance.now() - tRetry,
      };
    }
    throw err;
  }
}

function scheduleDecode(delayMs = 0) {
  if (decodeTimer !== null) return;
  decodeTimer = setTimeout(() => {
    decodeTimer = null;
    maybeDecode();
  }, Math.max(0, Math.round(delayMs)));
}

async function maybeDecode() {
  if (isDecoding || !transcriber) return;

  const sessionAtStart = activeSessionId;

  // Drop tiny sub-200ms noise clicks after silence without wasting a model decode
  if (!flushRequested && audioBuffer.isShortNoiseBurst(MICRO_PAUSE_MS)) {
    audioBuffer.resetUtterance();
    stabilizer.reset();
    self.postMessage({
      type: "INTERIM",
      sessionId: sessionAtStart,
      text: "",
      speechTimestampMs: audioBuffer.windowEndMs,
    });
    return;
  }

  const wasExplicitFlush = flushRequested;
  const isFinal = wasExplicitFlush || audioBuffer.shouldFinalize(MICRO_PAUSE_MS);

  if (isFinal) {
    flushRequested = false;
    const slice = audioBuffer.getWindowSlice();
    if (slice.audio.length === 0 || slice.activeSpeechMs < 150) {
      audioBuffer.resetUtterance();
      stabilizer.reset();
      if (wasExplicitFlush) {
        self.postMessage({
          type: "INTERIM",
          sessionId: sessionAtStart,
          text: "",
          isFinal: true,
          speechTimestampMs: slice.windowEndMs,
        });
      }
      return;
    }

    isDecoding = true;
    try {
      audioBuffer.markDecoded();
      const { text, inferenceMs } = await runInferenceWithFallback(slice.audio);
      consecutiveDecodeErrors = 0;
      lastInferenceMs = inferenceMs;
      lastDecodeEndMs = performance.now();

      if (sessionAtStart !== activeSessionId) return;

      const update = stabilizer.update(text, {
        isFinal: true,
        windowStartMs: slice.windowStartMs,
        windowEndMs: slice.windowEndMs,
        peakRms: slice.peakRms,
        activeSpeechMs: slice.activeSpeechMs,
      });

      audioBuffer.resetUtterance();
      stabilizer.reset();

      if (update.newWords.length > 0) {
        self.postMessage({
          type: "TRANSCRIPT",
          sessionId: sessionAtStart,
          text: update.text,
          words: update.newWords,
          modelKey: loadedKey,
          device: activeDevice,
          inferenceMs: Math.round(inferenceMs),
          audioDurationMs: Math.round(slice.durationSec * 1000),
          speechTimestampMs: update.speechTimestampMs,
          isFinal: true,
        });
      } else {
        self.postMessage({
          type: "INTERIM",
          sessionId: sessionAtStart,
          text: "",
          isFinal: true,
          speechTimestampMs: slice.windowEndMs,
        });
      }
    } catch (err) {
      audioBuffer.resetUtterance();
      stabilizer.reset();
      flushRequested = false;
      consecutiveDecodeErrors = 0;
      self.postMessage({
        type: "ERROR",
        sessionId: sessionAtStart,
        message: err?.message || String(err),
        recoverable: true,
      });
    } finally {
      isDecoding = false;
      if (flushRequested || audioBuffer.isSpeaking) {
        scheduleDecode(20);
      }
    }
    return;
  }

  // Active speech sliding-window step decode
  if (
    !audioBuffer.isSpeaking ||
    audioBuffer.activeSpeechMs < MIN_FIRST_DECODE_MS ||
    audioBuffer.newSamplesSinceLastDecode < MIN_NEW_SAMPLES
  ) {
    return;
  }

  const now = performance.now();
  const minGap = Math.max(STEP_CADENCE_MS, lastInferenceMs * 0.25);
  const elapsed = now - lastDecodeEndMs;
  if (lastDecodeEndMs > 0 && elapsed < minGap) {
    scheduleDecode(minGap - elapsed);
    return;
  }

  const slice = audioBuffer.getWindowSlice();
  if (slice.audio.length === 0) return;

  isDecoding = true;
  try {
    audioBuffer.markDecoded();
    const { text, inferenceMs } = await runInferenceWithFallback(slice.audio);
    consecutiveDecodeErrors = 0;
    lastInferenceMs = inferenceMs;
    lastDecodeEndMs = performance.now();

    if (sessionAtStart !== activeSessionId) return;

    const update = stabilizer.update(text, {
      isFinal: false,
      windowStartMs: slice.windowStartMs,
      windowEndMs: slice.windowEndMs,
      peakRms: slice.peakRms,
      activeSpeechMs: slice.activeSpeechMs,
    });

    if (update.newWords.length > 0) {
      self.postMessage({
        type: "TRANSCRIPT",
        sessionId: sessionAtStart,
        text: update.text,
        words: update.newWords,
        modelKey: loadedKey,
        device: activeDevice,
        inferenceMs: Math.round(inferenceMs),
        audioDurationMs: Math.round(slice.durationSec * 1000),
        speechTimestampMs: update.speechTimestampMs,
        isFinal: false,
      });
    }

    if (update.interimText) {
      self.postMessage({
        type: "INTERIM",
        sessionId: sessionAtStart,
        text: update.interimText,
        speechTimestampMs: slice.windowEndMs,
      });
    }

    // Slide window left during long continuous speech while retaining 1.4s overlap
    if (audioBuffer.durationSec >= MAX_WINDOW_SEC) {
      if (audioBuffer.compactLeft(COMPACT_KEEP_SEC)) {
        stabilizer.onWindowCompact();
      }
    }
  } catch (err) {
    consecutiveDecodeErrors++;
    if (consecutiveDecodeErrors >= 3) {
      audioBuffer.resetUtterance();
      stabilizer.reset();
      flushRequested = false;
      consecutiveDecodeErrors = 0;
    }
    self.postMessage({
      type: "ERROR",
      sessionId: sessionAtStart,
      message: err?.message || String(err),
      recoverable: true,
    });
  } finally {
    isDecoding = false;
    if (
      flushRequested ||
      audioBuffer.shouldFinalize(MICRO_PAUSE_MS) ||
      audioBuffer.newSamplesSinceLastDecode >= MIN_NEW_SAMPLES
    ) {
      scheduleDecode(20);
    }
  }
}

self.onmessage = async (event) => {
  const msg = event.data || {};
  const { type, modelKey, sessionId } = msg;

  try {
    if (type === "INIT") {
      if (typeof sessionId === "number" && sessionId > activeSessionId) {
        activeSessionId = sessionId;
      }
      await initModel(modelKey || "moonshine");
      maybeDecode();
    } else if (type === "RESET") {
      if (typeof sessionId === "number") {
        activeSessionId = sessionId;
      }
      if (decodeTimer !== null) {
        clearTimeout(decodeTimer);
        decodeTimer = null;
      }
      flushRequested = false;
      consecutiveDecodeErrors = 0;
      audioBuffer.resetAll();
      stabilizer.reset();
    } else if (type === "FLUSH") {
      if (typeof sessionId === "number" && sessionId !== activeSessionId) return;
      flushRequested = true;
      maybeDecode();
    } else if (type === "AUDIO_CHUNK") {
      if (typeof sessionId === "number" && sessionId !== activeSessionId) {
        if (sessionId < activeSessionId) return; // Stale chunk from previous session
        activeSessionId = sessionId;
        audioBuffer.resetAll();
        stabilizer.reset();
        flushRequested = false;
        consecutiveDecodeErrors = 0;
      }
      audioBuffer.pushChunk(msg.pcm, msg.rms, msg.captureTimestampMs);
      maybeDecode();
    }
  } catch (err) {
    self.postMessage({
      type: "ERROR",
      sessionId: typeof sessionId === "number" ? sessionId : activeSessionId,
      message: err?.message || String(err),
      recoverable: false,
    });
  }
};
