import { env, pipeline } from "@huggingface/transformers";

// Run 100% locally in-browser via HuggingFace CDN cache (no custom backend server)
env.allowLocalModels = false;

const MODEL_CONFIGS = {
  moonshine: {
    id: "onnx-community/moonshine-tiny-ONNX",
    label: "Moonshine Tiny",
    webgpuDtype: { encoder_model: "fp32", decoder_model_merged: "q4" },
    wasmDtype: "q8",
  },
  whisper: {
    id: "onnx-community/whisper-tiny.en",
    label: "Whisper Tiny.en",
    webgpuDtype: { encoder_model: "fp32", decoder_model_merged: "q4" },
    wasmDtype: "q8",
  },
};

let transcriber = null;
let loadedKey = null;
let activeDevice = "wasm";

async function hasWebGPU() {
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

async function initModel(modelKey = "moonshine") {
  const key = MODEL_CONFIGS[modelKey] ? modelKey : "moonshine";
  if (transcriber && loadedKey === key) {
    self.postMessage({ type: "READY", device: activeDevice, modelKey: key });
    return;
  }

  const cfg = MODEL_CONFIGS[key];
  const progress_callback = (info) => {
    if (info && typeof info.progress === "number") {
      self.postMessage({
        type: "PROGRESS",
        progress: info.progress,
        file: info.file || "",
        label: cfg.label,
      });
    }
  };

  // Prefer WebGPU when supported, with automatic fallback to quantized WASM
  const gpuAvailable = await hasWebGPU();
  if (gpuAvailable) {
    try {
      activeDevice = "webgpu";
      self.postMessage({
        type: "STATUS",
        text: `Loading ${cfg.label} on WebGPU…`,
      });
      transcriber = await pipeline("automatic-speech-recognition", cfg.id, {
        device: "webgpu",
        dtype: cfg.webgpuDtype,
        progress_callback,
      });
      loadedKey = key;
      self.postMessage({ type: "READY", device: activeDevice, modelKey: key });
      return;
    } catch (gpuErr) {
      self.postMessage({
        type: "STATUS",
        text: `WebGPU fallback to WASM (${gpuErr?.message || "unsupported op"})…`,
      });
    }
  }

  activeDevice = "wasm";
  self.postMessage({
    type: "STATUS",
    text: `Loading ${cfg.label} on WASM…`,
  });
  transcriber = await pipeline("automatic-speech-recognition", cfg.id, {
    device: "wasm",
    dtype: cfg.wasmDtype,
    progress_callback,
  });
  loadedKey = key;
  self.postMessage({ type: "READY", device: activeDevice, modelKey: key });
}

function isHallucination(text) {
  const cleaned = text.trim().toLowerCase();
  if (!cleaned || cleaned.length < 2) return true;
  if (/^[\[\(].*[\]\)]$/.test(cleaned)) return true; // [BLANK_AUDIO], (music), (silence)
  const noise = new Set([
    "thank you.",
    "thank you",
    "thanks for watching.",
    "thanks for watching",
    "you",
    "bye.",
    "bye",
    "hmm",
    "uh",
    "um",
    "oh",
    "okay.",
    "the",
    "a",
  ]);
  return noise.has(cleaned);
}

self.onmessage = async (event) => {
  const { type, audio, modelKey } = event.data || {};
  try {
    if (type === "INIT") {
      await initModel(modelKey || "moonshine");
    } else if (type === "TRANSCRIBE") {
      const targetKey = modelKey || loadedKey || "moonshine";
      if (!transcriber || loadedKey !== targetKey) {
        await initModel(targetKey);
      }

      const t0 = performance.now();
      // IMPORTANT: Never pass `language` or `task` to English-only models (moonshine-tiny / whisper-tiny.en)
      const result = await transcriber(audio);
      const text = (result?.text || "").trim();

      if (!isHallucination(text)) {
        self.postMessage({
          type: "TRANSCRIPT",
          text,
          modelKey: loadedKey,
          latencyMs: Math.round(performance.now() - t0),
        });
      } else {
        self.postMessage({ type: "IGNORED" });
      }
    }
  } catch (err) {
    self.postMessage({ type: "ERROR", message: err?.message || String(err) });
  }
};
