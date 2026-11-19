import { env, pipeline } from "@huggingface/transformers";

// Always run locally in browser via HuggingFace CDN cache (no custom backend server)
env.allowLocalModels = false;

let transcriber = null;
let activeDevice = "wasm";

async function detectDevice() {
  if (typeof navigator !== "undefined" && "gpu" in navigator) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) return "webgpu";
    } catch {
      // Fallback to WASM CPU
    }
  }
  return "wasm";
}

async function initWhisper(modelId = "onnx-community/whisper-tiny.en") {
  if (transcriber) {
    self.postMessage({ type: "READY", device: activeDevice, modelId });
    return;
  }

  activeDevice = await detectDevice();
  self.postMessage({
    type: "STATUS",
    text: `Loading ${modelId} on ${activeDevice.toUpperCase()}…`,
  });

  transcriber = await pipeline("automatic-speech-recognition", modelId, {
    device: activeDevice,
    dtype: activeDevice === "webgpu" ? { encoder_model: "fp32", decoder_model_merged: "q4" } : "q8",
    progress_callback: (info) => {
      if (info && typeof info.progress === "number") {
        self.postMessage({
          type: "PROGRESS",
          progress: info.progress,
          file: info.file || "",
        });
      }
    },
  });

  self.postMessage({ type: "READY", device: activeDevice, modelId });
}

function isWhisperHallucination(text) {
  const cleaned = text.trim().toLowerCase();
  if (!cleaned || cleaned.length < 2) return true;
  if (/^[\[\(].*[\]\)]$/.test(cleaned)) return true; // [BLANK_AUDIO], (music), (laughter)
  const hallucinations = new Set([
    "thank you.",
    "thank you",
    "thanks for watching.",
    "thanks for watching",
    "you",
    "bye.",
    "bye",
    "hmm",
    "uh",
  ]);
  return hallucinations.has(cleaned);
}

self.onmessage = async (event) => {
  const { type, audio, modelId } = event.data || {};
  try {
    if (type === "INIT") {
      await initWhisper(modelId);
    } else if (type === "TRANSCRIBE") {
      if (!transcriber) {
        await initWhisper(modelId);
      }
      const t0 = performance.now();
      const result = await transcriber(audio, {
        language: "english",
        task: "transcribe",
        chunk_length_s: 15,
        stride_length_s: 2,
      });
      const text = (result?.text || "").trim();
      if (!isWhisperHallucination(text)) {
        self.postMessage({
          type: "TRANSCRIPT",
          text,
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
