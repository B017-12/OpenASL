import { createStreamingGloss, translateRuleGloss } from "./nlp/ruleGloss.js";
import {
  WEBLLM_MODELS,
  WebLLMGlossEngine,
  checkWebGPUSupport,
} from "./nlp/webllmGloss.js";
import { ASLAssetStore } from "./player/assetLoader.js";
import { ASLStagePlayer } from "./player/aslStage.js";
import { VirtualCamClient } from "./player/virtualCamClient.js";
import { SpeechController } from "./stt/speechController.js";
import { MetricsCollector } from "./diagnostics/metricsCollector.js";
import { DiagnosticsHud } from "./diagnostics/diagnosticsHud.js";

const PREFETCH_HIGH_FREQ_GLOSSES = [
  "I", "YOU", "HELLO", "YES", "NO", "GOOD", "BAD", "WANT", "NEED", "GO",
  "WORK", "NOT", "THANKYOU", "PLEASE", "SORRY", "HELP", "WHAT", "WHERE",
  "WHEN", "WHO", "HOW", "WHY", "MY", "YOUR", "WE", "THEY", "LIKE", "KNOW",
  "TIME", "TODAY",
];

const els = {
  assetStatusBadge: document.getElementById("assetStatusBadge"),
  webgpuBadge: document.getElementById("webgpuBadge"),
  vcamStatusBadge: document.getElementById("vcamStatusBadge"),
  webcamSource: document.getElementById("webcamSource"),
  micToggleBtn: document.getElementById("micToggleBtn"),
  sttEngineSelect: document.getElementById("sttEngineSelect"),
  micLevelFill: document.getElementById("micLevelFill"),
  webcamToggleBtn: document.getElementById("webcamToggleBtn"),
  vcamToggleBtn: document.getElementById("vcamToggleBtn"),
  sttStatusText: document.getElementById("sttStatusText"),
  liveTranscriptText: document.getElementById("liveTranscriptText"),
  canvas: document.getElementById("aslStageCanvas"),
  playPauseBtn: document.getElementById("playPauseBtn"),
  replayBtn: document.getElementById("replayBtn"),
  loopToggle: document.getElementById("loopToggle"),
  speedSlider: document.getElementById("speedSlider"),
  speedValue: document.getElementById("speedValue"),
  glossTimeline: document.getElementById("glossTimeline"),
  glossSummary: document.getElementById("glossSummary"),
  translateForm: document.getElementById("translateForm"),
  englishInput: document.getElementById("englishInput"),
  translateBtn: document.getElementById("translateBtn"),
  engineLatency: document.getElementById("engineLatency"),
  modeRule: document.getElementById("modeRule"),
  modeWebLLM: document.getElementById("modeWebLLM"),
  webllmPanel: document.getElementById("webllmPanel"),
  webllmModelSelect: document.getElementById("webllmModelSelect"),
  loadWebLLMBtn: document.getElementById("loadWebLLMBtn"),
  webllmProgressWrap: document.getElementById("webllmProgressWrap"),
  webllmProgressBar: document.getElementById("webllmProgressBar"),
  webllmStatusText: document.getElementById("webllmStatusText"),
  slotTime: document.getElementById("slotTime"),
  slotTopic: document.getElementById("slotTopic"),
  slotComment: document.getElementById("slotComment"),
  slotEnd: document.getElementById("slotEnd"),
  slotDropped: document.getElementById("slotDropped"),
  poseGlossInput: document.getElementById("poseGlossInput"),
  poseDropzone: document.getElementById("poseDropzone"),
  poseFileInput: document.getElementById("poseFileInput"),
  localPoseList: document.getElementById("localPoseList"),
};

const assetStore = new ASLAssetStore();
const webllmEngine = new WebLLMGlossEngine();
const metrics = new MetricsCollector();
const diagnosticsHud = new DiagnosticsHud(metrics);
const streamingGloss = createStreamingGloss();

let timelinePoseCount = 0;
let timelineSpellCount = 0;
let renderedTimelineFirst = null;
let renderedTimelineCount = 0;

const stagePlayer = new ASLStagePlayer(els.canvas, assetStore, {
  onStateChange: (state) => {
    metrics.setQueueDepth(state.queueDepth, state.isPlaying);
    syncPlaybackUI(state);
  },
  onFrame: (now) => {
    metrics.recordFrame(now);
  },
  onTokensResolved: (resolvedItems) => {
    metrics.recordTokenResolution(resolvedItems);
  },
  onTokenRendered: (item, now) => {
    const trace = item?._trace;
    if (!trace || trace.source !== "speech") return;
    if (typeof trace.speechTimestampMs !== "number" || !Number.isFinite(trace.speechTimestampMs)) {
      return;
    }
    const speechToSignMs = Math.max(0, now - trace.speechTimestampMs);
    const queueWaitMs =
      typeof trace.enqueuedAtMs === "number" ? Math.max(0, now - trace.enqueuedAtMs) : 0;
    const pipelineMs =
      typeof trace.enqueuedAtMs === "number"
        ? Math.max(0, trace.enqueuedAtMs - trace.speechTimestampMs)
        : Math.max(0, speechToSignMs - queueWaitMs);

    metrics.recordEndToEnd({
      speechToSignMs,
      pipelineMs,
      queueWaitMs,
      gloss: item.gloss,
      estimated: Boolean(trace.sttEstimated),
    });
  },
});

let webcamStream = null;

const vcamClient = new VirtualCamClient(els.canvas, (state, label) => {
  els.vcamStatusBadge.textContent = label;
  els.vcamStatusBadge.classList.toggle("is-ready", state === "connected");
  metrics.setContext({ vcamState: state });
});

async function enqueueLiveTokens(res, meta = {}) {
  if (res.tokens.length > 0 || res.committedWords.length > 0) {
    metrics.recordGloss({
      mode: "stream-pse",
      glossMs: res.glossMs,
      semanticMs: 0,
      inputWords: res.committedWords.length,
      outputTokens: res.tokens.length,
    });
    els.engineLatency.textContent = `Stream · ${res.glossMs.toFixed(2)}ms`;
  }

  if (res.tokens.length === 0) return;

  // Record estimated STT latency on cumulative interim emissions that trigger signing
  if (meta.isCumulative && !meta.isFinal) {
    metrics.recordStt({
      engine: meta.engine || "webspeech",
      device: meta.device || "browser",
      latencyMs: meta.sttLatencyMs ?? meta.latencyMs ?? 0,
      inferenceMs: 0,
      audioDurationMs: 0,
      estimated: true,
      isFinal: false,
    });
  }

  const trace = {
    source: "speech",
    engine: meta.engine || speechController.engine,
    speechTimestampMs: meta.speechTimestampMs ?? performance.now(),
    sttLatencyMs: meta.sttLatencyMs ?? meta.latencyMs ?? 0,
    sttEstimated: Boolean(meta.sttEstimated),
    glossMs: res.glossMs,
  };

  const items = await stagePlayer.appendTokens(res.tokens, trace);

  if (
    renderedTimelineCount > 0 &&
    items[0] === renderedTimelineFirst &&
    items.length >= renderedTimelineCount
  ) {
    if (items.length > renderedTimelineCount) {
      appendTimelineChips(items.slice(renderedTimelineCount), renderedTimelineCount);
    }
  } else {
    renderTimeline(items);
  }

  syncPlaybackUI({
    isPlaying: stagePlayer.isPlaying,
    activeIndex: stagePlayer.activeIndex,
    totalItems: items.length,
  });
}

async function handleLiveSpeech(text, meta = {}) {
  const res = meta.isCumulative
    ? streamingGloss.ingestHypothesis(text, {
        isFinal: Boolean(meta.isFinal),
        utteranceId: meta.utteranceId,
        speechTimestampMs: meta.speechTimestampMs,
        emitTimestampMs: meta.emitTimestampMs,
      })
    : streamingGloss.ingestDelta(text, {
        isFinal: Boolean(meta.isFinal),
        utteranceId: meta.utteranceId,
        speechTimestampMs: meta.speechTimestampMs,
        emitTimestampMs: meta.emitTimestampMs,
      });

  await enqueueLiveTokens(res, meta);
}

const speechController = new SpeechController({
  onStatus: (text, state) => {
    els.sttStatusText.textContent = text;
    els.sttStatusText.classList.toggle("is-live", state === "ready");
    metrics.setContext({
      sttEngine: speechController.engine,
      sttDevice:
        speechController.engine === "webspeech"
          ? "browser"
          : speechController.activeDevice || "pending",
    });
  },
  onListeningChange: (isListening) => {
    els.micToggleBtn.textContent = isListening ? "Stop Mic" : "Start Mic";
    els.micToggleBtn.classList.toggle("is-active", isListening);
    if (!isListening) {
      streamingGloss.reset();
    }
  },
  onInterim: async (text, meta = {}) => {
    if (text) {
      els.liveTranscriptText.textContent = text;
    }
    if (meta.isCumulative && text) {
      try {
        await handleLiveSpeech(text, { ...meta, isFinal: false });
      } catch (err) {
        console.error("[OpenASL] Interim translation error recovered:", err);
      }
    } else if (meta.isFinal && !meta.isCumulative) {
      try {
        const flushed = streamingGloss.flush({
          isFinal: true,
          utteranceId: meta.utteranceId,
          speechTimestampMs: meta.speechTimestampMs,
          emitTimestampMs: meta.emitTimestampMs,
        });
        await enqueueLiveTokens(flushed, meta);
      } catch (err) {
        console.error("[OpenASL] Utterance flush error recovered:", err);
      }
    }
  },
  onLevel: (lvl) => {
    els.micLevelFill.style.width = `${Math.round(lvl * 100)}%`;
  },
  onTranscript: async (text, meta = {}) => {
    try {
      const latLabel =
        typeof meta.latencyMs === "number" && meta.latencyMs > 0
          ? ` · ${meta.latencyMs}ms${meta.sttEstimated ? " est" : ""}`
          : "";
      els.liveTranscriptText.textContent = `"${text}" (${meta.engine || speechController.engine}${latLabel})`;
      els.englishInput.value = text;

      metrics.recordStt({
        engine: meta.engine || speechController.engine,
        device: meta.device || speechController.activeDevice || "browser",
        latencyMs: meta.sttLatencyMs ?? meta.latencyMs ?? 0,
        inferenceMs: meta.inferenceMs ?? 0,
        audioDurationMs: meta.audioDurationMs ?? 0,
        estimated: Boolean(meta.sttEstimated),
        isFinal: Boolean(meta.isFinal),
      });

      await handleLiveSpeech(text, meta);
    } catch (err) {
      console.error("[OpenASL] Translation error recovered:", err);
    }
  },
});

function populateModels() {
  for (const model of WEBLLM_MODELS) {
    const opt = document.createElement("option");
    opt.value = model.id;
    opt.textContent = model.label;
    els.webllmModelSelect.appendChild(opt);
  }
}

function syncPlaybackUI(state) {
  const hasItems = state.totalItems > 0;
  els.playPauseBtn.disabled = !hasItems;
  els.replayBtn.disabled = !hasItems;
  els.playPauseBtn.textContent = state.isPlaying ? "Pause" : "Play";

  const chips = els.glossTimeline.querySelectorAll(".gloss-chip");
  chips.forEach((chip, idx) => {
    chip.classList.toggle("is-active", idx === state.activeIndex);
  });
}

function createTimelineChip(item, idx) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "gloss-chip";
  btn.setAttribute("role", "listitem");
  btn.addEventListener("click", () => stagePlayer.seekToToken(idx, true, false));

  const wordSpan = document.createElement("span");
  wordSpan.textContent = item.gloss;

  const kindBadge = document.createElement("span");
  kindBadge.className = `chip-kind ${item.type === "POSE" ? "pose" : "spell"}`;
  kindBadge.textContent = item.type === "POSE" ? "POSE" : "SPELL";

  btn.append(wordSpan, kindBadge);
  return btn;
}

function updateTimelineSummary() {
  els.glossSummary.textContent = `${timelinePoseCount} pose sign${
    timelinePoseCount === 1 ? "" : "s"
  } · ${timelineSpellCount} fingerspelled`;
}

function appendTimelineChips(newItems, startIndex) {
  if (!newItems || newItems.length === 0) return;
  const emptyHint = els.glossTimeline.querySelector(".empty-hint");
  if (emptyHint) emptyHint.remove();

  const frag = document.createDocumentFragment();
  newItems.forEach((item, offset) => {
    if (item.type === "POSE") timelinePoseCount++;
    else timelineSpellCount++;
    frag.appendChild(createTimelineChip(item, startIndex + offset));
  });
  els.glossTimeline.appendChild(frag);
  if (renderedTimelineCount === 0 && startIndex === 0) {
    renderedTimelineFirst = newItems[0];
  }
  renderedTimelineCount = startIndex + newItems.length;
  updateTimelineSummary();
}

function renderTimeline(items) {
  els.glossTimeline.innerHTML = "";
  timelinePoseCount = 0;
  timelineSpellCount = 0;
  renderedTimelineFirst = items && items.length > 0 ? items[0] : null;
  renderedTimelineCount = items ? items.length : 0;

  if (!items || items.length === 0) {
    els.glossTimeline.innerHTML =
      '<p class="empty-hint">No sign tokens found for this input.</p>';
    els.glossSummary.textContent = "";
    return;
  }

  const frag = document.createDocumentFragment();
  items.forEach((item, idx) => {
    if (item.type === "POSE") timelinePoseCount++;
    else timelineSpellCount++;
    frag.appendChild(createTimelineChip(item, idx));
  });

  els.glossTimeline.appendChild(frag);
  updateTimelineSummary();
}

function renderBreakdown(result) {
  const b = result.breakdown || {};
  const fmt = (arr) => (arr && arr.length > 0 ? arr.join(" ") : "—");
  els.slotTime.textContent = fmt(b.time);
  els.slotTopic.textContent = fmt(b.topic);
  els.slotComment.textContent = fmt(b.comment);
  const endCombined = [...(b.neg || []), ...(b.wh || [])];
  els.slotEnd.textContent = fmt(endCombined);
  els.slotDropped.textContent = fmt(b.dropped);
}

async function runTranslation({ append = false, overrideText = null } = {}) {
  const text = (overrideText !== null ? overrideText : els.englishInput.value).trim();
  const useWebLLM = els.modeWebLLM.checked && webllmEngine.isReady;

  if (!append) {
    streamingGloss.reset();
  }

  const t0 = performance.now();
  const result = useWebLLM
    ? await webllmEngine.translate(text)
    : translateRuleGloss(text, {
        mode: "ttc",
        knownGlosses: assetStore.getKnownGlosses(),
      });
  const elapsedMs = performance.now() - t0;

  const formattedMs = elapsedMs < 10 ? elapsedMs.toFixed(1) : String(Math.round(elapsedMs));
  els.engineLatency.textContent = `${result.engine === "webllm" ? "WebLLM" : "Rule"} · ${formattedMs}ms`;

  metrics.recordGloss({
    mode: result.engine === "webllm" ? "webllm" : "batch-ttc",
    glossMs: elapsedMs,
    semanticMs: 0,
    inputWords: text ? text.split(/\s+/).length : 0,
    outputTokens: result.tokens.length,
  });

  renderBreakdown(result);

  const items = append
    ? await stagePlayer.appendTokens(result.tokens)
    : await stagePlayer.setTokens(result.tokens);

  renderTimeline(items);
  syncPlaybackUI({
    isPlaying: stagePlayer.isPlaying,
    activeIndex: stagePlayer.activeIndex,
    totalItems: items.length,
  });
}

async function toggleWebcam() {
  if (webcamStream) {
    webcamStream.getTracks().forEach((t) => t.stop());
    webcamStream = null;
    els.webcamSource.srcObject = null;
    stagePlayer.setWebcamVideo(null);
    stagePlayer.setViewMode("stage");
    metrics.setContext({ viewMode: "stage" });
    els.webcamToggleBtn.textContent = "Enable Webcam PiP";
    els.webcamToggleBtn.classList.remove("is-active");
    return;
  }

  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
      audio: false,
    });
    els.webcamSource.srcObject = webcamStream;
    await els.webcamSource.play();
    stagePlayer.setWebcamVideo(els.webcamSource);
    stagePlayer.setViewMode("webcam");
    metrics.setContext({ viewMode: "webcam" });
    els.webcamToggleBtn.textContent = "Disable Webcam PiP";
    els.webcamToggleBtn.classList.add("is-active");

    // Automatically stream the composited canvas to the Virtual Camera bridge
    if (!vcamClient.enabled) {
      toggleVirtualCam();
    }
  } catch (err) {
    alert(`Could not access webcam: ${err.message}`);
  }
}

function toggleVirtualCam() {
  if (vcamClient.enabled) {
    vcamClient.stop();
    els.vcamToggleBtn.textContent = "Stream Virtual Cam";
    els.vcamToggleBtn.classList.remove("is-active");
  } else {
    vcamClient.start(() => stagePlayer.tickExternal());
    els.vcamToggleBtn.textContent = "Stop Virtual Cam";
    els.vcamToggleBtn.classList.add("is-active");
  }
}

async function toggleMic() {
  if (speechController.isListening) {
    speechController.stop();
    return;
  }
  try {
    streamingGloss.reset();
    await speechController.setEngine(els.sttEngineSelect.value);
    metrics.setContext({ sttEngine: els.sttEngineSelect.value });
    await speechController.start();
  } catch (err) {
    alert(`Could not access microphone: ${err.message}`);
  }
}

function renderLocalPoseList() {
  const list = assetStore.listLocalPoses();
  els.localPoseList.innerHTML = "";
  for (const item of list) {
    const li = document.createElement("li");
    li.className = "override-item";

    const label = document.createElement("span");
    label.textContent = `${item.gloss} (${item.frameCount}f)`;

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "override-remove";
    removeBtn.textContent = "Remove";
    removeBtn.addEventListener("click", async () => {
      assetStore.clearLocalPose(item.gloss);
      streamingGloss.setKnownGlosses(assetStore.getKnownGlosses());
      renderLocalPoseList();
      await runTranslation();
    });

    li.append(label, removeBtn);
    els.localPoseList.appendChild(li);
  }
}

async function handlePoseFiles(files) {
  const file = files?.[0];
  if (!file) return;

  try {
    const { gloss } = await assetStore.registerLocalPoseFile(
      file,
      els.poseGlossInput.value.trim()
    );
    els.poseGlossInput.value = gloss;
    streamingGloss.setKnownGlosses(assetStore.getKnownGlosses());
    renderLocalPoseList();
    await runTranslation();
  } catch (err) {
    alert(err.message);
  }
}

function bindEvents() {
  diagnosticsHud.bindShortcut(window);

  els.micToggleBtn.addEventListener("click", toggleMic);
  els.sttEngineSelect.addEventListener("change", async (e) => {
    const nextEngine = e.target.value;
    streamingGloss.reset();
    metrics.setContext({
      sttEngine: nextEngine,
      sttDevice: nextEngine === "webspeech" ? "browser" : speechController.activeDevice || "pending",
    });
    try {
      await speechController.setEngine(nextEngine);
    } catch (err) {
      console.error("[OpenASL] STT engine switch failed:", err);
    }
  });
  els.webcamToggleBtn.addEventListener("click", toggleWebcam);
  els.vcamToggleBtn.addEventListener("click", toggleVirtualCam);

  els.translateForm.addEventListener("submit", (e) => {
    e.preventDefault();
    runTranslation();
  });

  els.englishInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      runTranslation();
    }
  });

  document.querySelectorAll(".preset-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      els.englishInput.value = chip.dataset.phrase || "";
      runTranslation();
    });
  });

  els.playPauseBtn.addEventListener("click", () => stagePlayer.togglePlay());
  els.replayBtn.addEventListener("click", () => stagePlayer.replay());
  els.loopToggle.addEventListener("change", (e) =>
    stagePlayer.setLoop(e.target.checked)
  );

  els.speedSlider.addEventListener("input", (e) => {
    const val = Number(e.target.value);
    els.speedValue.textContent = `${val.toFixed(1)}x`;
    stagePlayer.setSpeed(val);
  });

  const onModeChange = () => {
    els.webllmPanel.hidden = !els.modeWebLLM.checked;
    metrics.setContext({ glossMode: els.modeWebLLM.checked ? "webllm" : "stream-pse" });
  };
  els.modeRule.addEventListener("change", onModeChange);
  els.modeWebLLM.addEventListener("change", onModeChange);

  els.loadWebLLMBtn.addEventListener("click", async () => {
    const modelId = els.webllmModelSelect.value;
    els.loadWebLLMBtn.disabled = true;
    els.webllmProgressWrap.hidden = false;
    try {
      await webllmEngine.load(modelId, (report) => {
        const pct = Math.round((report.progress || 0) * 100);
        els.webllmProgressBar.style.width = `${pct}%`;
        els.webllmStatusText.textContent = report.text || `${pct}%`;
      });
      els.loadWebLLMBtn.textContent = "Model Ready";
      els.webllmProgressBar.style.width = "100%";
      await runTranslation();
    } catch (err) {
      els.webllmStatusText.textContent = `Error: ${err.message}`;
      els.loadWebLLMBtn.disabled = false;
    }
  });

  els.poseFileInput.addEventListener("change", (e) => {
    handlePoseFiles(e.target.files);
    e.target.value = "";
  });

  els.poseDropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    els.poseDropzone.classList.add("is-dragover");
  });
  els.poseDropzone.addEventListener("dragleave", () => {
    els.poseDropzone.classList.remove("is-dragover");
  });
  els.poseDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    els.poseDropzone.classList.remove("is-dragover");
    handlePoseFiles(e.dataTransfer?.files);
  });
}

async function bootstrap() {
  populateModels();
  bindEvents();
  stagePlayer.startLoop();

  if (!SpeechController.hasWebSpeech()) {
    const wsOpt = els.sttEngineSelect.querySelector('option[value="webspeech"]');
    if (wsOpt) wsOpt.disabled = true;
    els.sttEngineSelect.value = "moonshine";
  }

  metrics.setContext({
    sttEngine: els.sttEngineSelect.value,
    glossMode: "stream-pse",
    viewMode: "stage",
  });

  const [assetSummary, gpuStatus] = await Promise.all([
    assetStore.init(),
    checkWebGPUSupport(),
  ]);

  streamingGloss.setKnownGlosses(assetStore.getKnownGlosses());
  metrics.setContext({
    vocabPoseCount: assetSummary.poseCount,
    sttDevice: gpuStatus.supported ? "webgpu" : "wasm",
  });

  // Warm-cache top ~30 conversational poses asynchronously in the background
  assetStore.prefetchPoses(PREFETCH_HIGH_FREQ_GLOSSES).catch(() => {});

  els.assetStatusBadge.textContent = `${assetSummary.letterCount} A–Z · ${assetSummary.poseCount} Word Poses`;
  els.assetStatusBadge.classList.add("is-ready");

  if (gpuStatus.supported) {
    els.webgpuBadge.textContent = "WebGPU Available";
    els.webgpuBadge.classList.add("is-ready");
  } else {
    els.webgpuBadge.textContent = "WASM / Rule Mode";
    els.modeWebLLM.disabled = true;
  }

  await runTranslation();

  // Auto-connect to local Virtual Camera bridge on startup so Unity Video Capture always shows the live canvas
  if (!vcamClient.enabled) {
    toggleVirtualCam();
  }
}

bootstrap();
