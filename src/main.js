import { translateRuleGloss } from "./nlp/ruleGloss.js";
import {
  WEBLLM_MODELS,
  WebLLMGlossEngine,
  checkWebGPUSupport,
} from "./nlp/webllmGloss.js";
import { ASLAssetStore } from "./player/assetLoader.js";
import { ASLStagePlayer } from "./player/aslStage.js";
import { VirtualCamClient } from "./player/virtualCamClient.js";
import { SpeechController } from "./stt/speechController.js";

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
const stagePlayer = new ASLStagePlayer(els.canvas, assetStore, {
  onStateChange: syncPlaybackUI,
});

let webcamStream = null;

const vcamClient = new VirtualCamClient(els.canvas, (state, label) => {
  els.vcamStatusBadge.textContent = label;
  els.vcamStatusBadge.classList.toggle("is-ready", state === "connected");
});

const speechController = new SpeechController({
  onStatus: (text, state) => {
    els.sttStatusText.textContent = text;
    els.sttStatusText.classList.toggle("is-live", state === "ready");
  },
  onInterim: (text) => {
    if (text) els.liveTranscriptText.textContent = text;
  },
  onLevel: (lvl) => {
    els.micLevelFill.style.width = `${Math.round(lvl * 100)}%`;
  },
  onTranscript: async (text, meta) => {
    try {
      els.liveTranscriptText.textContent = `"${text}" (${meta.engine}${
        meta.latencyMs ? ` · ${meta.latencyMs}ms` : ""
      })`;
      els.englishInput.value = text;
      await runTranslation({ append: true, overrideText: text });
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

function renderTimeline(items) {
  els.glossTimeline.innerHTML = "";
  if (!items || items.length === 0) {
    els.glossTimeline.innerHTML =
      '<p class="empty-hint">No sign tokens found for this input.</p>';
    els.glossSummary.textContent = "";
    return;
  }

  let poseCount = 0;
  let spellCount = 0;

  items.forEach((item, idx) => {
    if (item.type === "POSE") poseCount++;
    else spellCount++;

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
    els.glossTimeline.appendChild(btn);
  });

  els.glossSummary.textContent = `${poseCount} pose sign${
    poseCount === 1 ? "" : "s"
  } · ${spellCount} fingerspelled`;
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

  const t0 = performance.now();
  const result = useWebLLM
    ? await webllmEngine.translate(text)
    : translateRuleGloss(text);
  const elapsedMs = Math.round(performance.now() - t0);

  els.engineLatency.textContent = `${result.engine === "webllm" ? "WebLLM" : "Rule"} · ${elapsedMs}ms`;
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
    els.webcamToggleBtn.textContent = "Disable Webcam PiP";
    els.webcamToggleBtn.classList.add("is-active");
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
    els.micToggleBtn.textContent = "Start Mic";
    els.micToggleBtn.classList.remove("is-active");
    return;
  }
  try {
    speechController.setEngine(els.sttEngineSelect.value);
    await speechController.start();
    els.micToggleBtn.textContent = "Stop Mic";
    els.micToggleBtn.classList.add("is-active");
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
    renderLocalPoseList();
    await runTranslation();
  } catch (err) {
    alert(err.message);
  }
}

function bindEvents() {
  els.micToggleBtn.addEventListener("click", toggleMic);
  els.sttEngineSelect.addEventListener("change", (e) => {
    speechController.setEngine(e.target.value);
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

  const [assetSummary, gpuStatus] = await Promise.all([
    assetStore.init(),
    checkWebGPUSupport(),
  ]);

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
}

bootstrap();
