# OpenASL

Open-source **Real-Time WebAI Speech-to-ASL Virtual Webcam** built for accessible online communication (Zoom, Teams, Google Meet, Discord, Omegle).

**Zero AI-Generated Visuals · Zero Pre-Rendered Video Dependencies · Zero OBS Required:**
- **100% In-Browser WebAI Speech-to-Text:** Offline Whisper (`@huggingface/transformers` on WebGPU/WASM in a Web Worker + RMS VAD clause chunking) + instant Web Speech API fallback.
- **100% In-Browser WebAI ASL Grammar:** Instant (~2 ms) Time-Topic-Comment Gloss parser (`compromise.js`) + optional local WebGPU LLM (`@mlc-ai/web-llm`).
- **Live `1280×720` HTML5 `<canvas>` Webcam + Bottom-Right ASL PiP:** Renders **257 real adult ASL MediaPipe Holistic pose sequences** with smooth joint-to-joint transitions and public-domain A–Z fingerspelling fallback.
- **Zero-OBS Virtual Camera Output:** Bundles a lightweight virtual camera bridge (`bridge/virtualcam_bridge.py`) targeting **UnityCapture** (`UnityCaptureFilter64.dll`, ~250 KB, MIT-friendly) on Windows and `v4l2loopback` on Linux, plus an Electron wrapper (`electron/main.cjs` + `electron/installer.nsh`) for single-click desktop installation.

---

## Quickstart (Local Dev)

### 1. Start the WebAI App
```bash
npm install
npm run dev
```
Open the local URL in Chrome/Edge:
- Click **Start Mic** to transcribe live speech into ASL.
- Click **Enable Webcam PiP** to composite the ASL pose rig in the bottom-right corner of your live webcam feed.

### 2. Enable System-Level Virtual Webcam (Zero OBS)

**Windows (one-time driver registration, Administrator PowerShell):**
```powershell
powershell -ExecutionPolicy Bypass -File bridge\install_windows_vcam.ps1
```

**Linux:**
```bash
bash bridge/setup_linux_vcam.sh
```

**Start the Virtual Camera Bridge:**
```bash
pip install -r bridge/requirements.txt
python3 bridge/virtualcam_bridge.py
```
Then click **Stream Virtual Cam** in OpenASL and select **Unity Video Capture** (Windows) or **OpenASL Camera** (Linux) inside Zoom, Teams, Discord, or your browser.

---

## Expanding Vocabulary (~2,000 Adult Signs)

```bash
# Default 257-sign core conversational pack (already bundled):
npm run fetch:poses

# Full ~2,000-sign WLASL pose pack from HuggingFace:
uv run --with pyarrow --with numpy python3 scripts/fetch_wlasl_poses.py --all
```

### Dataset Attribution
- **Adult Word Pose Landmarks:** Derived from **WLASL** via [`Kibalama/poseformer-sign-language`](https://huggingface.co/datasets/Kibalama/poseformer-sign-language) (C-UDA — academic / non-commercial computational use).
- **Manual Alphabet (A–Z):** Public-domain SVG illustrations sourced from Wikimedia Commons (`Category:ASL_letters`).