const { app, BrowserWindow, session } = require("electron");
const { spawn } = require("node:child_process");
const path = require("node:path");
const fs = require("node:fs");

let bridgeProcess = null;

function startVirtualCamBridge() {
  // Packaged builds bundle a standalone executable (virtualcam_bridge.exe / virtualcam_bridge);
  // dev mode falls back to running bridge/virtualcam_bridge.py via python3/python.
  const packagedBin = path.join(
    process.resourcesPath || "",
    process.platform === "win32" ? "virtualcam_bridge.exe" : "virtualcam_bridge"
  );

  if (app.isPackaged && fs.existsSync(packagedBin)) {
    bridgeProcess = spawn(packagedBin, [], { stdio: "ignore", windowsHide: true });
    return;
  }

  const pyScript = path.join(__dirname, "..", "bridge", "virtualcam_bridge.py");
  if (fs.existsSync(pyScript)) {
    const pyCmd = process.platform === "win32" ? "python" : "python3";
    bridgeProcess = spawn(pyCmd, [pyScript], { stdio: "inherit", windowsHide: true });
  }
}

function stopVirtualCamBridge() {
  if (bridgeProcess && !bridgeProcess.killed) {
    bridgeProcess.kill();
    bridgeProcess = null;
  }
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1360,
    height: 860,
    backgroundColor: "#111317",
    title: "OpenASL — Real-Time Speech-to-ASL Virtual Webcam",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      backgroundThrottling: false, // Keep 30 FPS canvas & virtual cam active when minimized!
    },
  });

  // Auto-grant microphone & webcam permissions for the desktop app
  session.defaultSession.setPermissionRequestHandler((_wc, permission, callback) => {
    callback(permission === "media" || permission === "mediaKeySystem");
  });

  const devUrl = process.env.OPENASL_DEV_URL;
  if (devUrl) {
    win.loadURL(devUrl);
  } else {
    win.loadFile(path.join(__dirname, "..", "dist", "index.html"));
  }
}

app.whenReady().then(() => {
  startVirtualCamBridge();
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  stopVirtualCamBridge();
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", stopVirtualCamBridge);
