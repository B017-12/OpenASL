/**
 * Background-tab-safe WebSocket streamer that pipes 1280x720 HTML5 <canvas> frames
 * to the local zero-OBS Virtual Camera bridge (ws://127.0.0.1:8765).
 *
 * Uses an inline Web Worker interval so Chrome/Electron does NOT throttle frame output
 * when the user switches to Zoom, Teams, Discord, or Omegle.
 */

export class VirtualCamClient {
  constructor(canvas, onStatusChange = () => {}, wsUrl = "ws://127.0.0.1:8765") {
    this.canvas = canvas;
    this.wsUrl = wsUrl;
    this.onStatusChange = onStatusChange;

    this.ws = null;
    this.enabled = false;
    this.inFlight = false;
    this.reconnectTimer = null;
    this.tickerWorker = null;
  }

  start(onFrameTick) {
    this.enabled = true;
    this.onFrameTick = onFrameTick;
    this.startBackgroundTicker();
    this.connect();
  }

  stop() {
    this.enabled = false;
    clearTimeout(this.reconnectTimer);
    this.stopBackgroundTicker();
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }
    this.onStatusChange("disconnected", "Virtual Cam Off");
  }

  startBackgroundTicker() {
    if (this.tickerWorker) return;
    const code = `setInterval(() => postMessage(0), 33);`;
    const blob = new Blob([code], { type: "application/javascript" });
    this.tickerWorker = new Worker(URL.createObjectURL(blob));
    this.tickerWorker.onmessage = () => {
      if (!this.enabled) return;
      if (this.onFrameTick) this.onFrameTick();
      this.pushCanvasFrame();
    };
  }

  stopBackgroundTicker() {
    if (this.tickerWorker) {
      this.tickerWorker.terminate();
      this.tickerWorker = null;
    }
  }

  connect() {
    if (!this.enabled) return;
    this.onStatusChange("connecting", "Connecting to Virtual Cam…");

    try {
      const ws = new WebSocket(this.wsUrl);
      ws.binaryType = "arraybuffer";

      ws.onopen = () => {
        this.onStatusChange("connected", "Virtual Cam Live");
      };

      ws.onerror = () => {
        // Handled by onclose reconnect
      };

      ws.onclose = () => {
        this.ws = null;
        if (this.enabled) {
          this.onStatusChange("waiting", "Bridge offline (run virtualcam_bridge.py)");
          this.reconnectTimer = setTimeout(() => this.connect(), 2000);
        }
      };

      this.ws = ws;
    } catch {
      this.onStatusChange("waiting", "Bridge offline");
    }
  }

  pushCanvasFrame() {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN || this.inFlight) return;
    this.inFlight = true;

    this.canvas.toBlob(
      (blob) => {
        if (!blob || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
          this.inFlight = false;
          return;
        }
        blob
          .arrayBuffer()
          .then((buf) => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
              this.ws.send(buf);
            }
          })
          .finally(() => {
            this.inFlight = false;
          });
      },
      "image/jpeg",
      0.82
    );
  }
}
