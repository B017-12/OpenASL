/**
 * Background-tab-safe WebSocket streamer that pipes 1280x720 HTML5 <canvas> frames
 * to the local Virtual Camera bridge (ws://127.0.0.1:8765).
 *
 * Architecture:
 * - Offloads horizontal full-frame flipping, JPEG encoding (`OffscreenCanvas.convertToBlob`),
 *   30 FPS background ticking, and binary WebSocket streaming with `ws.bufferedAmount`
 *   backpressure protection onto a dedicated Web Worker (`virtualCam.worker.js`).
 * - Falls back cleanly to main-thread `toBlob` with identical `ws.bufferedAmount`
 *   backpressure protection if `OffscreenCanvas` or `createImageBitmap` is unavailable.
 */

const MAX_BUFFERED_BYTES = 256 * 1024;

export class VirtualCamClient {
  constructor(canvas, onStatusChange = () => {}, wsUrl = "ws://127.0.0.1:8765", deps = {}) {
    this.canvas = canvas;
    this.wsUrl = wsUrl;
    this.onStatusChange = onStatusChange;

    this._createWorker =
      deps.createWorker ||
      (() =>
        new Worker(new URL("./virtualCam.worker.js", import.meta.url), {
          type: "module",
        }));
    this._hasOffscreenPipeline =
      deps.hasOffscreenPipeline ??
      (() =>
        typeof Worker !== "undefined" &&
        typeof OffscreenCanvas !== "undefined" &&
        typeof createImageBitmap === "function");
    this._createImageBitmap =
      deps.createImageBitmap || ((source) => createImageBitmap(source));

    this.worker = null;
    this.usingWorkerPipeline = false;

    this.ws = null;
    this.enabled = false;
    this.inFlight = false;
    this.reconnectTimer = null;
    this.tickerWorker = null;

    // Flip the entire composited Virtual Camera output (webcam + ASL overlay together)
    // WITHOUT flipping the local webapp view
    this.flipOutput = false;
    this._flipCanvas = null;
    this._flipCtx = null;
  }

  setFlipOutput(flip) {
    this.flipOutput = Boolean(flip);
    if (this.worker && this.usingWorkerPipeline) {
      this.worker.postMessage({ type: "SET_FLIP", flipOutput: this.flipOutput });
    }
    return this.flipOutput;
  }

  toggleFlipOutput() {
    return this.setFlipOutput(!this.flipOutput);
  }

  _getSourceCanvas() {
    if (!this.flipOutput || typeof document === "undefined") {
      return this.canvas;
    }
    const W = this.canvas.width || 1280;
    const H = this.canvas.height || 720;
    if (!this._flipCanvas) {
      this._flipCanvas = document.createElement("canvas");
      this._flipCanvas.width = W;
      this._flipCanvas.height = H;
      this._flipCtx = this._flipCanvas.getContext("2d", { alpha: false });
    } else if (this._flipCanvas.width !== W || this._flipCanvas.height !== H) {
      this._flipCanvas.width = W;
      this._flipCanvas.height = H;
    }

    const ctx = this._flipCtx;
    ctx.save();
    ctx.translate(W, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(this.canvas, 0, 0, W, H);
    ctx.restore();
    return this._flipCanvas;
  }

  start(onFrameTick) {
    this.enabled = true;
    this.onFrameTick = onFrameTick;

    if (this._hasOffscreenPipeline()) {
      try {
        this._startWorkerPipeline();
        return;
      } catch {
        this.usingWorkerPipeline = false;
      }
    }

    this.startBackgroundTicker();
    this.connect();
  }

  _startWorkerPipeline() {
    if (!this.worker) {
      this.worker = this._createWorker();
      this.worker.onmessage = (event) => {
        const msg = event.data || {};
        if (msg.type === "STATUS") {
          this.onStatusChange(msg.state, msg.label);
          return;
        }
        if (msg.type === "TICK") {
          if (!this.enabled) return;
          if (this.onFrameTick && (typeof document === "undefined" || document.hidden)) {
            this.onFrameTick();
          }
          if (msg.canAcceptFrame) {
            this._transferFrameBitmap();
          }
        }
      };
    }

    this.usingWorkerPipeline = true;
    this.inFlight = false;
    this.worker.postMessage({
      type: "START",
      wsUrl: this.wsUrl,
      flipOutput: this.flipOutput,
    });
  }

  _transferFrameBitmap() {
    if (!this.enabled || !this.worker || this.inFlight || !this.canvas) return;
    this.inFlight = true;
    this._createImageBitmap(this.canvas)
      .then((bitmap) => {
        if (!this.enabled || !this.worker) {
          if (bitmap && typeof bitmap.close === "function") bitmap.close();
          return;
        }
        this.worker.postMessage(
          { type: "FRAME", bitmap, flipOutput: this.flipOutput },
          [bitmap]
        );
      })
      .catch(() => {
        // Ignore transient frame capture error
      })
      .finally(() => {
        this.inFlight = false;
      });
  }

  stop() {
    this.enabled = false;
    this.inFlight = false;
    clearTimeout(this.reconnectTimer);
    this.stopBackgroundTicker();

    if (this.worker && this.usingWorkerPipeline) {
      try {
        this.worker.postMessage({ type: "STOP" });
      } catch {
        // Ignore
      }
      this.onStatusChange("disconnected", "VCAM OFF");
      return;
    }

    if (this.ws) {
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }
    this.onStatusChange("disconnected", "VCAM OFF");
  }

  destroy() {
    this.stop();
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.usingWorkerPipeline = false;
  }

  startBackgroundTicker() {
    if (this.tickerWorker) return;
    const code = `setInterval(() => postMessage(0), 33);`;
    const blob = new Blob([code], { type: "application/javascript" });
    this.tickerWorker = new Worker(URL.createObjectURL(blob));
    this.tickerWorker.onmessage = () => {
      if (!this.enabled) return;
      if (this.onFrameTick && (typeof document === "undefined" || document.hidden)) {
        this.onFrameTick();
      }
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
    this.onStatusChange("connecting", "VCAM CONNECTING");

    try {
      const ws = new WebSocket(this.wsUrl);
      ws.binaryType = "arraybuffer";

      ws.onopen = () => {
        this.onStatusChange("connected", "VCAM LIVE");
      };

      ws.onerror = () => {
        // Handled by onclose reconnect
      };

      ws.onclose = () => {
        this.ws = null;
        if (this.enabled) {
          this.onStatusChange("waiting", "VCAM STANDBY");
          this.reconnectTimer = setTimeout(() => this.connect(), 2000);
        }
      };

      this.ws = ws;
    } catch {
      this.onStatusChange("waiting", "VCAM STANDBY");
    }
  }

  pushCanvasFrame() {
    if (
      !this.ws ||
      this.ws.readyState !== WebSocket.OPEN ||
      this.inFlight ||
      this.ws.bufferedAmount > MAX_BUFFERED_BYTES
    ) {
      return;
    }
    this.inFlight = true;

    const targetCanvas = this._getSourceCanvas();
    targetCanvas.toBlob(
      (blob) => {
        if (
          !blob ||
          !this.ws ||
          this.ws.readyState !== WebSocket.OPEN ||
          this.ws.bufferedAmount > MAX_BUFFERED_BYTES
        ) {
          this.inFlight = false;
          return;
        }
        blob
          .arrayBuffer()
          .then((buf) => {
            if (
              this.ws &&
              this.ws.readyState === WebSocket.OPEN &&
              this.ws.bufferedAmount <= MAX_BUFFERED_BYTES
            ) {
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
