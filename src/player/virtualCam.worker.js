/**
 * Dedicated OffscreenCanvas Web Worker for OpenASL Virtual Camera Streaming.
 *
 * Offloads from the main UI thread:
 *   1. Unthrottled 30 FPS background interval (prevents Chromium background-tab throttling).
 *   2. Full-frame horizontal flipping (`flipOutput`) on OffscreenCanvas.
 *   3. Hardware-accelerated `convertToBlob({ type: "image/jpeg", quality: 0.82 })` JPEG encoding.
 *   4. Binary WebSocket streaming with `ws.bufferedAmount` backpressure protection.
 */

const DEFAULT_WIDTH = 1280;
const DEFAULT_HEIGHT = 720;
const TICK_MS = 33;
const MAX_BUFFERED_BYTES = 256 * 1024;

let ws = null;
let wsUrl = "ws://127.0.0.1:8765";
let enabled = false;
let flipOutput = false;
let encoding = false;
let reconnectTimer = null;
let tickInterval = null;

let offscreen = null;
let offscreenCtx = null;

function ensureOffscreen(w, h) {
  if (typeof OffscreenCanvas === "undefined") return null;
  if (!offscreen || offscreen.width !== w || offscreen.height !== h) {
    offscreen = new OffscreenCanvas(w, h);
    offscreenCtx = offscreen.getContext("2d", { alpha: false });
  }
  return offscreenCtx;
}

function postStatus(state, label) {
  self.postMessage({ type: "STATUS", state, label });
}

function closeSocket() {
  clearTimeout(reconnectTimer);
  reconnectTimer = null;
  if (ws) {
    try {
      ws.onopen = null;
      ws.onclose = null;
      ws.onerror = null;
      ws.close();
    } catch {
      // Ignore
    }
    ws = null;
  }
}

function connectSocket() {
  if (!enabled) return;
  closeSocket();
  postStatus("connecting", "VCAM CONNECTING");

  try {
    const socket = new WebSocket(wsUrl);
    socket.binaryType = "arraybuffer";

    socket.onopen = () => {
      if (!enabled || ws !== socket) return;
      postStatus("connected", "VCAM LIVE");
    };

    socket.onerror = () => {
      // Handled via onclose reconnect
    };

    socket.onclose = () => {
      if (ws === socket) {
        ws = null;
      }
      if (enabled) {
        postStatus("waiting", "VCAM STANDBY");
        reconnectTimer = setTimeout(() => connectSocket(), 2000);
      }
    };

    ws = socket;
  } catch {
    postStatus("waiting", "VCAM STANDBY");
    if (enabled) {
      reconnectTimer = setTimeout(() => connectSocket(), 2000);
    }
  }
}

function startTicker() {
  if (tickInterval !== null) return;
  tickInterval = setInterval(() => {
    if (!enabled) return;
    const canAcceptFrame =
      Boolean(ws) &&
      ws.readyState === WebSocket.OPEN &&
      !encoding &&
      ws.bufferedAmount <= MAX_BUFFERED_BYTES;
    self.postMessage({ type: "TICK", canAcceptFrame });
  }, TICK_MS);
}

function stopTicker() {
  if (tickInterval !== null) {
    clearInterval(tickInterval);
    tickInterval = null;
  }
}

async function encodeAndSendBitmap(bitmap, shouldFlip) {
  if (!bitmap) return;

  if (
    !enabled ||
    !ws ||
    ws.readyState !== WebSocket.OPEN ||
    encoding ||
    ws.bufferedAmount > MAX_BUFFERED_BYTES
  ) {
    if (typeof bitmap.close === "function") bitmap.close();
    return;
  }

  const w = bitmap.width || DEFAULT_WIDTH;
  const h = bitmap.height || DEFAULT_HEIGHT;
  const ctx = ensureOffscreen(w, h);
  if (!ctx) {
    if (typeof bitmap.close === "function") bitmap.close();
    return;
  }

  encoding = true;
  try {
    ctx.save();
    if (shouldFlip) {
      ctx.translate(w, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(bitmap, 0, 0, w, h);
    ctx.restore();
    if (typeof bitmap.close === "function") bitmap.close();

    const blob = await offscreen.convertToBlob({ type: "image/jpeg", quality: 0.82 });
    if (!enabled || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (ws.bufferedAmount > MAX_BUFFERED_BYTES) return;

    const buf = await blob.arrayBuffer();
    if (enabled && ws && ws.readyState === WebSocket.OPEN && ws.bufferedAmount <= MAX_BUFFERED_BYTES) {
      ws.send(buf);
    }
  } catch {
    if (typeof bitmap.close === "function") {
      try {
        bitmap.close();
      } catch {
        // Ignore
      }
    }
  } finally {
    encoding = false;
  }
}

self.onmessage = (event) => {
  const msg = event.data || {};

  if (msg.type === "START") {
    if (typeof msg.wsUrl === "string" && msg.wsUrl) {
      wsUrl = msg.wsUrl;
    }
    if (typeof msg.flipOutput === "boolean") {
      flipOutput = msg.flipOutput;
    }
    enabled = true;
    startTicker();
    connectSocket();
    return;
  }

  if (msg.type === "STOP") {
    enabled = false;
    encoding = false;
    stopTicker();
    closeSocket();
    postStatus("disconnected", "VCAM OFF");
    return;
  }

  if (msg.type === "SET_FLIP") {
    flipOutput = Boolean(msg.flipOutput);
    return;
  }

  if (msg.type === "FRAME") {
    const useFlip = typeof msg.flipOutput === "boolean" ? msg.flipOutput : flipOutput;
    encodeAndSendBitmap(msg.bitmap, useFlip);
    return;
  }

  if (msg.type === "RAW_JPEG") {
    if (
      enabled &&
      ws &&
      ws.readyState === WebSocket.OPEN &&
      msg.buffer &&
      ws.bufferedAmount <= MAX_BUFFERED_BYTES
    ) {
      ws.send(msg.buffer);
    }
  }
};
