/**
 * Lightweight 320x180 Pixelated 8x8 Bayer Ordered-Dithering Studio Background (~12 FPS).
 *
 * Combines:
 *   1. Slow diagonal topographic / anamorphic wave folds quantized into crisp pixel stipple
 *      using 3 dark brand-matched obsidian/slate tones (zero halation, zero purple).
 *   2. A dithered vignette halo centered behind the 16:9 stage & mechanical control dock
 *      that smoothly expands (+5%) when the microphone is live.
 *
 * Automatically pauses rendering whenever `document.hidden` is true.
 */

const INTERNAL_W = 320;
const INTERNAL_H = 180;
const TARGET_FPS = 12;
const FRAME_INTERVAL_MS = 1000 / TARGET_FPS;

// Standard 8x8 Bayer threshold matrix normalized to (0, 1)
const BAYER_8X8 = new Float32Array([
   0, 32,  8, 40,  2, 34, 10, 42,
  48, 16, 56, 24, 50, 18, 58, 26,
  12, 44,  4, 36, 14, 46,  6, 38,
  60, 28, 52, 20, 62, 30, 54, 22,
   3, 35, 11, 43,  1, 33,  9, 41,
  51, 19, 59, 27, 49, 17, 57, 25,
  15, 47,  7, 39, 13, 45,  5, 37,
  63, 31, 55, 23, 61, 29, 53, 21,
]).map((v) => (v + 0.5) / 64);

// Dark charcoal/obsidian + WebAI steel-slate (#485060 darkened) stipple ramp (ABGR little-endian uint32)
// Tone 0: #080A0D (8, 10, 13)   — Base void obsidian
// Tone 1: #0E1319 (14, 19, 25)  — Architectural mid charcoal
// Tone 2: #19212D (25, 33, 45)  — Crisp logo-slate highlight stipple
// Tone 3: #222B3A (34, 43, 58)  — Peak halo ridge stipple (when mic live)
function packRgba32(r, g, b, a = 255) {
  return ((a & 0xff) << 24) | ((b & 0xff) << 16) | ((g & 0xff) << 8) | (r & 0xff);
}

const PALETTE_U32 = new Uint32Array([
  packRgba32(8, 10, 13, 255),
  packRgba32(14, 19, 25, 255),
  packRgba32(24, 32, 44, 255),
  packRgba32(33, 43, 58, 255),
]);

export class BayerDitherBackground {
  constructor(canvas, stageAnchorEl = null) {
    this.canvas = canvas;
    this.stageAnchorEl = stageAnchorEl;
    this.ctx = null;
    this.imageData = null;
    this.pixels32 = null;

    this.isMicLive = false;
    this.micLevel = 0;
    this._haloScale = 1.0;
    this._anchorX = 0.36;
    this._anchorY = 0.50;
    this._anchorRx = 0.34;
    this._anchorRy = 0.40;

    this._rafId = null;
    this._lastRenderMs = 0;
    this._startMs = typeof performance !== "undefined" ? performance.now() : 0;
    this._resizeObserver = null;
    this._onVisibilityChange = () => {
      if (typeof document !== "undefined" && document.hidden) {
        this._stopLoop();
      } else {
        this._startLoop();
      }
    };

    if (this.canvas && typeof this.canvas.getContext === "function") {
      this.canvas.width = INTERNAL_W;
      this.canvas.height = INTERNAL_H;
      this.ctx = this.canvas.getContext("2d", { alpha: false, desynchronized: true });
      if (this.ctx && typeof this.ctx.createImageData === "function") {
        this.imageData = this.ctx.createImageData(INTERNAL_W, INTERNAL_H);
        this.pixels32 = new Uint32Array(this.imageData.data.buffer);
      }
    }
  }

  setMicLive(isLive) {
    this.isMicLive = Boolean(isLive);
    if (!this.isMicLive) {
      this.micLevel = 0;
    }
  }

  setMicLevel(level) {
    this.micLevel = Math.max(0, Math.min(1, Number(level) || 0));
  }

  updateAnchorBounds() {
    if (
      !this.stageAnchorEl ||
      typeof window === "undefined" ||
      typeof this.stageAnchorEl.getBoundingClientRect !== "function"
    ) {
      return;
    }
    const vw = window.innerWidth || 1440;
    const vh = window.innerHeight || 900;
    if (vw <= 0 || vh <= 0) return;

    const rect = this.stageAnchorEl.getBoundingClientRect();
    if (rect.width > 40 && rect.height > 40) {
      this._anchorX = Math.max(0.15, Math.min(0.85, (rect.left + rect.width * 0.5) / vw));
      this._anchorY = Math.max(0.15, Math.min(0.85, (rect.top + rect.height * 0.48) / vh));
      this._anchorRx = Math.max(0.22, Math.min(0.52, (rect.width * 0.54) / vw));
      this._anchorRy = Math.max(0.26, Math.min(0.56, (rect.height * 0.52) / vh));
    }
  }

  start() {
    if (!this.ctx || !this.pixels32) return;
    this.updateAnchorBounds();

    if (typeof ResizeObserver !== "undefined" && this.stageAnchorEl) {
      this._resizeObserver = new ResizeObserver(() => this.updateAnchorBounds());
      this._resizeObserver.observe(this.stageAnchorEl);
    }
    if (typeof window !== "undefined") {
      window.addEventListener("resize", () => this.updateAnchorBounds(), { passive: true });
    }
    if (typeof document !== "undefined") {
      document.addEventListener("visibilitychange", this._onVisibilityChange);
    }

    // Paint frame 0 immediately so backdrop is ready on first paint
    this.renderStep(typeof performance !== "undefined" ? performance.now() : 0);
    if (typeof document === "undefined" || !document.hidden) {
      this._startLoop();
    }
  }

  _startLoop() {
    if (this._rafId !== null || !this.ctx) return;
    const tick = (now) => {
      this._rafId = requestAnimationFrame(tick);
      if (typeof document !== "undefined" && document.hidden) {
        this._stopLoop();
        return;
      }
      if (now - this._lastRenderMs < FRAME_INTERVAL_MS) return;
      this._lastRenderMs = now;
      this.renderStep(now);
    };
    this._rafId = requestAnimationFrame(tick);
  }

  _stopLoop() {
    if (this._rafId !== null) {
      cancelAnimationFrame(this._rafId);
      this._rafId = null;
    }
  }

  renderStep(nowMs) {
    if (!this.ctx || !this.pixels32) return;

    const t = (nowMs - this._startMs) * 0.001;

    // Smoothly expand stage vignette halo by +5% when mic is live (+ subtle audio breath)
    const targetScale = this.isMicLive ? 1.05 + this.micLevel * 0.025 : 1.0;
    this._haloScale += (targetScale - this._haloScale) * 0.22;

    const cx = this._anchorX;
    const cy = this._anchorY;
    const invRx = 1 / Math.max(0.12, this._anchorRx * this._haloScale);
    const invRy = 1 / Math.max(0.12, this._anchorRy * this._haloScale);
    const micBoost = this.isMicLive ? 0.12 + this.micLevel * 0.08 : 0;

    const out = this.pixels32;
    const invW = 1 / INTERNAL_W;
    const invH = 1 / INTERNAL_H;

    // Precompute X terms to keep 320x180 loop < 0.35ms per tick
    const uArr = new Float32Array(INTERNAL_W);
    const dxNormSqArr = new Float32Array(INTERNAL_W);
    for (let x = 0; x < INTERNAL_W; x++) {
      const u = (x + 0.5) * invW;
      uArr[x] = u;
      const dx = (u - cx) * invRx;
      dxNormSqArr[x] = dx * dx;
    }

    let idx = 0;
    for (let y = 0; y < INTERNAL_H; y++) {
      const v = (y + 0.5) * invH;
      const dy = (v - cy) * invRy;
      const dySq = dy * dy;
      const bayerRow = (y & 7) << 3;
      const waveY1 = v * 3.4 - t * 0.26;
      const waveY2 = v * 5.8 + t * 0.17;

      for (let x = 0; x < INTERNAL_W; x++) {
        const u = uArr[x];

        // Style 1: Slow diagonal topographic / anamorphic wave folds
        const foldA = Math.sin(u * 4.8 + waveY1 + Math.sin(u * 2.3 - waveY2) * 0.85);
        const foldB = Math.cos(u * 7.2 - waveY2 * 0.7 + foldA * 0.65);
        // Sharp topographic contour terrace in [0, 1]
        const topo = Math.max(0, foldA * 0.5 + 0.5) * 0.62 + (foldB * 0.5 + 0.5) * 0.38;

        // Style 2: Anamorphic vignette halo centered behind 16:9 stage & control dock
        const distSq = dxNormSqArr[x] + dySq;
        // Ring + core halo behind viewfinder border & dock
        const coreHalo = Math.max(0, 1 - distSq * 0.78);
        const rimRing = Math.exp(-Math.pow(distSq - 0.72, 2) * 3.4);
        const halo = coreHalo * 0.58 + rimRing * (0.52 + micBoost);

        // Combine Style 1 + Style 2 into continuous architectural field [0..2.35]
        const lum =
          topo * 0.62 +
          halo * (1.18 + micBoost) -
          Math.min(0.45, distSq * 0.16);

        // 8x8 Bayer ordered quantization across dark obsidian/slate tones
        const threshold = BAYER_8X8[bayerRow | (x & 7)];
        const q = Math.max(0, Math.min(3, Math.floor(lum + threshold - 0.28)));
        out[idx++] = PALETTE_U32[q];
      }
    }

    this.ctx.putImageData(this.imageData, 0, 0);
  }

  destroy() {
    this._stopLoop();
    if (this._resizeObserver) {
      this._resizeObserver.disconnect();
      this._resizeObserver = null;
    }
    if (typeof document !== "undefined") {
      document.removeEventListener("visibilitychange", this._onVisibilityChange);
    }
  }
}
