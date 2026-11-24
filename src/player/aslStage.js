import { interpolatePoseFrames, renderPoseFrame } from "./poseRenderer.js";

/**
 * 30 FPS HTML5 <canvas> ASL Stage & Webcam PiP Compositor (1280x720).
 * - "stage" mode:  Centered 440x440 smoked-obsidian architectural viewfinder with reticle brackets
 *                  and flush bottom caption bar.
 * - "webcam" mode: Full-frame mirrored physical webcam feed + bottom-right draggable 280x280
 *                  smoked-obsidian ASL PiP viewfinder + flush caption bar.
 * - Includes a synchronous 0ms cached fast-path in `appendTokens` when poses are warm in RAM.
 */

const FPS = 30;
const FRAME_MS = 1000 / FPS;
const POSE_TRANSITION_FRAMES = 5; // ~165ms smooth joint-to-joint blend between consecutive signs
const IDLE_FADE_DELAY_FRAMES = 18; // ~600ms hold after finishing a sentence before PiP settles
const FALLBACK_DIGIT_WORDS = [
  "ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE",
];

function expandRawTokenList(tokens) {
  const expanded = [];
  for (const rawToken of tokens || []) {
    const rawStr =
      typeof rawToken === "object" && rawToken !== null ? rawToken.gloss : rawToken;
    const gloss = String(rawStr ?? "").toUpperCase().trim();
    if (!gloss) continue;
    if (/^\d+$/.test(gloss)) {
      for (let i = 0; i < gloss.length; i++) {
        const d = gloss.charCodeAt(i) - 48;
        if (d >= 0 && d <= 9) expanded.push(FALLBACK_DIGIT_WORDS[d]);
      }
    } else {
      expanded.push(gloss);
    }
  }
  return expanded;
}

function buildFingerspellItem(gloss) {
  const alphaGloss = gloss.replace(/\d/g, (ch) => FALLBACK_DIGIT_WORDS[Number(ch)] || "");
  const letters = alphaGloss.replace(/[^A-Z]/g, "").split("");
  if (letters.length > 0) {
    return {
      type: "FINGERSPELL",
      gloss,
      letters,
    };
  }
  return null;
}

function strokeRectPath(ctx, x, y, w, h) {
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.stroke();
}

export class ASLStagePlayer {
  constructor(canvas, assetStore, callbacks = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { alpha: false });
    this.assets = assetStore;
    this.onStateChange = callbacks.onStateChange || (() => {});
    this.onFrame = callbacks.onFrame || (() => {});
    this.onTokensResolved = callbacks.onTokensResolved || (() => {});
    this.onTokenRendered = callbacks.onTokenRendered || (() => {});

    this.items = [];
    this.activeIndex = -1;
    this.isPlaying = false;
    this.loop = false;
    this.speed = 1.0;
    this._queueSource = "manual";

    // Webcam PiP compositing & draggable overlay state
    this.viewMode = "stage"; // "stage" | "webcam"
    this.webcamVideo = null;
    this.mirrorWebcam = true;
    this.idleFrames = IDLE_FADE_DELAY_FRAMES;

    // Custom draggable ASL PiP box coordinates (null = default bottom-right safe zone)
    this.pipSize = 280;
    this.pipX = null;
    this.pipY = null;
    this._draggingPiP = false;
    this._dragOffsetX = 0;
    this._dragOffsetY = 0;

    // Frame-level animation state
    this.charIndex = 0;
    this.frameInStep = 0;
    this.transitionFramesLeft = 0;
    this.prevPoseFrame = null;
    this.fadeOpacity = 0;

    this.rafId = null;
    this.lastTickTime = 0;
    this.accumulator = 0;
    this._queueLock = Promise.resolve();
    this._pendingQueueTasks = 0;

    this._bindCanvasDragEvents();
  }

  getQueueDepth() {
    if (!this.isPlaying || this.activeIndex < 0) return 0;
    return Math.max(0, this.items.length - this.activeIndex);
  }

  clearForLiveSpeech() {
    if (this._queueSource !== "speech") {
      this.isPlaying = false;
      this.activeIndex = this.items.length;
      this._queueSource = "speech";
      this.notify();
    }
  }

  setWebcamVideo(videoEl) {
    this.webcamVideo = videoEl;
  }

  setMirrorWebcam(mirror) {
    this.mirrorWebcam = Boolean(mirror);
    this.notify();
    return this.mirrorWebcam;
  }

  toggleMirrorWebcam() {
    this.mirrorWebcam = !this.mirrorWebcam;
    this.notify();
    return this.mirrorWebcam;
  }

  setViewMode(mode) {
    this.viewMode = mode === "webcam" ? "webcam" : "stage";
    this.notify();
  }

  /**
   * Computes clamped ASL viewfinder coordinates within the 1280x720 camera canvas space
   * (including room for the flush bottom caption bar).
   */
  getPiPBounds() {
    const W = this.canvas?.width || 1280;
    const H = this.canvas?.height || 720;
    const isWebcam = this.viewMode === "webcam";
    const cardSize = isWebcam ? this.pipSize : 436;
    const captionH = isWebcam ? 38 : 46;
    const pad = 18;

    const minX = pad;
    const maxX = Math.max(pad, W - cardSize - pad);
    const minY = pad;
    const maxY = Math.max(pad, H - cardSize - captionH - pad);

    const defaultX = isWebcam ? Math.min(maxX, W - cardSize - 24) : Math.round((W - cardSize) / 2);
    const defaultY = isWebcam
      ? Math.min(maxY, H - cardSize - captionH - 22)
      : Math.round((H - (cardSize + captionH)) / 2);

    const x =
      this.pipX !== null && Number.isFinite(this.pipX)
        ? Math.min(maxX, Math.max(minX, Math.round(this.pipX)))
        : defaultX;
    const y =
      this.pipY !== null && Number.isFinite(this.pipY)
        ? Math.min(maxY, Math.max(minY, Math.round(this.pipY)))
        : defaultY;

    return {
      cardX: x,
      cardY: y,
      cardSize,
      captionH,
      totalHeight: cardSize + captionH,
      minX,
      maxX,
      minY,
      maxY,
    };
  }

  setPiPPosition(x, y) {
    const W = this.canvas?.width || 1280;
    const H = this.canvas?.height || 720;
    const isWebcam = this.viewMode === "webcam";
    const cardSize = isWebcam ? this.pipSize : 436;
    const captionH = isWebcam ? 38 : 46;
    const pad = 18;

    const maxX = Math.max(pad, W - cardSize - pad);
    const maxY = Math.max(pad, H - cardSize - captionH - pad);

    this.pipX = Math.min(maxX, Math.max(pad, Math.round(x)));
    this.pipY = Math.min(maxY, Math.max(pad, Math.round(y)));
  }

  _clientToCanvasCoords(clientX, clientY) {
    if (!this.canvas || typeof this.canvas.getBoundingClientRect !== "function") {
      return null;
    }
    const rect = this.canvas.getBoundingClientRect();
    if (!rect || rect.width <= 0 || rect.height <= 0) return null;
    const scaleX = this.canvas.width / rect.width;
    const scaleY = this.canvas.height / rect.height;
    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY,
    };
  }

  _isPointInPiP(cx, cy) {
    const b = this.getPiPBounds();
    return (
      cx >= b.cardX &&
      cx <= b.cardX + b.cardSize &&
      cy >= b.cardY &&
      cy <= b.cardY + b.totalHeight
    );
  }

  _bindCanvasDragEvents() {
    if (!this.canvas || typeof this.canvas.addEventListener !== "function") return;

    const onPointerDown = (e) => {
      if (e.button !== undefined && e.button !== 0) return;
      const pt = this._clientToCanvasCoords(e.clientX, e.clientY);
      if (!pt || !this._isPointInPiP(pt.x, pt.y)) return;

      const b = this.getPiPBounds();
      this._draggingPiP = true;
      this._dragOffsetX = pt.x - b.cardX;
      this._dragOffsetY = pt.y - b.cardY;
      if (this.canvas.style) this.canvas.style.cursor = "grabbing";
      if (typeof this.canvas.setPointerCapture === "function" && e.pointerId !== undefined) {
        try {
          this.canvas.setPointerCapture(e.pointerId);
        } catch {
          // Ignore
        }
      }
      e.preventDefault();
    };

    const onPointerMove = (e) => {
      const pt = this._clientToCanvasCoords(e.clientX, e.clientY);
      if (!pt) return;

      if (this._draggingPiP) {
        this.setPiPPosition(pt.x - this._dragOffsetX, pt.y - this._dragOffsetY);
        e.preventDefault();
      } else if (this.canvas.style) {
        this.canvas.style.cursor = this._isPointInPiP(pt.x, pt.y) ? "grab" : "default";
      }
    };

    const onPointerUp = (e) => {
      if (!this._draggingPiP) return;
      this._draggingPiP = false;
      if (typeof this.canvas.releasePointerCapture === "function" && e?.pointerId !== undefined) {
        try {
          this.canvas.releasePointerCapture(e.pointerId);
        } catch {
          // Ignore
        }
      }
      const pt = e ? this._clientToCanvasCoords(e.clientX, e.clientY) : null;
      if (this.canvas.style) {
        this.canvas.style.cursor = pt && this._isPointInPiP(pt.x, pt.y) ? "grab" : "default";
      }
    };

    this.canvas.addEventListener("pointerdown", onPointerDown);
    this.canvas.addEventListener("pointermove", onPointerMove);
    this.canvas.addEventListener("pointerup", onPointerUp);
    this.canvas.addEventListener("pointercancel", onPointerUp);
  }

  startLoop() {
    if (this.rafId !== null) return;
    this.lastTickTime = performance.now();
    const step = (now) => {
      this.rafId = requestAnimationFrame(step);
      try {
        this.stepClock(now);
      } catch (err) {
        console.error("[OpenASL Stage] Frame error recovered:", err);
      }
    };
    this.rafId = requestAnimationFrame(step);
  }

  /**
   * Can be invoked from a background Web Worker timer when the tab is minimized behind Zoom/Teams.
   */
  tickExternal() {
    try {
      this.stepClock(performance.now());
    } catch (err) {
      console.error("[OpenASL Stage] External tick recovered:", err);
    }
  }

  stepClock(now) {
    const dt = Math.min(Math.max(0, now - this.lastTickTime), 100);
    this.lastTickTime = now;
    this.accumulator += dt;

    const remaining = this.getQueueDepth();
    const pressureBoost =
      remaining > 5 ? 2.5 : remaining > 3 ? 2.0 : remaining > 2 ? 1.7 : remaining > 1 ? 1.38 : 1.0;
    const effectiveSpeed = Math.min(2.8, this.speed * pressureBoost);

    const targetFrameMs = FRAME_MS / effectiveSpeed;
    let steps = 0;
    while (this.accumulator >= targetFrameMs && steps < 4) {
      this.advanceState();
      this.accumulator -= targetFrameMs;
      steps++;
    }
    if (steps >= 4) this.accumulator = 0;

    this.renderFrame();
  }

  destroy() {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
  }

  /**
   * Attempts synchronous zero-await token resolution when all requested pose JSONs
   * are already warm in `ASLAssetStore`'s RAM cache (or resolve to fingerspelling).
   * Returns `null` if any pose requires an async network fetch.
   */
  tryResolveTokensSync(tokens) {
    const glossList = expandRawTokenList(tokens);
    if (glossList.length === 0) return [];

    const canGetCached = typeof this.assets?.getCachedPoseData === "function";
    const resolved = [];

    for (const gloss of glossList) {
      const poseEntry = this.assets.resolvePoseEntry(gloss);
      if (poseEntry) {
        if (!canGetCached) return null;
        const cachedData = this.assets.getCachedPoseData(poseEntry);
        if (!cachedData) return null;
        if (cachedData.frames?.length > 0) {
          resolved.push({
            type: "POSE",
            gloss,
            canonical: poseEntry.canonical,
            isOverride: poseEntry.isOverride,
            frames: cachedData.frames,
            source: cachedData.source || "MediaPipe Pose",
          });
          continue;
        }
      }

      const spell = buildFingerspellItem(gloss);
      if (spell) resolved.push(spell);
    }

    return resolved;
  }

  async resolveTokens(tokens) {
    const glossList = expandRawTokenList(tokens);
    if (glossList.length === 0) return [];

    const resolvedEntries = await Promise.all(
      glossList.map(async (gloss) => {
        const poseEntry = this.assets.resolvePoseEntry(gloss);
        if (poseEntry) {
          try {
            const poseData = await this.assets.loadPoseData(poseEntry);
            if (poseData?.frames?.length > 0) {
              return {
                type: "POSE",
                gloss,
                canonical: poseEntry.canonical,
                isOverride: poseEntry.isOverride,
                frames: poseData.frames,
                source: poseData.source || "MediaPipe Pose",
              };
            }
          } catch (err) {
            console.warn(`Pose JSON failed for ${gloss}, falling back to fingerspelling:`, err);
          }
        }

        return buildFingerspellItem(gloss);
      })
    );

    return resolvedEntries.filter(Boolean);
  }

  _commitAppendedItems(resolved, trace, poseFetchMs) {
    if (resolved.length === 0) return this.items;

    if (trace) {
      resolved[0]._trace = {
        ...trace,
        enqueuedAtMs: performance.now(),
        poseFetchMs,
      };
    }
    this.onTokensResolved(resolved);

    const isLiveSpeech = trace?.source === "speech";
    const preemptManual = isLiveSpeech && this._queueSource !== "speech";
    if (isLiveSpeech) {
      this._queueSource = "speech";
    }

    const wasIdle =
      preemptManual ||
      !this.isPlaying ||
      this.activeIndex < 0 ||
      this.activeIndex >= this.items.length;

    if (wasIdle) {
      this.items = resolved;
      this.prevPoseFrame = null;
      this.transitionFramesLeft = 0;
      this.charIndex = 0;
      this.frameInStep = 0;
      this.idleFrames = 0;
      this.accumulator = 0;
      this.lastTickTime = performance.now();
      this.seekToToken(0, true);
      if (this.rafId !== null) {
        this.renderFrame();
      }
    } else {
      this.items.push(...resolved);
      if (this.activeIndex > 12) {
        const trim = this.activeIndex - 6;
        this.items.splice(0, trim);
        this.activeIndex = Math.max(0, this.activeIndex - trim);
      }
      this.idleFrames = 0;
      this.notify();
    }
    return this.items;
  }

  setTokens(tokens, trace = null) {
    const fetchStartMs = performance.now();
    const resolvePromise = this.resolveTokens(tokens);

    this._pendingQueueTasks++;
    const task = this._queueLock.then(async () => {
      try {
        const resolved = await resolvePromise;
        const poseFetchMs = performance.now() - fetchStartMs;

        if (resolved.length > 0) {
          if (trace) {
            resolved[0]._trace = {
              ...trace,
              enqueuedAtMs: performance.now(),
              poseFetchMs,
            };
          }
          this.onTokensResolved(resolved);
        }

        this._queueSource = trace?.source === "speech" ? "speech" : "manual";
        this.items = resolved;
        this.prevPoseFrame = null;
        this.transitionFramesLeft = 0;
        this.charIndex = 0;
        this.frameInStep = 0;

        if (this.items.length > 0) {
          this.idleFrames = 0;
          this.seekToToken(0, true);
        } else {
          this.activeIndex = -1;
          this.isPlaying = false;
          this.notify();
        }
        return this.items;
      } finally {
        this._pendingQueueTasks = Math.max(0, this._pendingQueueTasks - 1);
      }
    });
    this._queueLock = task.catch(() => {});
    return task;
  }

  /**
   * Serialized queue mode for continuous live speech:
   * - Synchronous fast-path: when no async queue task is pending and all requested poses are
   *   already cached in RAM, commits and renders Frame 0 immediately in the current call stack
   *   with zero microtask latency!
   * - Async fallback: resolves uncached pose JSONs concurrently via Promise.all outside the
   *   mutation lock while preserving strict FIFO ordering.
   */
  appendTokens(tokens, trace = null) {
    if (this._pendingQueueTasks === 0) {
      const syncStartMs = performance.now();
      const syncResolved = this.tryResolveTokensSync(tokens);
      if (syncResolved !== null) {
        const poseFetchMs = performance.now() - syncStartMs;
        const items = this._commitAppendedItems(syncResolved, trace, poseFetchMs);
        return Promise.resolve(items);
      }
    }

    const fetchStartMs = performance.now();
    const resolvePromise = this.resolveTokens(tokens);

    this._pendingQueueTasks++;
    const task = this._queueLock.then(async () => {
      try {
        const resolved = await resolvePromise;
        if (resolved.length === 0) return this.items;
        const poseFetchMs = performance.now() - fetchStartMs;
        return this._commitAppendedItems(resolved, trace, poseFetchMs);
      } finally {
        this._pendingQueueTasks = Math.max(0, this._pendingQueueTasks - 1);
      }
    });
    this._queueLock = task.catch(() => {});
    return task;
  }

  play() {
    if (this.items.length === 0) return;
    if (this.activeIndex < 0 || this.activeIndex >= this.items.length) {
      this.seekToToken(0, true);
      return;
    }
    this.isPlaying = true;
    this.idleFrames = 0;
    this.notify();
  }

  pause() {
    this.isPlaying = false;
    this.notify();
  }

  togglePlay() {
    if (this.isPlaying) this.pause();
    else this.play();
  }

  replay() {
    if (this.items.length === 0) return;
    this.prevPoseFrame = null;
    this.idleFrames = 0;
    this.seekToToken(0, true);
  }

  setSpeed(newSpeed) {
    this.speed = Math.max(0.5, Math.min(2.0, Number(newSpeed) || 1.0));
    this.notify();
  }

  setLoop(enabled) {
    this.loop = Boolean(enabled);
    this.notify();
  }

  seekToToken(index, autoPlay = true, enableTransition = false) {
    if (index < 0 || index >= this.items.length) return;

    const prevItem = this.items[this.activeIndex];
    const nextItem = this.items[index];

    if (
      enableTransition &&
      prevItem?.type === "POSE" &&
      nextItem?.type === "POSE" &&
      prevItem.frames.length > 0
    ) {
      const lastFrameIdx = Math.min(
        Math.max(0, this.frameInStep),
        prevItem.frames.length - 1
      );
      this.prevPoseFrame = prevItem.frames[lastFrameIdx];
      this.transitionFramesLeft =
        this.items.length - index > 2 ? 3 : POSE_TRANSITION_FRAMES;
    } else {
      this.prevPoseFrame = null;
      this.transitionFramesLeft = 0;
    }

    this.activeIndex = index;
    this.charIndex = 0;
    this.frameInStep = 0;
    this.isPlaying = autoPlay;
    this.idleFrames = 0;
    this.notify();
  }

  notify() {
    this.onStateChange({
      isPlaying: this.isPlaying,
      activeIndex: this.activeIndex,
      charIndex: this.charIndex,
      totalItems: this.items.length,
      queueDepth: this.getQueueDepth(),
      loop: this.loop,
      speed: this.speed,
      viewMode: this.viewMode,
      items: this.items,
    });
  }

  advanceState() {
    const item = this.items[this.activeIndex];
    if (!item) {
      this.fadeOpacity = this.viewMode === "webcam" ? 0.90 : Math.max(0, this.fadeOpacity - 0.12);
      return;
    }

    if (!this.isPlaying) {
      if (this.viewMode === "webcam") {
        this.idleFrames++;
        if (this.idleFrames > IDLE_FADE_DELAY_FRAMES) {
          this.fadeOpacity = Math.max(0.90, this.fadeOpacity - 0.04);
        } else {
          this.fadeOpacity = Math.min(1, this.fadeOpacity + 0.15);
        }
      } else {
        this.fadeOpacity = Math.min(1, this.fadeOpacity + 0.18);
      }
      return;
    }

    this.idleFrames = 0;
    this.fadeOpacity = Math.min(1, this.fadeOpacity + 0.24);
    const hasPendingQueue = this.getQueueDepth() > 1;

    if (item.type === "POSE") {
      if (this.transitionFramesLeft > 0) {
        this.transitionFramesLeft--;
        return;
      }
      this.frameInStep++;
      const effectiveLen =
        hasPendingQueue && item.frames.length > 24
          ? Math.max(20, Math.floor(item.frames.length * 0.75))
          : item.frames.length;
      if (this.frameInStep >= effectiveLen) {
        this.advanceToNextToken();
      }
      return;
    }

    if (item.type === "FINGERSPELL") {
      const framesPerLetter = hasPendingQueue ? 6 : 8;
      const isLastChar = this.charIndex === item.letters.length - 1;
      const holdFrames = isLastChar && !hasPendingQueue ? framesPerLetter + 4 : framesPerLetter;

      this.frameInStep++;
      if (this.frameInStep >= holdFrames) {
        this.frameInStep = 0;
        this.charIndex++;
        if (this.charIndex >= item.letters.length) {
          this.advanceToNextToken();
        } else {
          this.notify();
        }
      }
    }
  }

  advanceToNextToken() {
    const nextIndex = this.activeIndex + 1;
    if (nextIndex < this.items.length) {
      this.seekToToken(nextIndex, true, true);
    } else if (this.loop && this.items.length > 0) {
      this.seekToToken(0, true, true);
    } else {
      this.isPlaying = false;
      this.notify();
    }
  }

  drawWebcamCover(ctx, W, H) {
    const v = this.webcamVideo;
    if (v && v.srcObject && v.paused && !v.ended) {
      v.play().catch(() => {});
    }
    if (!v || v.readyState < 2 || !v.videoWidth || !v.videoHeight) {
      ctx.fillStyle = "#080A0D";
      ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = "rgba(234, 236, 239, 0.52)";
      ctx.font = "500 16px 'JetBrains Mono', monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("AWAITING WEBCAM SIGNAL // 1280×720", W / 2, H / 2);
      return;
    }

    const scale = Math.max(W / v.videoWidth, H / v.videoHeight);
    const dw = v.videoWidth * scale;
    const dh = v.videoHeight * scale;
    const dx = (W - dw) / 2;
    const dy = (H - dh) / 2;

    ctx.save();
    if (this.mirrorWebcam) {
      ctx.translate(W, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(v, dx, dy, dw, dh);
    ctx.restore();
  }

  drawStudioGridBackdrop(ctx, W, H) {
    const grad = ctx.createRadialGradient(W * 0.5, H * 0.46, 40, W * 0.5, H * 0.5, W * 0.68);
    grad.addColorStop(0, "#121720");
    grad.addColorStop(1, "#080A0D");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, W, H);

    // Subtle broadcast safe-area crosshairs & corner fiducials on studio stage
    ctx.strokeStyle = "rgba(255, 255, 255, 0.045)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(W * 0.5, 24);
    ctx.lineTo(W * 0.5, H - 24);
    ctx.moveTo(24, H * 0.5);
    ctx.lineTo(W - 24, H * 0.5);
    ctx.stroke();

    const m = 24;
    const arm = 10;
    ctx.strokeStyle = "rgba(234, 236, 239, 0.22)";
    ctx.beginPath();
    // Top-left
    ctx.moveTo(m, m + arm);
    ctx.lineTo(m, m);
    ctx.lineTo(m + arm, m);
    // Top-right
    ctx.moveTo(W - m - arm, m);
    ctx.lineTo(W - m, m);
    ctx.lineTo(W - m, m + arm);
    // Bottom-left
    ctx.moveTo(m, H - m - arm);
    ctx.lineTo(m, H - m);
    ctx.lineTo(m + arm, H - m);
    // Bottom-right
    ctx.moveTo(W - m - arm, H - m);
    ctx.lineTo(W - m, H - m);
    ctx.lineTo(W - m, H - m - arm);
    ctx.stroke();
  }

  drawViewfinderFrame(ctx, cardX, cardY, cardSize, captionH, isWebcam, accentColor) {
    // Smoked-obsidian architectural viewfinder (0px border-radius)
    ctx.shadowColor = "rgba(0, 0, 0, 0.52)";
    ctx.shadowBlur = isWebcam ? 20 : 28;
    ctx.shadowOffsetY = isWebcam ? 6 : 10;

    ctx.fillStyle = isWebcam ? "rgba(11, 14, 19, 0.90)" : "#0E1217";
    ctx.fillRect(cardX, cardY, cardSize, cardSize + captionH);

    ctx.shadowColor = "transparent";
    ctx.shadowBlur = 0;
    ctx.shadowOffsetY = 0;

    // 1px architectural hairline perimeter around entire unit (viewfinder + flush caption bar)
    ctx.strokeStyle = "rgba(255, 255, 255, 0.14)";
    ctx.lineWidth = 1;
    strokeRectPath(ctx, cardX + 0.5, cardY + 0.5, cardSize - 1, cardSize + captionH - 1);

    // Subtle internal center crosshair inside signer box
    ctx.strokeStyle = "rgba(255, 255, 255, 0.035)";
    ctx.beginPath();
    ctx.moveTo(cardX + cardSize * 0.5, cardY + 28);
    ctx.lineTo(cardX + cardSize * 0.5, cardY + cardSize - 10);
    ctx.stroke();

    // Sharp L-shaped corner reticle brackets inside the signer viewport
    const inset = 8;
    const arm = isWebcam ? 11 : 14;
    const x0 = cardX + inset;
    const y0 = cardY + inset;
    const x1 = cardX + cardSize - inset;
    const y1 = cardY + cardSize - inset;

    ctx.strokeStyle = accentColor || "rgba(234, 236, 239, 0.45)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    // TL
    ctx.moveTo(x0, y0 + arm);
    ctx.lineTo(x0, y0);
    ctx.lineTo(x0 + arm, y0);
    // TR
    ctx.moveTo(x1 - arm, y0);
    ctx.lineTo(x1, y0);
    ctx.lineTo(x1, y0 + arm);
    // BL
    ctx.moveTo(x0, y1 - arm);
    ctx.lineTo(x0, y1);
    ctx.lineTo(x0 + arm, y1);
    // BR
    ctx.moveTo(x1 - arm, y1);
    ctx.lineTo(x1, y1);
    ctx.lineTo(x1, y1 - arm);
    ctx.stroke();
  }

  renderFrame() {
    this.onFrame(performance.now());
    const ctx = this.ctx;
    const W = this.canvas.width;
    const H = this.canvas.height;
    const isWebcam = this.viewMode === "webcam";

    if (isWebcam) {
      this.drawWebcamCover(ctx, W, H);
    } else {
      this.drawStudioGridBackdrop(ctx, W, H);
    }

    const { cardX, cardY, cardSize, captionH } = this.getPiPBounds();
    const item = this.items[this.activeIndex];

    if (!item) {
      if (isWebcam) {
        this.renderWebcamIdlePiP(ctx, cardX, cardY, cardSize, captionH);
      } else {
        this.renderEmptyStage(ctx, W, H);
      }
      return;
    }

    const reticleColor = this.isPlaying
      ? item.type === "POSE"
        ? "rgba(42, 195, 162, 0.78)"
        : "rgba(240, 124, 24, 0.82)"
      : "rgba(234, 236, 239, 0.36)";

    ctx.save();
    ctx.globalAlpha = isWebcam ? Math.max(0.90, this.fadeOpacity) : Math.max(0.25, this.fadeOpacity);

    this.drawViewfinderFrame(ctx, cardX, cardY, cardSize, captionH, isWebcam, reticleColor);

    if (item.type === "POSE") {
      const rect = { x: cardX, y: cardY, size: cardSize };
      let activeFrame = item.frames[Math.min(this.frameInStep, item.frames.length - 1)];

      if (this.transitionFramesLeft > 0 && this.prevPoseFrame && item.frames[0]) {
        const progress =
          (POSE_TRANSITION_FRAMES - this.transitionFramesLeft) / POSE_TRANSITION_FRAMES;
        activeFrame = interpolatePoseFrames(this.prevPoseFrame, item.frames[0], progress);
      }

      renderPoseFrame(ctx, activeFrame, rect);
      this.drawFlushCaptionBar(ctx, cardX, cardY + cardSize, cardSize, captionH, item.gloss, isWebcam, item);
    } else if (item.type === "FINGERSPELL") {
      const safeCharIdx = Math.min(Math.max(0, this.charIndex), item.letters.length - 1);
      this.drawFingerspellAsset(ctx, item, cardX, cardY, cardSize, safeCharIdx);
      this.drawFlushFingerspellBar(
        ctx,
        cardX,
        cardY + cardSize,
        cardSize,
        captionH,
        item.letters,
        safeCharIdx,
        isWebcam,
        item
      );
    }

    this.drawCornerStatus(ctx, cardX, cardY, cardSize, item, isWebcam);
    ctx.restore();

    if (item._trace && !item._rendered) {
      item._rendered = true;
      this.onTokenRendered(item, performance.now());
    }
  }

  drawFingerspellAsset(ctx, item, cardX, cardY, cardSize, safeCharIdx) {
    const letter = item.letters?.[safeCharIdx];
    if (!letter) return;
    const prevLetter = safeCharIdx > 0 ? item.letters[safeCharIdx - 1] : null;
    const img = this.assets.getLetterImage(letter);
    if (!img) return;

    const pad = Math.round(cardSize * 0.13);
    const avail = cardSize - pad * 2;
    const nw = img.naturalWidth || 320;
    const nh = img.naturalHeight || 400;
    const fitScale = Math.min(avail / nw, avail / nh);

    const attackScale = this.frameInStep < 2 ? 0.94 + this.frameInStep * 0.03 : 1.0;
    const isDouble = letter === prevLetter;
    const bounceX = isDouble && this.frameInStep < 4 ? (4 - this.frameInStep) * 6 : 0;

    const dw = nw * fitScale;
    const dh = nh * fitScale;

    ctx.save();
    ctx.translate(cardX + cardSize / 2 + bounceX, cardY + cardSize / 2 + 4);
    ctx.scale(attackScale, attackScale);
    // Wikimedia ASL alphabet SVGs are black ink on transparent; invert to soft bone-white on obsidian
    ctx.filter = "invert(0.92) brightness(1.04)";
    ctx.drawImage(img, -dw / 2, -dh / 2, dw, dh);
    ctx.filter = "none";
    ctx.restore();
  }

  drawCornerStatus(ctx, cardX, cardY, cardSize, item, isCompact) {
    const badgeLabel =
      item.type === "POSE"
        ? item.isOverride
          ? "LOCAL"
          : "POSE"
        : "SPELL";

    ctx.save();
    ctx.font = `700 ${isCompact ? 10 : 11}px 'JetBrains Mono', monospace`;
    const textW = ctx.measureText(badgeLabel).width;
    const cellW = textW + 14;
    const cellH = isCompact ? 19 : 22;
    const px = cardX + cardSize - cellW - 14;
    const py = cardY + 13;

    const isPose = item.type === "POSE";
    ctx.fillStyle = isPose ? "rgba(42, 195, 162, 0.16)" : "rgba(240, 124, 24, 0.18)";
    ctx.fillRect(px, py, cellW, cellH);

    ctx.strokeStyle = isPose ? "rgba(42, 195, 162, 0.45)" : "rgba(240, 124, 24, 0.48)";
    ctx.lineWidth = 1;
    strokeRectPath(ctx, px + 0.5, py + 0.5, cellW - 1, cellH - 1);

    ctx.fillStyle = isPose ? "#5CE2C6" : "#FFAA5A";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(badgeLabel, px + cellW / 2, py + cellH / 2 + 0.5);

    const padIdx = String(this.activeIndex + 1).padStart(2, "0");
    const padTot = String(this.items.length).padStart(2, "0");
    const indexLabel = `${padIdx}/${padTot}`;
    ctx.fillStyle = "rgba(234, 236, 239, 0.62)";
    ctx.textAlign = "left";
    ctx.fillText(indexLabel, cardX + 16, py + cellH / 2 + 0.5);
    ctx.restore();
  }

  _computeItemProgress(item) {
    if (!item) return 0;
    if (item.type === "POSE") {
      return Math.min(1, this.frameInStep / Math.max(1, item.frames.length - 1));
    }
    if (item.type === "FINGERSPELL") {
      const totalSteps = item.letters.length * 9 + 5;
      const elapsed = this.charIndex * 9 + this.frameInStep;
      return Math.min(1, elapsed / totalSteps);
    }
    return 0;
  }

  drawFlushCaptionBar(ctx, barX, barY, barW, barH, label, isCompact, item = null) {
    ctx.save();
    // Flush architectural bottom caption cell
    ctx.fillStyle = "#141922";
    ctx.fillRect(barX + 1, barY, barW - 2, barH - 1);

    // 1px top seam + flush linear progress trace
    ctx.fillStyle = "rgba(255, 255, 255, 0.12)";
    ctx.fillRect(barX, barY, barW, 1);

    if (item) {
      const progress = this._computeItemProgress(item);
      ctx.fillStyle = item.type === "POSE" ? "#2AC3A2" : "#F07C18";
      ctx.fillRect(barX, barY, Math.max(3, Math.round(barW * progress)), 2);
    }

    // Square LED status dot on left of caption bar
    const ledSize = isCompact ? 6 : 7;
    ctx.fillStyle = item?.type === "SPELL" ? "#F07C18" : "#2AC3A2";
    ctx.fillRect(barX + 14, Math.round(barY + (barH - ledSize) / 2), ledSize, ledSize);

    ctx.font = `700 ${isCompact ? 15 : 19}px 'JetBrains Mono', monospace`;
    ctx.fillStyle = "#EAECEF";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, barX + barW / 2, barY + barH / 2 + 1);
    ctx.restore();
  }

  drawFlushFingerspellBar(ctx, barX, barY, barW, barH, letters, activeIdx, isCompact, item = null) {
    ctx.save();
    ctx.fillStyle = "#141922";
    ctx.fillRect(barX + 1, barY, barW - 2, barH - 1);

    ctx.fillStyle = "rgba(255, 255, 255, 0.12)";
    ctx.fillRect(barX, barY, barW, 1);

    if (item) {
      const progress = this._computeItemProgress(item);
      ctx.fillStyle = "#F07C18";
      ctx.fillRect(barX, barY, Math.max(3, Math.round(barW * progress)), 2);
    }

    ctx.font = `700 ${isCompact ? 14 : 18}px 'JetBrains Mono', monospace`;
    const gap = isCompact ? 6 : 8;
    const charWidths = letters.map((ch) =>
      Math.max(isCompact ? 15 : 19, ctx.measureText(ch).width + 6)
    );
    const totalTextW =
      charWidths.reduce((a, b) => a + b, 0) + gap * Math.max(0, letters.length - 1);

    const scaleX = totalTextW > barW - 24 ? (barW - 24) / totalTextW : 1;
    const cy = barY + barH / 2 + 1;

    ctx.translate(barX + barW / 2, cy);
    ctx.scale(scaleX, 1);

    let cursorX = -totalTextW / 2;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    for (let i = 0; i < letters.length; i++) {
      const cw = charWidths[i];
      const charCx = cursorX + cw / 2;

      if (i === activeIdx) {
        const boxH = isCompact ? 22 : 26;
        ctx.fillStyle = "#F07C18";
        ctx.fillRect(charCx - cw / 2 - 2, -boxH / 2, cw + 4, boxH);
        ctx.fillStyle = "#080A0D";
      } else if (i < activeIdx) {
        ctx.fillStyle = "rgba(234, 236, 239, 0.38)";
      } else {
        ctx.fillStyle = "#EAECEF";
      }

      ctx.fillText(letters[i], charCx, 0);
      cursorX += cw + gap;
    }

    ctx.restore();
  }

  renderEmptyStage(ctx, W, H) {
    ctx.save();
    ctx.fillStyle = "#EAECEF";
    ctx.font = "600 22px 'Plus Jakarta Sans', system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("SPEAK INTO MIC OR ENTER TEXT TO SIGN", W / 2, H / 2 - 14);

    ctx.fillStyle = "#949EAE";
    ctx.font = "500 13px 'JetBrains Mono', monospace";
    ctx.fillText(
      "ENABLE WEBCAM PIP TO COMPOSITE SIGNER IN 16:9 SAFE ZONE",
      W / 2,
      H / 2 + 20
    );
    ctx.restore();
  }

  renderWebcamIdlePiP(ctx, cardX, cardY, cardSize, captionH) {
    ctx.save();
    ctx.globalAlpha = 0.90;
    this.drawViewfinderFrame(
      ctx,
      cardX,
      cardY,
      cardSize,
      captionH,
      true,
      "rgba(234, 236, 239, 0.34)"
    );

    ctx.fillStyle = "#EAECEF";
    ctx.font = "700 18px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("OPENASL STANDBY", cardX + cardSize / 2, cardY + cardSize / 2 - 10);

    ctx.fillStyle = "#949EAE";
    ctx.font = "500 12px 'JetBrains Mono', monospace";
    ctx.fillText("SPEAK TO SIGN LIVE", cardX + cardSize / 2, cardY + cardSize / 2 + 14);

    this.drawFlushCaptionBar(ctx, cardX, cardY + cardSize, cardSize, captionH, "ASL READY", true);
    ctx.restore();
  }
}
