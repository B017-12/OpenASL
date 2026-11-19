import { interpolatePoseFrames, renderPoseFrame } from "./poseRenderer.js";

/**
 * 30 FPS HTML5 <canvas> ASL Stage & Webcam PiP Compositor (1280x720).
 * - "stage" mode:  Centered 440x440 studio card for Text-to-ASL inspection.
 * - "webcam" mode: Full-frame mirrored physical webcam feed + bottom-right 310x310 ASL PiP overlay
 *                  that fades in while signing and fades out cleanly during silence.
 */

const FPS = 30;
const FRAME_MS = 1000 / FPS;
const POSE_TRANSITION_FRAMES = 5; // ~165ms smooth joint-to-joint blend between consecutive signs
const IDLE_FADE_DELAY_FRAMES = 18; // ~600ms hold after finishing a sentence before PiP fades out

export class ASLStagePlayer {
  constructor(canvas, assetStore, callbacks = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { alpha: false });
    this.assets = assetStore;
    this.onStateChange = callbacks.onStateChange || (() => {});

    this.items = [];
    this.activeIndex = -1;
    this.isPlaying = false;
    this.loop = false;
    this.speed = 1.0;

    // Webcam PiP compositing state
    this.viewMode = "stage"; // "stage" | "webcam"
    this.webcamVideo = null;
    this.mirrorWebcam = true;
    this.idleFrames = IDLE_FADE_DELAY_FRAMES;

    // Frame-level animation state
    this.charIndex = 0;
    this.frameInStep = 0;
    this.transitionFramesLeft = 0;
    this.prevPoseFrame = null;
    this.fadeOpacity = 0;

    this.rafId = null;
    this.lastTickTime = 0;
    this.accumulator = 0;
  }

  setWebcamVideo(videoEl) {
    this.webcamVideo = videoEl;
  }

  setViewMode(mode) {
    this.viewMode = mode === "webcam" ? "webcam" : "stage";
    this.notify();
  }

  startLoop() {
    if (this.rafId !== null) return;
    this.lastTickTime = performance.now();
    const step = (now) => {
      this.stepClock(now);
      this.rafId = requestAnimationFrame(step);
    };
    this.rafId = requestAnimationFrame(step);
  }

  /**
   * Can be invoked from a background Web Worker timer when the tab is minimized behind Zoom/Teams.
   */
  tickExternal() {
    this.stepClock(performance.now());
  }

  stepClock(now) {
    const dt = Math.min(Math.max(0, now - this.lastTickTime), 100);
    this.lastTickTime = now;
    this.accumulator += dt;

    // Adaptive backpressure: speed up tightly if live speech queue builds up
    const remaining = this.activeIndex >= 0 ? this.items.length - this.activeIndex : 0;
    const pressureBoost = remaining > 6 ? 1.75 : remaining > 3 ? 1.45 : remaining > 1 ? 1.15 : 1.0;
    const effectiveSpeed = Math.min(2.2, this.speed * pressureBoost);

    const targetFrameMs = FRAME_MS / effectiveSpeed;
    while (this.accumulator >= targetFrameMs) {
      this.advanceState();
      this.accumulator -= targetFrameMs;
    }

    this.renderFrame();
  }

  destroy() {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
  }

  async resolveTokens(tokens) {
    const resolved = [];
    for (const rawToken of tokens) {
      const gloss = rawToken.toUpperCase().trim();
      if (!gloss) continue;

      const poseEntry = this.assets.resolvePoseEntry(gloss);
      if (poseEntry) {
        try {
          const poseData = await this.assets.loadPoseData(poseEntry);
          resolved.push({
            type: "POSE",
            gloss,
            canonical: poseEntry.canonical,
            isOverride: poseEntry.isOverride,
            frames: poseData.frames,
            source: poseData.source || "MediaPipe Pose",
          });
          continue;
        } catch (err) {
          console.warn(`Pose JSON failed for ${gloss}, falling back to fingerspelling:`, err);
        }
      }

      const letters = gloss.replace(/[^A-Z]/g, "").split("");
      if (letters.length > 0) {
        resolved.push({
          type: "FINGERSPELL",
          gloss,
          letters,
        });
      }
    }
    return resolved;
  }

  async setTokens(tokens) {
    const resolved = await this.resolveTokens(tokens);
    this.items = resolved;
    this.prevPoseFrame = null;
    this.transitionFramesLeft = 0;

    if (this.items.length > 0) {
      this.idleFrames = 0;
      this.seekToToken(0, true);
    } else {
      this.activeIndex = -1;
      this.isPlaying = false;
      this.notify();
    }
    return this.items;
  }

  /**
   * Queue mode for continuous live speech: appends newly spoken gloss tokens without cutting off
   * the currently playing sign.
   */
  async appendTokens(tokens) {
    const resolved = await this.resolveTokens(tokens);
    if (resolved.length === 0) return this.items;

    const wasIdle = !this.isPlaying || this.activeIndex < 0 || this.activeIndex >= this.items.length;

    if (wasIdle) {
      // Start fresh queue for the new spoken utterance
      this.items = resolved;
      this.prevPoseFrame = null;
      this.transitionFramesLeft = 0;
      this.idleFrames = 0;
      this.seekToToken(0, true);
    } else {
      this.items.push(...resolved);
      // Trim already-played history so timeline stays concise during long calls
      if (this.activeIndex > 12) {
        const trim = this.activeIndex - 6;
        this.items.splice(0, trim);
        this.activeIndex -= trim;
      }
      this.idleFrames = 0;
      this.notify();
    }
    return this.items;
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
      this.prevPoseFrame = prevItem.frames[prevItem.frames.length - 1];
      this.transitionFramesLeft = POSE_TRANSITION_FRAMES;
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
      loop: this.loop,
      speed: this.speed,
      viewMode: this.viewMode,
      items: this.items,
    });
  }

  advanceState() {
    const item = this.items[this.activeIndex];
    if (!item) {
      this.fadeOpacity = Math.max(0, this.fadeOpacity - 0.12);
      return;
    }

    if (!this.isPlaying) {
      // In webcam PiP mode, gently fade out the corner card after a short pause of silence
      if (this.viewMode === "webcam") {
        this.idleFrames++;
        if (this.idleFrames > IDLE_FADE_DELAY_FRAMES) {
          this.fadeOpacity = Math.max(0, this.fadeOpacity - 0.08);
        }
      } else {
        this.fadeOpacity = Math.min(1, this.fadeOpacity + 0.18);
      }
      return;
    }

    this.idleFrames = 0;
    this.fadeOpacity = Math.min(1, this.fadeOpacity + 0.2);

    if (item.type === "POSE") {
      if (this.transitionFramesLeft > 0) {
        this.transitionFramesLeft--;
        return;
      }
      this.frameInStep++;
      if (this.frameInStep >= item.frames.length) {
        this.advanceToNextToken();
      }
      return;
    }

    if (item.type === "FINGERSPELL") {
      const framesPerLetter = 9;
      const isLastChar = this.charIndex === item.letters.length - 1;
      const holdFrames = isLastChar ? framesPerLetter + 5 : framesPerLetter;

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
    if (!v || v.readyState < 2 || !v.videoWidth || !v.videoHeight) {
      ctx.fillStyle = "#14171c";
      ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = "rgba(243, 241, 236, 0.55)";
      ctx.font = "500 20px 'Plus Jakarta Sans', system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Waiting for webcam feed…", W / 2, H / 2);
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

  renderFrame() {
    const ctx = this.ctx;
    const W = this.canvas.width;
    const H = this.canvas.height;
    const isWebcam = this.viewMode === "webcam";

    if (isWebcam) {
      this.drawWebcamCover(ctx, W, H);
    } else {
      const grad = ctx.createRadialGradient(W * 0.5, H * 0.44, 60, W * 0.5, H * 0.5, W * 0.65);
      grad.addColorStop(0, "#20242a");
      grad.addColorStop(1, "#111317");
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, W, H);
    }

    const item = this.items[this.activeIndex];
    if (!item) {
      if (!isWebcam) this.renderEmptyStage(ctx, W, H);
      return;
    }

    if (isWebcam && this.fadeOpacity <= 0.02) {
      return;
    }

    // Layout geometry: bottom-right PiP in webcam mode vs centered stage card in text mode
    const cardSize = isWebcam ? 300 : 440;
    const cardX = isWebcam ? W - cardSize - 26 : (W - cardSize) / 2;
    const cardY = isWebcam ? H - cardSize - 76 : 92;
    const hudCenterX = cardX + cardSize / 2;
    const hudCenterY = cardY + cardSize + (isWebcam ? 34 : 44);

    ctx.save();
    ctx.globalAlpha = isWebcam ? this.fadeOpacity : Math.max(0.2, this.fadeOpacity);

    ctx.shadowColor = "rgba(0, 0, 0, 0.45)";
    ctx.shadowBlur = isWebcam ? 22 : 32;
    ctx.shadowOffsetY = isWebcam ? 8 : 14;

    ctx.fillStyle = isWebcam ? "rgba(246, 243, 235, 0.94)" : "#f6f3eb";
    ctx.beginPath();
    ctx.roundRect(cardX, cardY, cardSize, cardSize, isWebcam ? 20 : 26);
    ctx.fill();

    ctx.shadowColor = "transparent";
    ctx.shadowBlur = 0;
    ctx.shadowOffsetY = 0;

    ctx.strokeStyle = "rgba(17, 19, 23, 0.14)";
    ctx.lineWidth = 2;
    ctx.stroke();

    if (item.type === "POSE") {
      const rect = { x: cardX, y: cardY, size: cardSize };
      let activeFrame = item.frames[Math.min(this.frameInStep, item.frames.length - 1)];

      if (this.transitionFramesLeft > 0 && this.prevPoseFrame && item.frames[0]) {
        const progress =
          (POSE_TRANSITION_FRAMES - this.transitionFramesLeft) / POSE_TRANSITION_FRAMES;
        activeFrame = interpolatePoseFrames(this.prevPoseFrame, item.frames[0], progress);
      }

      renderPoseFrame(ctx, activeFrame, rect);
      this.drawTokenProgressBar(ctx, cardX, cardY, cardSize, item);
      this.drawHudPill(ctx, hudCenterX, hudCenterY, item.gloss, isWebcam);
    } else if (item.type === "FINGERSPELL") {
      this.drawFingerspellAsset(ctx, item, cardX, cardY, cardSize);
      this.drawTokenProgressBar(ctx, cardX, cardY, cardSize, item);
      this.drawFingerspellHud(
        ctx,
        hudCenterX,
        hudCenterY,
        item.letters,
        this.charIndex,
        isWebcam
      );
    }

    this.drawCornerStatus(ctx, cardX, cardY, cardSize, item, isWebcam);
    ctx.restore();
  }

  drawFingerspellAsset(ctx, item, cardX, cardY, cardSize) {
    const letter = item.letters[this.charIndex];
    const prevLetter = this.charIndex > 0 ? item.letters[this.charIndex - 1] : null;
    const img = this.assets.getLetterImage(letter);
    if (!img) return;

    const pad = Math.round(cardSize * 0.11);
    const avail = cardSize - pad * 2;
    const nw = img.naturalWidth || 320;
    const nh = img.naturalHeight || 400;
    const fitScale = Math.min(avail / nw, avail / nh);

    const attackScale = this.frameInStep < 2 ? 0.93 + this.frameInStep * 0.035 : 1.0;
    const isDouble = letter === prevLetter;
    const bounceX = isDouble && this.frameInStep < 4 ? (4 - this.frameInStep) * 6 : 0;

    const dw = nw * fitScale;
    const dh = nh * fitScale;

    ctx.save();
    ctx.translate(cardX + cardSize / 2 + bounceX, cardY + cardSize / 2);
    ctx.scale(attackScale, attackScale);
    ctx.drawImage(img, -dw / 2, -dh / 2, dw, dh);
    ctx.restore();
  }

  drawCornerStatus(ctx, cardX, cardY, cardSize, item, isCompact) {
    const badgeLabel =
      item.type === "POSE"
        ? item.isOverride
          ? "LOCAL"
          : "ASL"
        : "SPELL";

    ctx.save();
    ctx.font = `600 ${isCompact ? 11 : 12}px 'JetBrains Mono', monospace`;
    const textW = ctx.measureText(badgeLabel).width;
    const pillW = textW + 18;
    const pillH = isCompact ? 22 : 26;
    const px = cardX + cardSize - pillW - 12;
    const py = cardY + 12;

    ctx.fillStyle = item.type === "POSE" ? "rgba(56, 152, 138, 0.16)" : "rgba(217, 119, 54, 0.16)";
    ctx.beginPath();
    ctx.roundRect(px, py, pillW, pillH, 999);
    ctx.fill();

    ctx.fillStyle = item.type === "POSE" ? "#206b60" : "#a85219";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(badgeLabel, px + pillW / 2, py + pillH / 2 + 0.5);

    const indexLabel = `${this.activeIndex + 1}/${this.items.length}`;
    ctx.fillStyle = "rgba(17, 19, 23, 0.48)";
    ctx.textAlign = "left";
    ctx.fillText(indexLabel, cardX + 16, py + pillH / 2 + 0.5);
    ctx.restore();
  }

  drawTokenProgressBar(ctx, cardX, cardY, cardSize, item) {
    let progress = 0;
    if (item.type === "POSE") {
      progress = Math.min(1, this.frameInStep / Math.max(1, item.frames.length - 1));
    } else if (item.type === "FINGERSPELL") {
      const totalSteps = item.letters.length * 9 + 5;
      const elapsed = this.charIndex * 9 + this.frameInStep;
      progress = Math.min(1, elapsed / totalSteps);
    }

    const barW = cardSize - 36;
    const barX = cardX + 18;
    const barY = cardY + cardSize - 14;

    ctx.fillStyle = "rgba(17, 19, 23, 0.1)";
    ctx.beginPath();
    ctx.roundRect(barX, barY, barW, 4, 2);
    ctx.fill();

    ctx.fillStyle = item.type === "POSE" ? "#38988a" : "#d97736";
    ctx.beginPath();
    ctx.roundRect(barX, barY, Math.max(4, barW * progress), 4, 2);
    ctx.fill();
  }

  drawHudPill(ctx, cx, cy, label, isCompact) {
    ctx.save();
    ctx.font = `700 ${isCompact ? 18 : 24}px 'Plus Jakarta Sans', system-ui, sans-serif`;
    const textW = ctx.measureText(label).width;
    const pillW = Math.max(isCompact ? 140 : 180, textW + (isCompact ? 34 : 52));
    const pillH = isCompact ? 38 : 48;

    ctx.fillStyle = isCompact ? "rgba(17, 19, 23, 0.84)" : "rgba(255, 255, 255, 0.08)";
    ctx.beginPath();
    ctx.roundRect(cx - pillW / 2, cy - pillH / 2, pillW, pillH, 12);
    ctx.fill();

    ctx.strokeStyle = "rgba(255, 255, 255, 0.16)";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.fillStyle = "#f3f1ec";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, cx, cy + 1);
    ctx.restore();
  }

  drawFingerspellHud(ctx, cx, cy, letters, activeIdx, isCompact) {
    ctx.save();
    ctx.font = `700 ${isCompact ? 16 : 22}px 'JetBrains Mono', monospace`;

    const gap = isCompact ? 8 : 12;
    const charWidths = letters.map((ch) =>
      Math.max(isCompact ? 16 : 20, ctx.measureText(ch).width + 6)
    );
    const totalTextW =
      charWidths.reduce((a, b) => a + b, 0) + gap * Math.max(0, letters.length - 1);

    const pillW = Math.max(isCompact ? 140 : 180, totalTextW + (isCompact ? 32 : 48));
    const pillH = isCompact ? 38 : 50;

    ctx.fillStyle = isCompact ? "rgba(17, 19, 23, 0.84)" : "rgba(255, 255, 255, 0.08)";
    ctx.beginPath();
    ctx.roundRect(cx - pillW / 2, cy - pillH / 2, pillW, pillH, 12);
    ctx.fill();

    ctx.strokeStyle = "rgba(255, 255, 255, 0.16)";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    let cursorX = cx - totalTextW / 2;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    for (let i = 0; i < letters.length; i++) {
      const cw = charWidths[i];
      const charCx = cursorX + cw / 2;

      if (i === activeIdx) {
        ctx.fillStyle = "#d97736";
        ctx.beginPath();
        ctx.roundRect(
          charCx - cw / 2 - 2,
          cy - (isCompact ? 12 : 16),
          cw + 4,
          isCompact ? 24 : 32,
          7
        );
        ctx.fill();
        ctx.fillStyle = "#111317";
      } else if (i < activeIdx) {
        ctx.fillStyle = "rgba(243, 241, 236, 0.45)";
      } else {
        ctx.fillStyle = "#f3f1ec";
      }

      ctx.fillText(letters[i], charCx, cy + 1);
      cursorX += cw + gap;
    }

    ctx.restore();
  }

  renderEmptyStage(ctx, W, H) {
    ctx.save();
    ctx.fillStyle = "rgba(243, 241, 236, 0.72)";
    ctx.font = "600 26px 'Plus Jakarta Sans', system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("Speak into your mic or type an English phrase", W / 2, H / 2 - 14);

    ctx.fillStyle = "rgba(243, 241, 236, 0.42)";
    ctx.font = "400 16px 'Plus Jakarta Sans', system-ui, sans-serif";
    ctx.fillText(
      "Enable Webcam PiP to overlay the ASL signer in the bottom-right corner",
      W / 2,
      H / 2 + 24
    );
    ctx.restore();
  }
}
