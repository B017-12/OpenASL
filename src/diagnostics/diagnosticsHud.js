/**
 * Hidden-by-default DOM-only Live Latency & Diagnostics HUD for OpenASL.
 *
 * Design invariants:
 * - Mounted as a `position: fixed` DOM <aside> (NEVER painted on #aslStageCanvas,
 *   so VirtualCam output remains 100% clean).
 * - Hidden by default (`hidden` attribute) with ZERO visible toggle button anywhere
 *   in the main UI.
 * - Toggled ONLY via Ctrl+Shift+D (prevents browser bookmark-all-tabs default).
 * - Pauses its 250ms DOM refresh interval whenever hidden (zero overhead for normal users).
 */

function formatMetricMs(val, decimals = 0) {
  if (val === null || val === undefined || Number.isNaN(val)) return "—";
  return `${val.toFixed(decimals)}ms`;
}

export class DiagnosticsHud {
  constructor(metricsCollector, options = {}) {
    this.metrics = metricsCollector;
    this.parent = options.parent || document.body;
    this.refreshMs = Math.max(100, options.refreshMs || 250);
    this.onVisibilityChange = options.onVisibilityChange || (() => {});

    this._timerId = null;
    this._copyResetTimer = null;
    this._unbindShortcut = null;
    this._isOpen = false;

    this.root = this._createDom();
    this.parent.appendChild(this.root);
  }

  _createDom() {
    const aside = document.createElement("aside");
    aside.id = "diagnosticsHud";
    aside.className = "diag-hud";
    aside.hidden = true;
    aside.setAttribute("role", "region");
    aside.setAttribute("aria-label", "Live Latency and Diagnostics HUD");

    aside.innerHTML = `
      <div class="diag-hud-header">
        <div class="diag-hud-title-wrap">
          <span class="diag-hud-dot" aria-hidden="true"></span>
          <strong class="diag-hud-title">Live Diagnostics</strong>
          <span class="diag-hud-kbd">Ctrl+Shift+D</span>
        </div>
        <button type="button" class="diag-hud-close" data-diag="close" aria-label="Close Diagnostics HUD" title="Close (Ctrl+Shift+D)">×</button>
      </div>

      <div class="diag-hud-context" data-diag="context">moonshine · pending · stream-pse</div>

      <div class="diag-hud-grid">
        <div class="diag-metric">
          <span class="diag-label">STT ms</span>
          <span class="diag-value" data-diag="sttVal">—</span>
          <span class="diag-sub" data-diag="sttSub">avg — · p95 —</span>
        </div>

        <div class="diag-metric">
          <span class="diag-label">Gloss / Semantic ms</span>
          <span class="diag-value" data-diag="glossVal">—</span>
          <span class="diag-sub" data-diag="glossSub">avg — · sem 0.0ms</span>
        </div>

        <div class="diag-metric diag-metric-wide">
          <span class="diag-label">End-to-End Speech→Sign ms</span>
          <span class="diag-value" data-diag="e2eVal">—</span>
          <span class="diag-sub" data-diag="e2eSub">avg — · p95 — · pipe — / wait —</span>
        </div>

        <div class="diag-metric">
          <span class="diag-label">Queue Depth</span>
          <span class="diag-value" data-diag="queueVal">0</span>
          <span class="diag-sub" data-diag="queueSub">peak 0 · idle</span>
        </div>

        <div class="diag-metric">
          <span class="diag-label">FPS</span>
          <span class="diag-value" data-diag="fpsVal">0</span>
          <span class="diag-sub" data-diag="fpsSub">stage · 1s rolling</span>
        </div>

        <div class="diag-metric diag-metric-wide">
          <span class="diag-label">Pose Hit Rate %</span>
          <span class="diag-value" data-diag="hitVal">—</span>
          <span class="diag-sub" data-diag="hitSub">0/0 tokens</span>
        </div>
      </div>

      <div class="diag-hud-actions">
        <button type="button" class="diag-btn diag-btn-primary" data-diag="copy">Copy Test Report</button>
        <button type="button" class="diag-btn diag-btn-ghost" data-diag="reset">Reset</button>
      </div>
    `;

    this._els = {
      context: aside.querySelector('[data-diag="context"]'),
      sttVal: aside.querySelector('[data-diag="sttVal"]'),
      sttSub: aside.querySelector('[data-diag="sttSub"]'),
      glossVal: aside.querySelector('[data-diag="glossVal"]'),
      glossSub: aside.querySelector('[data-diag="glossSub"]'),
      e2eVal: aside.querySelector('[data-diag="e2eVal"]'),
      e2eSub: aside.querySelector('[data-diag="e2eSub"]'),
      queueVal: aside.querySelector('[data-diag="queueVal"]'),
      queueSub: aside.querySelector('[data-diag="queueSub"]'),
      fpsVal: aside.querySelector('[data-diag="fpsVal"]'),
      fpsSub: aside.querySelector('[data-diag="fpsSub"]'),
      hitVal: aside.querySelector('[data-diag="hitVal"]'),
      hitSub: aside.querySelector('[data-diag="hitSub"]'),
      copyBtn: aside.querySelector('[data-diag="copy"]'),
      resetBtn: aside.querySelector('[data-diag="reset"]'),
      closeBtn: aside.querySelector('[data-diag="close"]'),
    };

    this._els.copyBtn.addEventListener("click", () => this.copyReport());
    this._els.resetBtn.addEventListener("click", () => {
      this.metrics.reset();
      this.render();
    });
    this._els.closeBtn.addEventListener("click", () => this.close());

    return aside;
  }

  bindShortcut(target = window) {
    if (this._unbindShortcut) this._unbindShortcut();

    const onKeyDown = (e) => {
      if (
        e.ctrlKey &&
        e.shiftKey &&
        !e.altKey &&
        !e.metaKey &&
        !e.repeat &&
        (e.code === "KeyD" || (typeof e.key === "string" && e.key.toLowerCase() === "d"))
      ) {
        e.preventDefault();
        this.toggle();
      }
    };

    target.addEventListener("keydown", onKeyDown);
    this._unbindShortcut = () => {
      target.removeEventListener("keydown", onKeyDown);
      this._unbindShortcut = null;
    };
    return this._unbindShortcut;
  }

  get isOpen() {
    return this._isOpen;
  }

  open() {
    if (this._isOpen) return;
    this._isOpen = true;
    this.root.hidden = false;
    this.render();
    this._timerId = setInterval(() => this.render(), this.refreshMs);
    this.onVisibilityChange(true);
  }

  close() {
    if (!this._isOpen) return;
    this._isOpen = false;
    this.root.hidden = true;
    if (this._timerId !== null) {
      clearInterval(this._timerId);
      this._timerId = null;
    }
    this.onVisibilityChange(false);
  }

  toggle() {
    if (this._isOpen) this.close();
    else this.open();
  }

  render() {
    if (!this._isOpen) return;
    const snap = this.metrics.getSnapshot();
    const ctx = snap.context;

    this._els.context.textContent = `${ctx.sttEngine} · ${ctx.sttDevice}${
      snap.sttMs.estimated ? " (est)" : ""
    } · ${ctx.glossMode}`;

    // 1. STT ms
    this._els.sttVal.textContent = formatMetricMs(snap.sttMs.last, 0);
    this._els.sttSub.textContent = `avg ${formatMetricMs(snap.sttMs.avg, 0)} · p95 ${formatMetricMs(
      snap.sttMs.p95,
      0
    )}`;

    // 2. Gloss / Semantic ms
    this._els.glossVal.textContent = formatMetricMs(snap.glossSemanticMs.last, 2);
    this._els.glossSub.textContent = `avg ${formatMetricMs(
      snap.glossSemanticMs.avg,
      2
    )} · sem ${formatMetricMs(snap.glossSemanticMs.semanticLast ?? 0, 1)}`;

    // 3. End-to-End Speech->Sign ms
    this._els.e2eVal.textContent = formatMetricMs(snap.e2eMs.last, 0);
    this._els.e2eVal.classList.remove("diag-good", "diag-warn", "diag-bad");
    if (snap.e2eMs.last !== null) {
      if (snap.e2eMs.last <= 250) this._els.e2eVal.classList.add("diag-good");
      else if (snap.e2eMs.last <= 600) this._els.e2eVal.classList.add("diag-warn");
      else this._els.e2eVal.classList.add("diag-bad");
    }
    this._els.e2eSub.textContent = `avg ${formatMetricMs(snap.e2eMs.avg, 0)} · p95 ${formatMetricMs(
      snap.e2eMs.p95,
      0
    )} · pipe ${formatMetricMs(snap.e2eMs.pipelineLast, 0)} / wait ${formatMetricMs(
      snap.e2eMs.queueWaitLast,
      0
    )}`;

    // 4. Queue Depth
    this._els.queueVal.textContent = String(snap.queueDepth);
    this._els.queueSub.textContent = `peak ${snap.peakQueueDepth} · ${
      snap.isPlaying ? "playing" : "idle"
    }`;

    // 5. FPS
    this._els.fpsVal.textContent = String(snap.fps);
    this._els.fpsSub.textContent = `${ctx.viewMode} · 1s rolling`;

    // 6. Pose Hit Rate %
    if (snap.poseHitRate.pct === null) {
      this._els.hitVal.textContent = "—";
      this._els.hitSub.textContent = "0/0 tokens";
    } else {
      this._els.hitVal.textContent = `${snap.poseHitRate.pct.toFixed(1)}%`;
      const missPreview =
        snap.poseHitRate.topMisses.length > 0
          ? ` · misses: ${snap.poseHitRate.topMisses
              .slice(0, 3)
              .map((m) => m.gloss)
              .join(", ")}`
          : "";
      this._els.hitSub.textContent = `${snap.poseHitRate.hits}/${snap.poseHitRate.total} poses${missPreview}`;
    }
  }

  async copyReport() {
    const report = this.metrics.formatTestReport();
    let copied = false;

    if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
      try {
        await navigator.clipboard.writeText(report);
        copied = true;
      } catch {
        copied = false;
      }
    }

    if (!copied && typeof document !== "undefined") {
      const ta = document.createElement("textarea");
      try {
        ta.value = report;
        ta.setAttribute("readonly", "");
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        ta.style.pointerEvents = "none";
        document.body.appendChild(ta);
        ta.select();
        copied = document.execCommand("copy");
      } catch {
        copied = false;
      } finally {
        if (ta.parentNode) {
          ta.parentNode.removeChild(ta);
        }
      }
    }

    clearTimeout(this._copyResetTimer);
    this._els.copyBtn.textContent = copied ? "Copied!" : "Copy Failed";
    this._copyResetTimer = setTimeout(() => {
      this._els.copyBtn.textContent = "Copy Test Report";
    }, 1500);

    return copied;
  }

  destroy() {
    this.close();
    if (this._unbindShortcut) this._unbindShortcut();
    clearTimeout(this._copyResetTimer);
    if (this.root.parentNode) {
      this.root.parentNode.removeChild(this.root);
    }
  }
}
