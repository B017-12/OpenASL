/**
 * Pure, DOM-free Live Latency & Telemetry Collector for OpenASL Phase 1.
 * Tracks rolling windows (avg / p95), actual canvas FPS, Pose Hit Rate %,
 * live Queue Depth (with peak), and formats a clean Markdown test report.
 */

function computeStats(samples) {
  const count = samples.length;
  if (count === 0) {
    return { last: null, avg: null, p95: null, count: 0 };
  }
  let sum = 0;
  for (let i = 0; i < count; i++) sum += samples[i];
  const sorted = [...samples].sort((a, b) => a - b);
  const p95Idx = Math.min(count - 1, Math.max(0, Math.ceil(count * 0.95) - 1));
  return {
    last: samples[count - 1],
    avg: sum / count,
    p95: sorted[p95Idx],
    count,
  };
}

function fmtMs(val, decimals = 0) {
  if (val === null || val === undefined || Number.isNaN(val)) return "—";
  return `${val.toFixed(decimals)} ms`;
}

export class MetricsCollector {
  constructor(options = {}) {
    this.windowSize = Math.max(5, options.windowSize || 50);
    this.fpsWindowMs = Math.max(250, options.fpsWindowMs || 1000);
    this.now = options.now || (() => performance.now());

    this.context = {
      sttEngine: "moonshine",
      sttDevice: "pending",
      glossMode: "stream-pse",
      vocabPoseCount: 0,
      viewMode: "stage",
      vcamState: "idle",
      userAgent:
        typeof navigator !== "undefined" && navigator.userAgent
          ? navigator.userAgent
          : "unknown",
    };

    this.reset();
  }

  setContext(patch = {}) {
    if (!patch || typeof patch !== "object") return;
    Object.assign(this.context, patch);
  }

  _pushBounded(arr, value) {
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0) return;
    arr.push(value);
    if (arr.length > this.windowSize) {
      arr.shift();
    }
  }

  recordStt({
    engine = this.context.sttEngine,
    device = this.context.sttDevice,
    latencyMs = 0,
    inferenceMs = 0,
    audioDurationMs = 0,
    estimated = false,
    isFinal = false,
  } = {}) {
    if (typeof latencyMs !== "number" || !Number.isFinite(latencyMs) || latencyMs < 0) {
      return;
    }
    if (engine) this.context.sttEngine = engine;
    if (device) this.context.sttDevice = device;

    this._pushBounded(this._sttSamples, latencyMs);
    if (typeof inferenceMs === "number" && Number.isFinite(inferenceMs) && inferenceMs >= 0) {
      this._sttInferenceLast = inferenceMs;
    }
    if (typeof audioDurationMs === "number" && Number.isFinite(audioDurationMs) && audioDurationMs >= 0) {
      this._sttAudioDurationLast = audioDurationMs;
    }
    this._sttEstimated = Boolean(estimated);
    this._sttTotalCount++;
    if (isFinal) this._sttFinalCount++;
  }

  recordGloss({
    mode = this.context.glossMode,
    glossMs = 0,
    semanticMs = 0,
    inputWords = 0,
    outputTokens = 0,
  } = {}) {
    if (
      typeof glossMs !== "number" ||
      !Number.isFinite(glossMs) ||
      glossMs < 0 ||
      typeof semanticMs !== "number" ||
      !Number.isFinite(semanticMs) ||
      semanticMs < 0
    ) {
      return;
    }
    if (mode) this.context.glossMode = mode;
    const totalMs = glossMs + semanticMs;

    this._pushBounded(this._glossSemanticSamples, totalMs);
    this._glossLast = glossMs;
    this._semanticLast = semanticMs;
    this._glossTotalCount++;
    this._totalInputWords += Math.max(0, Number(inputWords) || 0);
    this._totalOutputTokens += Math.max(0, Number(outputTokens) || 0);
  }

  recordTokenResolution(resolvedItems = []) {
    if (!Array.isArray(resolvedItems) || resolvedItems.length === 0) return;
    for (const item of resolvedItems) {
      if (!item || !item.type) continue;
      if (item.type === "POSE") {
        this._poseHits++;
      } else if (item.type === "FINGERSPELL") {
        this._poseMisses++;
        const gloss = String(item.gloss || "").toUpperCase().trim();
        if (gloss) {
          this._missCounts.set(gloss, (this._missCounts.get(gloss) || 0) + 1);
        }
      }
    }
  }

  recordEndToEnd({
    speechToSignMs,
    pipelineMs = null,
    queueWaitMs = 0,
    gloss = "",
    estimated = false,
  } = {}) {
    if (typeof speechToSignMs !== "number" || !Number.isFinite(speechToSignMs) || speechToSignMs < 0) {
      return;
    }
    this._pushBounded(this._e2eSamples, speechToSignMs);
    this._e2eTotalCount++;
    this._e2eEstimated = Boolean(estimated);

    const safeQueueWait =
      typeof queueWaitMs === "number" && Number.isFinite(queueWaitMs) && queueWaitMs >= 0
        ? queueWaitMs
        : 0;
    const safePipeline =
      typeof pipelineMs === "number" && Number.isFinite(pipelineMs) && pipelineMs >= 0
        ? pipelineMs
        : Math.max(0, speechToSignMs - safeQueueWait);

    this._e2ePipelineLast = safePipeline;
    this._e2eQueueWaitLast = safeQueueWait;
    this._pushBounded(this._pipelineSamples, safePipeline);

    this._recentEvents.push({
      tSec: Number(((this.now() - this._startedAtMs) / 1000).toFixed(1)),
      gloss: String(gloss || ""),
      e2eMs: Math.round(speechToSignMs),
      pipelineMs: Math.round(safePipeline),
      queueWaitMs: Math.round(safeQueueWait),
      estimated: Boolean(estimated),
    });
    if (this._recentEvents.length > 10) {
      this._recentEvents.shift();
    }
  }

  recordFrame(nowMs = this.now()) {
    const t = typeof nowMs === "number" && Number.isFinite(nowMs) ? nowMs : this.now();
    const len = this._frameTimestamps.length;
    if (len > 0 && t < this._frameTimestamps[len - 1]) {
      this._frameTimestamps = [t];
      return;
    }
    this._frameTimestamps.push(t);
    const cutoff = t - this.fpsWindowMs;
    while (this._frameTimestamps.length > 0 && this._frameTimestamps[0] < cutoff) {
      this._frameTimestamps.shift();
    }
  }

  setQueueDepth(depth, isPlaying = true) {
    const num = Number(depth);
    const safeDepth = isPlaying && Number.isFinite(num) ? Math.max(0, Math.floor(num)) : 0;
    this._queueDepth = safeDepth;
    this._isPlaying = Boolean(isPlaying);
    if (safeDepth > this._peakQueueDepth) {
      this._peakQueueDepth = safeDepth;
    }
  }

  _computeFps() {
    const now = this.now();
    const cutoff = now - this.fpsWindowMs;
    if (this._frameTimestamps.length > 0) {
      this._frameTimestamps = this._frameTimestamps.filter((ts) => ts >= cutoff && ts <= now);
    }
    if (this._frameTimestamps.length === 0) return 0;
    const spanSec = this.fpsWindowMs / 1000;
    return Math.round(this._frameTimestamps.length / spanSec);
  }

  _getTopMisses(limit = 8) {
    return Array.from(this._missCounts.entries())
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .slice(0, limit)
      .map(([gloss, count]) => ({ gloss, count }));
  }

  getSnapshot() {
    const now = this.now();
    const uptimeSec = Math.max(0, (now - this._startedAtMs) / 1000);
    const sttStats = computeStats(this._sttSamples);
    const glossStats = computeStats(this._glossSemanticSamples);
    const e2eStats = computeStats(this._e2eSamples);
    const pipelineStats = computeStats(this._pipelineSamples);

    const totalTokens = this._poseHits + this._poseMisses;
    const hitPct = totalTokens > 0 ? (this._poseHits / totalTokens) * 100 : null;
    const topMisses = this._getTopMisses(8);

    return {
      uptimeSec: Number(uptimeSec.toFixed(1)),
      context: { ...this.context },
      sttMs: {
        ...sttStats,
        totalCount: this._sttTotalCount,
        finalCount: this._sttFinalCount,
        inferenceLast: this._sttInferenceLast,
        audioDurationLast: this._sttAudioDurationLast,
        estimated: this._sttEstimated,
      },
      glossSemanticMs: {
        ...glossStats,
        totalCount: this._glossTotalCount,
        glossLast: this._glossLast,
        semanticLast: this._semanticLast,
        inputWords: this._totalInputWords,
        outputTokens: this._totalOutputTokens,
      },
      e2eMs: {
        ...e2eStats,
        totalCount: this._e2eTotalCount,
        pipelineLast: this._e2ePipelineLast,
        pipelineAvg: pipelineStats.avg,
        queueWaitLast: this._e2eQueueWaitLast,
        estimated: this._e2eEstimated,
      },
      queueDepth: this._queueDepth,
      peakQueueDepth: this._peakQueueDepth,
      isPlaying: this._isPlaying,
      fps: this._computeFps(),
      poseHitRate: {
        pct: hitPct,
        hits: this._poseHits,
        misses: this._poseMisses,
        total: totalTokens,
        topMisses,
      },
      recentEvents: [...this._recentEvents],
    };
  }

  formatTestReport() {
    const snap = this.getSnapshot();
    const ctx = snap.context;
    const sttTag = `${ctx.sttEngine} (${ctx.sttDevice}${snap.sttMs.estimated ? ", browser-estimated" : ""})`;
    const hitText =
      snap.poseHitRate.pct === null
        ? "— (0 tokens)"
        : `${snap.poseHitRate.pct.toFixed(1)}% (${snap.poseHitRate.hits}/${snap.poseHitRate.total})`;

    const lines = [
      `## OpenASL Phase 1 Latency & Diagnostics Report`,
      `- **Timestamp (UTC):** ${new Date().toISOString()}`,
      `- **Session Uptime:** ${snap.uptimeSec}s`,
      `- **STT Engine:** ${sttTag}`,
      `- **Gloss Mode:** ${ctx.glossMode}`,
      `- **Vocabulary:** ${ctx.vocabPoseCount} word poses`,
      `- **View / Virtual Cam:** ${ctx.viewMode} / ${ctx.vcamState}`,
      `- **User-Agent:** ${ctx.userAgent}`,
      ``,
      `| Metric | Latest | Rolling Avg | P95 | Notes |`,
      `| :--- | :---: | :---: | :---: | :--- |`,
      `| **STT ms** | ${fmtMs(snap.sttMs.last, 0)} | ${fmtMs(snap.sttMs.avg, 0)} | ${fmtMs(snap.sttMs.p95, 0)} | infer=${fmtMs(snap.sttMs.inferenceLast, 0)}, samples=${snap.sttMs.totalCount}${snap.sttMs.estimated ? " (est.)" : ""} |`,
      `| **Gloss / Semantic ms** | ${fmtMs(snap.glossSemanticMs.last, 2)} | ${fmtMs(snap.glossSemanticMs.avg, 2)} | ${fmtMs(snap.glossSemanticMs.p95, 2)} | gloss=${fmtMs(snap.glossSemanticMs.glossLast, 2)}, semantic=${fmtMs(snap.glossSemanticMs.semanticLast, 2)} |`,
      `| **End-to-End Speech→Sign ms** | ${fmtMs(snap.e2eMs.last, 0)} | ${fmtMs(snap.e2eMs.avg, 0)} | ${fmtMs(snap.e2eMs.p95, 0)} | pipeline=${fmtMs(snap.e2eMs.pipelineLast, 0)}, queueWait=${fmtMs(snap.e2eMs.queueWaitLast, 0)}${snap.e2eMs.estimated ? " (est.)" : ""} |`,
      `| **Queue Depth** | ${snap.queueDepth} | — | — | peak=${snap.peakQueueDepth}, playing=${snap.isPlaying ? "yes" : "no"} |`,
      `| **FPS** | ${snap.fps} | — | — | rolling 1s canvas render FPS |`,
      `| **Pose Hit Rate %** | ${hitText} | — | — | misses=${snap.poseHitRate.misses} |`,
    ];

    if (snap.poseHitRate.topMisses.length > 0) {
      const missStr = snap.poseHitRate.topMisses
        .map((m) => `${m.gloss} (${m.count})`)
        .join(", ");
      lines.push(``, `**Top Fingerspelled Misses:** ${missStr}`);
    }

    if (snap.recentEvents.length > 0) {
      lines.push(``, `**Recent Speech→Sign Traces:**`);
      for (const ev of snap.recentEvents.slice(-6)) {
        lines.push(
          `- \`${ev.tSec}s\` **${ev.gloss || "?"}**: E2E ${ev.e2eMs}ms (pipeline ${ev.pipelineMs}ms + queue ${ev.queueWaitMs}ms${ev.estimated ? ", est." : ""})`
        );
      }
    }

    return lines.join("\n");
  }

  reset() {
    this._startedAtMs = this.now();
    this._sttSamples = [];
    this._sttTotalCount = 0;
    this._sttFinalCount = 0;
    this._sttInferenceLast = null;
    this._sttAudioDurationLast = null;
    this._sttEstimated = false;

    this._glossSemanticSamples = [];
    this._glossTotalCount = 0;
    this._glossLast = null;
    this._semanticLast = null;
    this._totalInputWords = 0;
    this._totalOutputTokens = 0;

    this._e2eSamples = [];
    this._pipelineSamples = [];
    this._e2eTotalCount = 0;
    this._e2ePipelineLast = null;
    this._e2eQueueWaitLast = null;
    this._e2eEstimated = false;
    this._recentEvents = [];

    this._poseHits = 0;
    this._poseMisses = 0;
    this._missCounts = new Map();

    this._frameTimestamps = [];
    this._queueDepth = 0;
    this._peakQueueDepth = 0;
    this._isPlaying = false;
  }
}
