import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { MetricsCollector } from "../src/diagnostics/metricsCollector.js";

describe("MetricsCollector", () => {
  it("computes rolling avg and p95 for STT, Gloss/Semantic, and End-to-End latencies", () => {
    let clock = 1000;
    const mc = new MetricsCollector({ windowSize: 20, now: () => clock });
    mc.setContext({
      sttEngine: "moonshine",
      sttDevice: "webgpu",
      glossMode: "stream-pse",
      vocabPoseCount: 257,
      viewMode: "stage",
      vcamState: "idle",
    });

    const sttVals = [120, 140, 160, 180, 200, 220, 240, 260, 280, 400];
    for (const v of sttVals) {
      mc.recordStt({
        engine: "moonshine",
        device: "webgpu",
        latencyMs: v,
        inferenceMs: Math.round(v * 0.5),
        audioDurationMs: 1200,
        estimated: false,
        isFinal: true,
      });
    }

    mc.recordGloss({ mode: "stream-pse", glossMs: 0.08, semanticMs: 0, inputWords: 2, outputTokens: 2 });
    mc.recordGloss({ mode: "stream-pse", glossMs: 0.12, semanticMs: 4.5, inputWords: 1, outputTokens: 1 });

    mc.recordEndToEnd({ speechToSignMs: 210, pipelineMs: 190, queueWaitMs: 20, gloss: "HELLO", estimated: false });
    mc.recordEndToEnd({ speechToSignMs: 320, pipelineMs: 210, queueWaitMs: 110, gloss: "COFFEE", estimated: false });

    const snap = mc.getSnapshot();
    assert.equal(snap.sttMs.count, 10);
    assert.equal(snap.sttMs.last, 400);
    assert.equal(snap.sttMs.avg, 220);
    assert.equal(snap.sttMs.p95, 400);
    assert.equal(snap.sttMs.estimated, false);

    assert.equal(snap.glossSemanticMs.count, 2);
    assert.ok(Math.abs(snap.glossSemanticMs.last - 4.62) < 1e-6);
    assert.equal(snap.glossSemanticMs.glossLast, 0.12);
    assert.equal(snap.glossSemanticMs.semanticLast, 4.5);

    assert.equal(snap.e2eMs.count, 2);
    assert.equal(snap.e2eMs.last, 320);
    assert.equal(snap.e2eMs.avg, 265);
    assert.equal(snap.e2eMs.pipelineLast, 210);
    assert.equal(snap.e2eMs.queueWaitLast, 110);
  });

  it("computes rolling 1-second FPS with injected monotonic clock", () => {
    let clock = 5000;
    const mc = new MetricsCollector({ fpsWindowMs: 1000, now: () => clock });

    // Record 30 frames evenly spaced across 1 second
    for (let i = 0; i < 30; i++) {
      clock += 33.33;
      mc.recordFrame(clock);
    }
    assert.equal(mc.getSnapshot().fps, 30);

    // Advance past 1s window with no new frames -> FPS drops to 0
    clock += 1500;
    assert.equal(mc.getSnapshot().fps, 0);
  });

  it("tracks Pose Hit Rate %, top fingerspelled misses, and idle queueDepth === 0", () => {
    const mc = new MetricsCollector();
    assert.equal(mc.getSnapshot().poseHitRate.pct, null);

    mc.recordTokenResolution([
      { type: "POSE", gloss: "HELLO" },
      { type: "POSE", gloss: "MY" },
      { type: "POSE", gloss: "NAME" },
      { type: "FINGERSPELL", gloss: "MAYA" },
      { type: "FINGERSPELL", gloss: "PARIS" },
      { type: "FINGERSPELL", gloss: "MAYA" },
    ]);

    mc.setQueueDepth(5, true);
    assert.equal(mc.getSnapshot().queueDepth, 5);
    assert.equal(mc.getSnapshot().peakQueueDepth, 5);

    // When idle (isPlaying=false), live queueDepth MUST report 0 while retaining peak
    mc.setQueueDepth(1, false);
    const snap = mc.getSnapshot();
    assert.equal(snap.queueDepth, 0);
    assert.equal(snap.peakQueueDepth, 5);
    assert.equal(snap.isPlaying, false);

    assert.equal(snap.poseHitRate.hits, 3);
    assert.equal(snap.poseHitRate.misses, 3);
    assert.equal(snap.poseHitRate.total, 6);
    assert.equal(snap.poseHitRate.pct, 50);
    assert.equal(snap.poseHitRate.topMisses[0].gloss, "MAYA");
    assert.equal(snap.poseHitRate.topMisses[0].count, 2);
  });

  it("formats a complete Markdown test report with all 6 metrics and ZERO OBS mentions", () => {
    let clock = 0;
    const mc = new MetricsCollector({ now: () => clock });
    mc.setContext({
      sttEngine: "moonshine",
      sttDevice: "webgpu",
      glossMode: "stream-pse",
      vocabPoseCount: 257,
      viewMode: "webcam",
      vcamState: "connected",
    });
    clock = 12300;
    mc.recordStt({ engine: "moonshine", device: "webgpu", latencyMs: 185, inferenceMs: 92, estimated: false, isFinal: true });
    mc.recordGloss({ mode: "stream-pse", glossMs: 0.09, semanticMs: 0, inputWords: 3, outputTokens: 2 });
    mc.recordTokenResolution([{ type: "POSE", gloss: "HELLO" }, { type: "FINGERSPELL", gloss: "MAYA" }]);
    mc.recordEndToEnd({ speechToSignMs: 215, pipelineMs: 195, queueWaitMs: 20, gloss: "HELLO", estimated: false });
    mc.setQueueDepth(2, true);
    mc.recordFrame(clock);

    const report = mc.formatTestReport();
    assert.match(report, /OpenASL Phase 1 Latency & Diagnostics Report/);
    assert.match(report, /\*\*STT ms\*\*/);
    assert.match(report, /\*\*Gloss \/ Semantic ms\*\*/);
    assert.match(report, /\*\*End-to-End Speech→Sign ms\*\*/);
    assert.match(report, /\*\*Queue Depth\*\*/);
    assert.match(report, /\*\*FPS\*\*/);
    assert.match(report, /\*\*Pose Hit Rate %\*\*/);
    assert.match(report, /50\.0% \(1\/2\)/);
    assert.match(report, /MAYA \(1\)/);
    assert.equal(/\bobs\b/i.test(report), false, "Test report must never mention OBS");

    // Reset clears counters without losing context
    mc.reset();
    const afterReset = mc.getSnapshot();
    assert.equal(afterReset.sttMs.count, 0);
    assert.equal(afterReset.poseHitRate.total, 0);
    assert.equal(afterReset.peakQueueDepth, 0);
    assert.equal(afterReset.context.sttEngine, "moonshine");
  });

  it("Fix #16: rejects invalid STT/Gloss/QueueDepth inputs and handles backward clock jumps in FPS", () => {
    let clock = 5000;
    const mc = new MetricsCollector({ fpsWindowMs: 1000, now: () => clock });

    mc.recordGloss({ glossMs: 10, semanticMs: 0 });
    for (const bad of [-50, NaN, null, Infinity, -Infinity, "5"]) {
      mc.recordGloss({ glossMs: bad });
      mc.recordGloss({ glossMs: 5, semanticMs: bad });
    }
    mc.recordStt({ latencyMs: 100, isFinal: true });
    for (const bad of [-50, NaN, null, Infinity, -Infinity, "10"]) {
      mc.recordStt({ latencyMs: bad, isFinal: true });
    }
    mc.setQueueDepth(Infinity, true);
    mc.setQueueDepth(NaN, true);

    const snap = mc.getSnapshot();
    assert.equal(snap.glossSemanticMs.count, 1, "Invalid glossMs must not push bogus 0ms samples");
    assert.equal(snap.glossSemanticMs.avg, 10);
    assert.equal(snap.sttMs.count, 1);
    assert.equal(snap.sttMs.totalCount, 1, "Invalid latencyMs must not inflate sttMs.totalCount");
    assert.equal(snap.sttMs.finalCount, 1, "Invalid latencyMs must not inflate sttMs.finalCount");
    assert.equal(snap.queueDepth, 0);
    assert.equal(snap.peakQueueDepth, 0);
    const report = mc.formatTestReport();
    assert.equal(report.includes("Infinity"), false);
    assert.equal(report.includes("NaN"), false);

    // Backward clock jump: ancient timestamps must not remain trapped behind a newer timestamp
    mc.recordFrame(5000);
    for (let i = 0; i < 10; i++) {
      mc.recordFrame(1000 + i);
    }
    clock = 5500;
    assert.equal(mc.getSnapshot().fps, 0, "Backward clock jump must not leave stale timestamps counted");

    // Subsequent valid monotonic frame at 5400 reports fps === 1
    mc.recordFrame(5400);
    assert.equal(mc.getSnapshot().fps, 1);
  });

  it("Fix #17: DiagnosticsHud.copyReport() removes fallback textarea even when execCommand throws", async () => {
    const { DiagnosticsHud } = await import("../src/diagnostics/diagnosticsHud.js");

    // Minimal DOM shim for Node unit testing
    const makeEl = (tag) => {
      const el = {
        tagName: tag.toUpperCase(),
        children: [],
        parentNode: null,
        hidden: false,
        style: {},
        textContent: "",
        attrs: {},
        classList: {
          add() {},
          remove() {},
          toggle() {},
        },
        setAttribute(k, v) {
          this.attrs[k] = v;
        },
        appendChild(child) {
          child.parentNode = this;
          this.children.push(child);
          return child;
        },
        removeChild(child) {
          const idx = this.children.indexOf(child);
          if (idx >= 0) {
            this.children.splice(idx, 1);
            child.parentNode = null;
          }
          return child;
        },
        addEventListener() {},
        removeEventListener() {},
        select() {},
        querySelector() {
          return makeEl("span");
        },
      };
      return el;
    };

    const body = makeEl("body");
    const prevDoc = globalThis.document;
    const prevNav = globalThis.navigator;
    globalThis.document = {
      body,
      createElement: (tag) => makeEl(tag),
      execCommand: () => {
        throw new Error("execCommand blocked");
      },
    };
    Object.defineProperty(globalThis, "navigator", {
      value: { userAgent: "node-test" },
      configurable: true,
    });

    try {
      const mc = new MetricsCollector();
      const hud = new DiagnosticsHud(mc, { parent: body });
      const initialBodyCount = body.children.length;
      const copied = await hud.copyReport();
      assert.equal(copied, false);
      assert.equal(
        body.children.length,
        initialBodyCount,
        "Temporary fallback <textarea> must be removed in finally block even when execCommand throws"
      );
      hud.destroy();
    } finally {
      if (prevDoc === undefined) delete globalThis.document;
      else globalThis.document = prevDoc;
      if (prevNav !== undefined) {
        Object.defineProperty(globalThis, "navigator", {
          value: prevNav,
          configurable: true,
        });
      }
    }
  });
});
