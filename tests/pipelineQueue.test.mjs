import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { ASLStagePlayer } from "../src/player/aslStage.js";
import { MetricsCollector } from "../src/diagnostics/metricsCollector.js";

function createMockCanvas() {
  const noop = () => {};
  const grad = { addColorStop: noop };
  const ctx = {
    save: noop,
    restore: noop,
    beginPath: noop,
    closePath: noop,
    moveTo: noop,
    lineTo: noop,
    arc: noop,
    ellipse: noop,
    quadraticCurveTo: noop,
    bezierCurveTo: noop,
    setLineDash: noop,
    fill: noop,
    stroke: noop,
    fillRect: noop,
    fillText: noop,
    roundRect: noop,
    translate: noop,
    scale: noop,
    drawImage: noop,
    rect: noop,
    clip: noop,
    createRadialGradient: () => grad,
    createLinearGradient: () => grad,
    measureText: (txt) => ({ width: String(txt || "").length * 10 }),
  };
  return {
    width: 1280,
    height: 720,
    getContext: () => ctx,
  };
}

function makeFakePoseFrame() {
  return {
    p: Array.from({ length: 7 }, (_, i) => [0.5 + i * 0.01, 0.5, 0]),
    lh: null,
    rh: null,
    f: null,
  };
}

function createDelayedAssetStore(delayMs = 40) {
  const poses = {
    HELLO: "/assets/poses/HELLO.json",
    COFFEE: "/assets/poses/COFFEE.json",
    TOMORROW: "/assets/poses/TOMORROW.json",
    THREE: "/assets/poses/THREE.json",
    ZERO: "/assets/poses/ZERO.json",
  };
  let fetchCount = 0;
  return {
    getFetchCount: () => fetchCount,
    getLetterImage: () => null,
    resolvePoseEntry(gloss) {
      const key = String(gloss || "").toUpperCase().trim();
      if (poses[key]) return { canonical: key, src: poses[key], isOverride: false };
      return null;
    },
    async loadPoseData(entry) {
      fetchCount++;
      await new Promise((r) => setTimeout(r, delayMs));
      return {
        gloss: entry.canonical,
        source: "mock",
        frames: [makeFakePoseFrame(), makeFakePoseFrame()],
      };
    },
  };
}

describe("ASLStagePlayer parallel resolution, queue depth, digit fallback & E2E telemetry", () => {
  it("resolves pose JSONs in parallel via Promise.all outside _queueLock", async () => {
    const assets = createDelayedAssetStore(45);
    const canvas = createMockCanvas();
    const player = new ASLStagePlayer(canvas, assets);

    const t0 = performance.now();
    // 3 uncached poses x 45ms each -> parallel should finish well under 110ms (not 135ms+)
    const items = await player.setTokens(["HELLO", "COFFEE", "TOMORROW"]);
    const elapsed = performance.now() - t0;

    assert.equal(items.length, 3);
    assert.equal(assets.getFetchCount(), 3);
    assert.ok(
      elapsed < 115,
      `Expected parallel Promise.all (< 115ms), got ${elapsed.toFixed(1)}ms`
    );

    // Concurrent appendTokens batches should also start fetches in parallel while preserving order
    const t1 = performance.now();
    const pA = player.appendTokens(["HELLO"]);
    const pB = player.appendTokens(["COFFEE"]);
    const [, finalItems] = await Promise.all([pA, pB]);
    const elapsedConcurrent = performance.now() - t1;

    assert.equal(finalItems[finalItems.length - 2].gloss, "HELLO");
    assert.equal(finalItems[finalItems.length - 1].gloss, "COFFEE");
    assert.ok(
      elapsedConcurrent < 85,
      `Expected overlapped appendTokens fetches (< 85ms), got ${elapsedConcurrent.toFixed(1)}ms`
    );
  });

  it("fixes idle getQueueDepth() off-by-one bug (returns 0 when playback finishes)", async () => {
    const assets = createDelayedAssetStore(5);
    const canvas = createMockCanvas();
    const player = new ASLStagePlayer(canvas, assets);

    assert.equal(player.getQueueDepth(), 0);
    await player.setTokens(["HELLO"]);
    assert.equal(player.isPlaying, true);
    assert.equal(player.getQueueDepth(), 1);

    // Advance past the 2 frames of HELLO so playback completes
    player.advanceState();
    player.advanceState();

    assert.equal(player.isPlaying, false);
    assert.equal(player.activeIndex, 0); // activeIndex stays at last sign
    assert.equal(player.getQueueDepth(), 0, "Idle player must report queueDepth === 0");
  });

  it("never silently drops raw numeric tokens and tracks Pose Hit Rate % + single-fire onTokenRendered", async () => {
    const assets = createDelayedAssetStore(5);
    const canvas = createMockCanvas();
    const metrics = new MetricsCollector();
    const renderedEvents = [];
    let frameCount = 0;

    const player = new ASLStagePlayer(canvas, assets, {
      onFrame: (now) => {
        frameCount++;
        metrics.recordFrame(now);
      },
      onTokensResolved: (resolved) => {
        metrics.recordTokenResolution(resolved);
      },
      onTokenRendered: (item, now) => {
        renderedEvents.push({ gloss: item.gloss, now });
        const tr = item._trace;
        metrics.recordEndToEnd({
          speechToSignMs: Math.max(0, now - tr.speechTimestampMs),
          pipelineMs: Math.max(0, tr.enqueuedAtMs - tr.speechTimestampMs),
          queueWaitMs: Math.max(0, now - tr.enqueuedAtMs),
          gloss: item.gloss,
        });
      },
    });

    const speechTs = performance.now() - 120;
    const items = await player.appendTokens(["3", "MAYA", "HELLO"], {
      source: "speech",
      speechTimestampMs: speechTs,
    });

    // "3" -> THREE (POSE), "MAYA" -> FINGERSPELL, "HELLO" -> POSE
    assert.deepEqual(
      items.map((it) => `${it.gloss}:${it.type}`),
      ["THREE:POSE", "MAYA:FINGERSPELL", "HELLO:POSE"]
    );

    // Render twice: onTokenRendered MUST fire only ONCE for the traced first item
    player.fadeOpacity = 1;
    player.renderFrame();
    player.renderFrame();

    assert.equal(frameCount, 2);
    assert.equal(renderedEvents.length, 1, "onTokenRendered must fire once per traced token");
    assert.equal(renderedEvents[0].gloss, "THREE");

    const snap = metrics.getSnapshot();
    assert.equal(snap.poseHitRate.hits, 2);
    assert.equal(snap.poseHitRate.misses, 1);
    assert.ok(Math.abs(snap.poseHitRate.pct - 66.6667) < 0.1);
    assert.equal(snap.e2eMs.count, 1);
    assert.ok(snap.e2eMs.last >= 100);
  });

  it("Fix #18: preserves numeric JS 0 token in ASLStagePlayer.resolveTokens / appendTokens", async () => {
    const assets = createDelayedAssetStore(5);
    const canvas = createMockCanvas();
    const player = new ASLStagePlayer(canvas, assets);

    const items = await player.setTokens([0, 3]);
    assert.deepEqual(
      items.map((it) => `${it.gloss}:${it.type}`),
      ["ZERO:POSE", "THREE:POSE"]
    );
  });

  it("Fix #14: DOM-synced post-await timeline append avoids duplicate chips during overlapping appendTokens calls", async () => {
    const delays = { INIT: 5, SLOW_A: 70, FAST_B: 5 };
    const assets = {
      getLetterImage: () => null,
      resolvePoseEntry: (g) => {
        const key = String(g || "").toUpperCase().trim();
        return delays[key] ? { canonical: key, src: `/assets/${key}.json`, isOverride: false } : null;
      },
      async loadPoseData(entry) {
        await new Promise((r) => setTimeout(r, delays[entry.canonical] || 5));
        return { gloss: entry.canonical, source: "mock", frames: [makeFakePoseFrame(), makeFakePoseFrame()] };
      },
    };
    const canvas = createMockCanvas();
    const player = new ASLStagePlayer(canvas, assets);

    // Mirror main.js DOM-synced timeline reconciliation state
    let renderedTimelineFirst = null;
    let renderedTimelineCount = 0;
    const timelineChips = [];

    const renderTimeline = (items) => {
      timelineChips.length = 0;
      renderedTimelineFirst = items.length > 0 ? items[0] : null;
      renderedTimelineCount = items.length;
      items.forEach((it, idx) => timelineChips.push({ gloss: it.gloss, seekIdx: idx }));
    };
    const appendTimelineChips = (newItems, startIndex) => {
      newItems.forEach((it, offset) =>
        timelineChips.push({ gloss: it.gloss, seekIdx: startIndex + offset })
      );
      if (renderedTimelineCount === 0 && startIndex === 0) {
        renderedTimelineFirst = newItems[0];
      }
      renderedTimelineCount = startIndex + newItems.length;
    };

    const initial = await player.setTokens(["INIT"]);
    renderTimeline(initial);

    const enqueueAndSync = async (tokens) => {
      const items = await player.appendTokens(tokens);
      if (
        renderedTimelineCount > 0 &&
        items[0] === renderedTimelineFirst &&
        items.length >= renderedTimelineCount
      ) {
        if (items.length > renderedTimelineCount) {
          appendTimelineChips(items.slice(renderedTimelineCount), renderedTimelineCount);
        }
      } else {
        renderTimeline(items);
      }
    };

    const pA = enqueueAndSync(["SLOW_A"]);
    await new Promise((r) => setTimeout(r, 10));
    const pB = enqueueAndSync(["FAST_B"]);
    await Promise.all([pA, pB]);

    assert.deepEqual(
      timelineChips,
      [
        { gloss: "INIT", seekIdx: 0 },
        { gloss: "SLOW_A", seekIdx: 1 },
        { gloss: "FAST_B", seekIdx: 2 },
      ],
      "Overlapping appendTokens must produce exact non-duplicated timeline chips"
    );
  });

  it("Fix #15: empty-tail isFinal signal flushes held StreamingRuleGloss pendingPrefix into ASLStagePlayer", async () => {
    const { createStreamingGloss } = await import("../src/nlp/ruleGloss.js");
    const { SpeechController } = await import("../src/stt/speechController.js");

    const assets = createDelayedAssetStore(5);
    const canvas = createMockCanvas();
    const player = new ASLStagePlayer(canvas, assets);
    const sg = createStreamingGloss();

    let workerOnMessage = null;
    const mockWorker = {
      postMessage() {},
      set onmessage(fn) {
        workerOnMessage = fn;
      },
      get onmessage() {
        return workerOnMessage;
      },
    };

    const pendingOps = [];
    const ctrl = new SpeechController(
      {
        onTranscript: (text, meta) => {
          const res = sg.ingestDelta(text, meta);
          if (res.tokens.length > 0) {
            pendingOps.push(player.appendTokens(res.tokens));
          }
        },
        onInterim: (text, meta) => {
          if (meta.isFinal && !meta.isCumulative) {
            const flushed = sg.flush(meta);
            if (flushed.tokens.length > 0) {
              pendingOps.push(player.appendTokens(flushed.tokens));
            }
          }
        },
      },
      {
        createWorker: () => mockWorker,
        getUserMedia: async () => ({ getTracks: () => [{ stop: () => {} }] }),
        createAudioContext: () => ({ state: "running", close: async () => {} }),
        createPcmCapture: async () => ({ stop: () => {} }),
      }
    );

    await ctrl.start();
    // Delta ends in micro-prefix "this" (held in pendingPrefix)
    workerOnMessage({
      data: {
        type: "TRANSCRIPT",
        sessionId: ctrl.sessionId,
        text: "hello this",
        words: ["hello", "this"],
        modelKey: "moonshine",
        device: "wasm",
        inferenceMs: 40,
        speechTimestampMs: 1000,
        isFinal: false,
      },
    });
    // Empty-tail isFinal INTERIM triggers streamingGloss.flush()
    workerOnMessage({
      data: {
        type: "INTERIM",
        sessionId: ctrl.sessionId,
        text: "",
        isFinal: true,
        speechTimestampMs: 1300,
      },
    });
    await Promise.all(pendingOps);
    ctrl.stop();

    assert.deepEqual(
      player.items.map((it) => it.gloss),
      ["HELLO", "THIS"],
      "Held micro-prefix 'this' must be flushed and enqueued on empty-tail isFinal"
    );
  });
});
