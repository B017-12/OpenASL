import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { ASLAssetStore, trimStationaryPoseFrames } from "../src/player/assetLoader.js";
import { ASLStagePlayer } from "../src/player/aslStage.js";
import { VirtualCamClient } from "../src/player/virtualCamClient.js";
import { DEFAULT_PCM_BUFFER_SIZE, createPcmCapture } from "../src/stt/pcmCapture.js";
import { SpeechController } from "../src/stt/speechController.js";

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
    fill: noop,
    stroke: noop,
    fillRect: noop,
    fillText: noop,
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

function makeHand(x, y) {
  return Array.from({ length: 21 }, (_, i) => [x + i * 0.002, y + i * 0.002]);
}

function makePoseFrame(handX, handY, hasHands = true) {
  return {
    p: [
      [0.5, 0.25],
      [0.4, 0.42],
      [0.6, 0.42],
      [0.36, 0.62],
      [0.64, 0.62],
      [handX, handY],
      [handX, handY],
    ],
    lh: hasHands ? makeHand(handX, handY) : null,
    rh: hasHands ? makeHand(handX, handY) : null,
    f: null,
  };
}

describe("Latency & Bug-Fix Overhaul Verification", () => {
  it("trims stationary dead lead-in and lead-out frames on WLASL poses at load time", () => {
    const leadDead = Array.from({ length: 5 }, () => makePoseFrame(0.5, 0.88, false));
    const activeSign = Array.from({ length: 14 }, (_, i) =>
      makePoseFrame(0.38 + i * 0.025, 0.55 - Math.sin(i * 0.3) * 0.12, true)
    );
    const tailDead = Array.from({ length: 6 }, () => {
      const last = activeSign[activeSign.length - 1];
      return JSON.parse(JSON.stringify(last));
    });

    const rawFrames = [...leadDead, ...activeSign, ...tailDead];
    const trimmed = trimStationaryPoseFrames(rawFrames);

    assert.ok(trimmed.length < rawFrames.length, "Expected dead frames to be trimmed");
    assert.ok(
      rawFrames.length - trimmed.length >= 8,
      `Expected >=8 stationary frames shaved, shaved ${rawFrames.length - trimmed.length}`
    );
    assert.ok(Array.isArray(trimmed[0].rh), "Frame 0 must begin immediately on active handshape");
  });

  it("pre-warms all poses in idle batches and executes appendTokens synchronously (0 microtasks) when cached", async () => {
    const store = new ASLAssetStore();
    store.manifest = {
      letters: {},
      poses: {
        HELLO: "/assets/poses/HELLO.json",
        WORLD: "/assets/poses/WORLD.json",
        COFFEE: "/assets/poses/COFFEE.json",
      },
      aliases: { HI: "HELLO" },
    };

    const prevFetch = globalThis.fetch;
    globalThis.fetch = async (url) => ({
      ok: true,
      json: async () => ({
        gloss: url,
        frames: [makePoseFrame(0.45, 0.5), makePoseFrame(0.5, 0.46)],
      }),
    });

    try {
      let idleBatches = 0;
      const loaded = await store.prewarmAllPosesInIdle({
        batchSize: 2,
        scheduleIdle: (cb) => {
          idleBatches++;
          return setTimeout(cb, 0);
        },
      });
      assert.equal(loaded, 3);
      assert.equal(idleBatches, 2);
      assert.equal(store.poseCache.size, 3);

      const player = new ASLStagePlayer(createMockCanvas(), store);

      // Synchronous cached fast-path: player.items & player.isPlaying MUST update BEFORE awaiting!
      const promise = player.appendTokens(["HI", "WORLD"], {
        source: "speech",
        speechTimestampMs: performance.now() - 40,
      });
      assert.equal(player.isPlaying, true, "Expected immediate synchronous playback start before await");
      assert.equal(player.items.length, 2, "Expected immediate synchronous queue commit before await");
      assert.equal(player.items[0].canonical, "HELLO");
      await promise;
    } finally {
      globalThis.fetch = prevFetch;
    }
  });

  it("offloads VirtualCamClient frames to OffscreenCanvas Worker and respects backpressure", async () => {
    const workerMessages = [];
    let workerOnMessage = null;
    const mockWorker = {
      postMessage(msg) {
        workerMessages.push(msg);
      },
      terminate() {},
      set onmessage(fn) {
        workerOnMessage = fn;
      },
      get onmessage() {
        return workerOnMessage;
      },
    };

    let closedBitmaps = 0;
    const statuses = [];
    const client = new VirtualCamClient(
      createMockCanvas(),
      (state, label) => statuses.push({ state, label }),
      "ws://127.0.0.1:8765",
      {
        hasOffscreenPipeline: () => true,
        createWorker: () => mockWorker,
        createImageBitmap: async () => ({
          width: 1280,
          height: 720,
          close: () => {
            closedBitmaps++;
          },
        }),
      }
    );

    client.setFlipOutput(true);
    client.start(() => {});
    assert.equal(client.usingWorkerPipeline, true);
    assert.equal(workerMessages[0].type, "START");
    assert.equal(workerMessages[0].flipOutput, true);

    workerOnMessage({ data: { type: "STATUS", state: "connected", label: "VCAM LIVE" } });
    assert.equal(statuses[statuses.length - 1].state, "connected");

    // Backpressure active (canAcceptFrame: false) -> must NOT allocate or post FRAME bitmap
    workerOnMessage({ data: { type: "TICK", canAcceptFrame: false } });
    await new Promise((r) => setTimeout(r, 5));
    assert.equal(workerMessages.some((m) => m.type === "FRAME"), false);

    // Backpressure clear (canAcceptFrame: true) -> transfers ImageBitmap to worker
    workerOnMessage({ data: { type: "TICK", canAcceptFrame: true } });
    await new Promise((r) => setTimeout(r, 5));
    const frameMsg = workerMessages.find((m) => m.type === "FRAME");
    assert.ok(frameMsg, "Expected FRAME message transferred to worker");
    assert.equal(frameMsg.flipOutput, true);
    assert.equal(closedBitmaps, 0);

    client.destroy();
  });

  it("drives #micLevelFill in Web Speech mode via passive AnalyserNode, coalesces rapid interim bursts, and uses 2048 PCM buffer", async () => {
    assert.equal(DEFAULT_PCM_BUFFER_SIZE, 2048);

    let capturedBufferSize = 0;
    const mockCtx = {
      sampleRate: 48000,
      destination: {},
      createMediaStreamSource: () => ({ connect: () => {}, disconnect: () => {} }),
      createScriptProcessor: (bufSize) => {
        capturedBufferSize = bufSize;
        return { connect: () => {}, disconnect: () => {}, onaudioprocess: null };
      },
      createGain: () => ({ gain: { value: 1 }, connect: () => {}, disconnect: () => {} }),
    };
    const cap = await createPcmCapture({
      audioCtx: mockCtx,
      mediaStream: {},
      onChunk: () => {},
    });
    assert.equal(capturedBufferSize, 2048);
    assert.equal(cap.bufferSize, 2048);
    cap.stop();

    // Web Speech passive AnalyserNode meter + rapid interim burst coalescing
    const levels = [];
    const interims = [];
    let srInstance = null;
    let meterTrackStopped = false;

    class MockSR {
      constructor() {
        srInstance = this;
      }
      start() {
        if (this.onstart) this.onstart();
      }
      abort() {}
    }

    let clock = 1000;
    const ctrl = new SpeechController(
      {
        onLevel: (lvl) => levels.push(lvl),
        onInterim: (txt) => {
          if (txt) interims.push(txt);
        },
      },
      {
        now: () => clock,
        getSpeechRecognition: () => MockSR,
        enableWebSpeechMeter: () => true,
        getUserMedia: async () => ({
          getTracks: () => [
            {
              stop: () => {
                meterTrackStopped = true;
              },
            },
          ],
        }),
        createAudioContext: () => ({
          state: "running",
          close: async () => {},
          createMediaStreamSource: () => ({ connect: () => {}, disconnect: () => {} }),
          createAnalyser: () => ({
            fftSize: 256,
            smoothingTimeConstant: 0.45,
            getFloatTimeDomainData: (arr) => arr.fill(0.08),
          }),
        }),
      }
    );

    ctrl.engine = "webspeech";
    await ctrl.start();

    // Wait for passive AnalyserNode poll (~45ms)
    await new Promise((r) => setTimeout(r, 65));
    assert.ok(
      levels.some((l) => l > 0.5),
      `Expected Web Speech passive AnalyserNode to report non-zero mic levels, got ${JSON.stringify(levels)}`
    );

    // Rapid interim burst: 1st emits immediately (0ms onset), 2nd/3rd/4th within 8ms coalesce into single trailing update
    clock = 2000;
    srInstance.onresult({
      resultIndex: 0,
      results: [{ isFinal: false, 0: { transcript: "I" } }],
    });
    assert.deepEqual(interims, ["I"]);

    clock = 2004;
    srInstance.onresult({
      resultIndex: 0,
      results: [{ isFinal: false, 0: { transcript: "I want" } }],
    });
    clock = 2008;
    srInstance.onresult({
      resultIndex: 0,
      results: [{ isFinal: false, 0: { transcript: "I want coffee" } }],
    });
    clock = 2012;
    srInstance.onresult({
      resultIndex: 0,
      results: [{ isFinal: false, 0: { transcript: "I want coffee tomorrow" } }],
    });

    // Still coalescing before 24ms window elapses
    assert.equal(interims.length, 1);

    await new Promise((r) => setTimeout(r, 35));
    assert.deepEqual(
      interims,
      ["I", "I want coffee tomorrow"],
      "Rapid interim burst must coalesce intermediate hypotheses into a single trailing update"
    );

    ctrl.stop();
    assert.equal(meterTrackStopped, true, "Passive meter MediaStream track must stop cleanly on stop()");
    assert.equal(levels[levels.length - 1], 0);
  });
});
