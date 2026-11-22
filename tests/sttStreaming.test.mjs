import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  TARGET_SAMPLE_RATE,
  downsampleTo16k,
  StreamingAudioBuffer,
} from "../src/stt/audioRingBuffer.js";
import {
  StreamingStabilizer,
  computeMaxNewTokens,
  sanitizeSttText,
} from "../src/stt/streamingStabilizer.js";
import { SpeechController } from "../src/stt/speechController.js";

function makeChunk(durationMs, value = 0.05) {
  const len = Math.round((durationMs / 1000) * TARGET_SAMPLE_RATE);
  const arr = new Float32Array(len);
  arr.fill(value);
  return arr;
}

describe("downsampleTo16k & StreamingAudioBuffer", () => {
  it("downsamples 48kHz, 44.1kHz, and passes through 16kHz cleanly", () => {
    const in48 = new Float32Array(4800).fill(0.25);
    const out48 = downsampleTo16k(in48, 48000);
    assert.equal(out48.length, 1600);
    assert.ok(Math.abs(out48[0] - 0.25) < 1e-5);

    const in44 = new Float32Array(4410).fill(0.5);
    const out44 = downsampleTo16k(in44, 44100);
    assert.equal(out44.length, 1600);

    const in16 = new Float32Array(1600).fill(-0.1);
    const out16 = downsampleTo16k(in16, 16000);
    assert.equal(out16.length, 1600);
    assert.notEqual(out16, in16); // returns independent copy
  });

  it("includes 180ms pre-roll on speech onset, triggers micro-pause finalize at >=280ms, and compacts 4.5s -> 1.4s accurately", () => {
    const buf = new StreamingAudioBuffer({
      preRollMs: 180,
      maxWindowSec: 4.5,
      compactKeepSec: 1.4,
    });

    let ts = 1000;
    // Push 3 quiet pre-roll chunks (80ms each = 240ms)
    for (let i = 0; i < 3; i++) {
      ts += 80;
      buf.pushChunk(makeChunk(80, 0.001), 0.001, ts);
    }
    assert.equal(buf.isSpeaking, false);

    // Speech onset: 400ms of active speech (5 x 80ms)
    const speechStartTs = ts;
    for (let i = 0; i < 5; i++) {
      ts += 80;
      buf.pushChunk(makeChunk(80, 0.08), 0.08, ts);
    }
    assert.equal(buf.isSpeaking, true);
    assert.equal(buf.activeSpeechMs, 400);

    // Pre-roll must have been included (windowStartMs earlier than speechStartTs)
    const slice = buf.getWindowSlice();
    assert.ok(slice.windowStartMs < speechStartTs, "Pre-roll should precede speech onset");
    assert.ok(slice.durationSec > 0.45, "Window should include pre-roll + active speech");

    // 240ms silence should NOT finalize yet
    for (let i = 0; i < 3; i++) {
      ts += 80;
      buf.pushChunk(makeChunk(80, 0.001), 0.001, ts);
    }
    assert.equal(buf.shouldFinalize(280), false);

    // Another 80ms silence (320ms total) MUST trigger finalize
    ts += 80;
    buf.pushChunk(makeChunk(80, 0.001), 0.001, ts);
    assert.equal(buf.shouldFinalize(280), true);

    // Test long continuous speech compaction (5s -> 1.4s tail with advanced windowStartMs)
    buf.resetUtterance();
    const longStartTs = ts;
    for (let i = 0; i < 60; i++) {
      ts += 80;
      buf.pushChunk(makeChunk(80, 0.07), 0.07, ts);
    }
    const beforeCompact = buf.getWindowSlice();
    assert.ok(buf.durationSec >= 4.5);
    const compacted = buf.compactLeft(1.4);
    assert.equal(compacted, true);
    assert.ok(Math.abs(buf.durationSec - 1.4) < 0.05);
    assert.ok(buf.windowStartMs > longStartTs, "windowStartMs must advance after compactLeft");
    assert.equal(buf.windowEndMs, beforeCompact.windowEndMs);
  });
});

describe("StreamingStabilizer (LocalAgreement-2, overlap anchor, safe hallucination, Moonshine max_new_tokens)", () => {
  it("commits words progressively with LocalAgreement-2 and trailing-word holdback", () => {
    const st = new StreamingStabilizer();

    // Tick 1: "I want tom" -> first hypothesis has no prior agreement yet
    const t1 = st.update("I want tom", { isFinal: false, windowStartMs: 1000, windowEndMs: 1500 });
    assert.deepEqual(t1.newWords, []);
    assert.equal(t1.interimText, "I want tom");

    // Tick 2: "I want tomorrow" -> "I want" agreed across 2 ticks, "tomorrow" held back
    const t2 = st.update("I want tomorrow", { isFinal: false, windowStartMs: 1000, windowEndMs: 1800 });
    assert.deepEqual(t2.newWords, ["I", "want"]);
    assert.equal(t2.interimText, "tomorrow");
    assert.ok(t2.speechTimestampMs >= 1000 && t2.speechTimestampMs <= 1800);

    // Tick 3: "I want tomorrow coffee" -> "tomorrow" now agreed & followed by "coffee"
    const t3 = st.update("I want tomorrow coffee", { isFinal: false, windowStartMs: 1000, windowEndMs: 2100 });
    assert.deepEqual(t3.newWords, ["tomorrow"]);
    assert.equal(t3.interimText, "coffee");

    // Final micro-pause flush commits remaining "coffee"
    const t4 = st.update("I want tomorrow coffee", { isFinal: true, windowStartMs: 1000, windowEndMs: 2400 });
    assert.deepEqual(t4.newWords, ["coffee"]);
    assert.equal(t4.interimText, "");
  });

  it("deduplicates re-decoded overlap audio after window compaction via anchorTail", () => {
    const st = new StreamingStabilizer();
    st.update("I went to buy coffee", { isFinal: false, windowStartMs: 0, windowEndMs: 2000 });
    const c1 = st.update("I went to buy coffee yesterday", { isFinal: false, windowStartMs: 0, windowEndMs: 2500 });
    assert.deepEqual(c1.newWords, ["I", "went", "to", "buy", "coffee"]);

    // Window compacts left keeping 1.4s tail ("buy coffee yesterday ...")
    st.onWindowCompact();

    // Post-compaction decode still hears "buy coffee" at left edge
    const p1 = st.update("buy coffee yesterday morning", { isFinal: false, windowStartMs: 1100, windowEndMs: 3000 });
    assert.deepEqual(p1.newWords, []); // waiting for agreement on "yesterday"
    const p2 = st.update("buy coffee yesterday morning at", { isFinal: false, windowStartMs: 1100, windowEndMs: 3300 });
    assert.deepEqual(p2.newWords, ["yesterday", "morning"]);
    assert.equal(p2.newWords.includes("buy"), false);
    assert.equal(p2.newWords.includes("coffee"), false);
  });

  it("strips [BLANK_AUDIO]/(music) while NEVER dropping real ASL words thank you, bye, okay, you", () => {
    assert.equal(sanitizeSttText("[BLANK_AUDIO]"), "");
    assert.equal(sanitizeSttText("(music)"), "");
    assert.equal(sanitizeSttText("Thanks for watching."), "");

    assert.equal(sanitizeSttText("thank you"), "thank you");
    assert.equal(sanitizeSttText("bye"), "bye");
    assert.equal(sanitizeSttText("okay"), "okay");
    assert.equal(sanitizeSttText("you"), "you");
    assert.equal(sanitizeSttText("[music] thank you (cough)"), "thank you");
  });

  it("computes max_new_tokens >= 12 for <1.0s audio to prevent Transformers.js Moonshine 0-token bug", () => {
    // 0.35s = 5600 samples, 0.5s = 8000 samples, 0.95s = 15200 samples
    assert.ok(computeMaxNewTokens(5600) >= 12);
    assert.ok(computeMaxNewTokens(8000) >= 12);
    assert.ok(computeMaxNewTokens(15200) >= 12);
    // Longer 4.5s window scales proportionally and caps at 96
    assert.equal(computeMaxNewTokens(16000 * 4.5), 36);
    assert.equal(computeMaxNewTokens(16000 * 20), 96);
  });
});

describe("SpeechController lifecycle, mic-leak prevention, stale sessionId & engine switches", () => {
  function makeMockWorker() {
    const posted = [];
    const worker = {
      posted,
      onmessage: null,
      postMessage(msg) {
        posted.push(msg);
      },
      emit(msg) {
        if (this.onmessage) this.onmessage({ data: msg });
      },
    };
    return worker;
  }

  it("stops MediaStream tracks immediately (zero mic leak) if stop() is called during pending getUserMedia", async () => {
    let resolveGUM;
    let trackStopped = false;
    const mockStream = {
      getTracks: () => [{ stop: () => { trackStopped = true; } }],
    };
    const mockWorker = makeMockWorker();

    const ctrl = new SpeechController(
      {},
      {
        createWorker: () => mockWorker,
        getUserMedia: () => new Promise((r) => { resolveGUM = r; }),
        createAudioContext: () => ({ close: async () => {} }),
        createPcmCapture: async () => ({ stop: () => {} }),
      }
    );

    const startPromise = ctrl.start();
    // User clicks Stop Mic while permission prompt is still pending
    ctrl.stop();
    resolveGUM(mockStream);
    await startPromise;

    assert.equal(trackStopped, true, "Mic track must be stopped immediately if stop() won the race");
    assert.equal(ctrl.isListening, false);
    assert.equal(ctrl.mediaStream, null);
  });

  it("discards stale worker TRANSCRIPT/INTERIM from old sessionIds and resets worker on engine switch", async () => {
    const mockWorker = makeMockWorker();
    const transcripts = [];
    const interims = [];
    const listeningStates = [];
    let mockRecInstances = [];

    class MockSpeechRecognition {
      constructor() {
        mockRecInstances.push(this);
      }
      start() {
        if (this.onstart) this.onstart();
      }
      abort() {
        this.aborted = true;
      }
    }

    let clock = 2000;
    const ctrl = new SpeechController(
      {
        onTranscript: (text, meta) => transcripts.push({ text, meta }),
        onInterim: (text, meta) => interims.push({ text, meta }),
        onListeningChange: (v) => listeningStates.push(v),
      },
      {
        now: () => clock,
        createWorker: () => mockWorker,
        getUserMedia: async () => ({ getTracks: () => [{ stop: () => {} }] }),
        createAudioContext: () => ({ close: async () => {} }),
        createPcmCapture: async () => ({ stop: () => {} }),
        getSpeechRecognition: () => MockSpeechRecognition,
      }
    );

    await ctrl.start();
    const session1 = ctrl.sessionId;
    assert.equal(ctrl.engine, "moonshine");

    // Emit valid transcript for current session
    clock = 2350;
    mockWorker.emit({
      type: "TRANSCRIPT",
      sessionId: session1,
      text: "hello",
      words: ["hello"],
      modelKey: "moonshine",
      device: "webgpu",
      inferenceMs: 85,
      audioDurationMs: 600,
      speechTimestampMs: 2150,
      isFinal: false,
    });
    assert.equal(transcripts.length, 1);
    assert.equal(transcripts[0].meta.sttLatencyMs, 200);
    assert.equal(transcripts[0].meta.sttEstimated, false);
    assert.equal(transcripts[0].meta.isCumulative, false);

    // Switch to webspeech while listening
    await ctrl.setEngine("webspeech");
    assert.equal(ctrl.engine, "webspeech");
    assert.equal(ctrl.isListening, true);
    assert.ok(mockWorker.posted.some((m) => m.type === "RESET" && m.sessionId > session1));

    // Late worker message from session1 MUST be ignored!
    mockWorker.emit({
      type: "TRANSCRIPT",
      sessionId: session1,
      text: "ghost word",
      words: ["ghost", "word"],
      modelKey: "moonshine",
      device: "webgpu",
      inferenceMs: 50,
      speechTimestampMs: 2200,
      isFinal: true,
    });
    assert.equal(transcripts.length, 1, "Stale sessionId transcript must be discarded");

    // Simulate Web Speech cumulative interim & final emission
    const rec = mockRecInstances[mockRecInstances.length - 1];
    clock = 3000;
    rec.onspeechstart();
    clock = 3180;
    rec.onresult({
      resultIndex: 0,
      results: [{ isFinal: false, 0: { transcript: "I want coffee" } }],
    });
    assert.equal(interims.length, 1);
    assert.equal(interims[0].text, "I want coffee");
    assert.equal(interims[0].meta.isCumulative, true);
    assert.equal(interims[0].meta.sttEstimated, true);

    clock = 3260;
    rec.onresult({
      resultIndex: 0,
      results: [{ isFinal: true, 0: { transcript: "I want coffee" } }],
    });
    assert.equal(transcripts.length, 2);
    assert.equal(transcripts[1].meta.engine, "webspeech");
    assert.equal(transcripts[1].meta.isCumulative, true);
    assert.equal(transcripts[1].meta.isFinal, true);
    assert.equal(transcripts[1].meta.sttLatencyMs, 260);

    ctrl.stop();
    assert.equal(rec.aborted, true);
    assert.equal(ctrl.isListening, false);
    assert.equal(listeningStates[listeningStates.length - 1], false);
  });

  it("Fix #5: StreamingAudioBuffer sanitizes NaN/negative RMS and createPcmCapture sanitizes NaN samples", async () => {
    const buf = new StreamingAudioBuffer();
    buf.pushChunk(new Float32Array([NaN, Infinity, -Infinity]), NaN, 80);
    buf.pushChunk(makeChunk(80, 0.01), -10.0, 160);
    buf.pushChunk(makeChunk(80, 0.01), Infinity, 240);
    assert.equal(Number.isFinite(buf.noiseFloor), true);
    assert.ok(buf.noiseFloor >= 0, `Expected non-negative noiseFloor, got ${buf.noiseFloor}`);
    assert.equal(Number.isFinite(buf.threshold), true);

    // Subsequent loud speech MUST still trigger VAD cleanly
    buf.pushChunk(makeChunk(80, 0.25), 0.25, 320);
    assert.equal(buf.isSpeaking, true);
    assert.ok(buf.activeSpeechMs >= 80);

    // Verify ScriptProcessor fallback in pcmCapture sanitizes NaN audio samples
    const { createPcmCapture } = await import("../src/stt/pcmCapture.js");
    const chunks = [];
    let scriptNode = null;
    const mockAudioCtx = {
      sampleRate: 16000,
      destination: {},
      createMediaStreamSource: () => ({ connect: () => {}, disconnect: () => {} }),
      createScriptProcessor: () => {
        scriptNode = { connect: () => {}, disconnect: () => {}, onaudioprocess: null };
        return scriptNode;
      },
      createGain: () => ({ gain: { value: 1 }, connect: () => {}, disconnect: () => {} }),
    };
    const cap = await createPcmCapture({
      audioCtx: mockAudioCtx,
      mediaStream: {},
      onChunk: (c) => chunks.push(c),
    });
    scriptNode.onaudioprocess({
      inputBuffer: { getChannelData: () => new Float32Array([NaN, 0.5, -Infinity, 0.5]) },
    });
    cap.stop();
    assert.equal(chunks.length, 1);
    assert.equal(Number.isFinite(chunks[0].rms), true);
    assert.ok(chunks[0].rms > 0);
    assert.equal(Number.isFinite(chunks[0].pcm[0]), true);
  });

  it("Fix #6 & #7: StreamingStabilizer never shrinks committedWords, allows single digits, gates boilerplate on low RMS, and tightens compaction anchor", () => {
    // Fix #6: transient shorter hypothesis MUST NOT duplicate committed words
    const st = new StreamingStabilizer();
    st.update("hello world how are you doing", { isFinal: false, windowStartMs: 0, windowEndMs: 1000 });
    const t2 = st.update("hello world how are you doing", { isFinal: false, windowStartMs: 0, windowEndMs: 1200 });
    assert.deepEqual(t2.newWords, ["hello", "world", "how", "are", "you"]);

    // Transient shorter decode
    const t3 = st.update("hello world", { isFinal: false, windowStartMs: 0, windowEndMs: 1400 });
    assert.deepEqual(t3.newWords, []);

    // Recovery -> must NOT re-emit "how", "are", "you"
    const t4 = st.update("hello world how are you doing today", { isFinal: false, windowStartMs: 0, windowEndMs: 1600 });
    const t5 = st.update("hello world how are you doing today", { isFinal: true, windowStartMs: 0, windowEndMs: 1800 });
    assert.deepEqual([...t4.newWords, ...t5.newWords], ["doing", "today"]);

    // Fix #7a: single digits "0"-"9" are NOT dropped
    assert.equal(sanitizeSttText("0"), "0");
    assert.equal(sanitizeSttText("5", { peakRms: 0.08, activeSpeechMs: 450 }), "5");
    assert.equal(sanitizeSttText("9"), "9");

    // Fix #7b: SILENCE_BOILERPLATE filtered at low RMS/short duration, preserved on high-RMS sustained speech
    assert.equal(sanitizeSttText("thanks for watching", { peakRms: 0.02, activeSpeechMs: 300 }), "");
    assert.equal(sanitizeSttText("thanks for watching", { peakRms: 0.09, activeSpeechMs: 900 }), "thanks for watching");

    // Fix #7c: compaction 1-word fallback tightened to first 2 words + clears stale anchorTail
    const st2 = new StreamingStabilizer();
    st2.update("I go home now", { isFinal: false });
    st2.update("I go home now", { isFinal: false }); // commits ["I", "go", "home"] -> anchorTail ["go", "home"]
    st2.onWindowCompact();
    // "home" at index 4 must NOT falsely trigger 1-word anchor strip
    const c1 = st2.update("now let us walk home together", { isFinal: true });
    assert.deepEqual(c1.newWords, ["now", "let", "us", "walk", "home", "together"]);

    // Stale anchorTail clears once post-compaction words commit without matching
    const st3 = new StreamingStabilizer();
    st3.update("buy coffee today", { isFinal: false });
    st3.update("buy coffee today", { isFinal: false }); // commits ["buy", "coffee"] -> anchorTail ["buy", "coffee"]
    st3.onWindowCompact();
    st3.update("now I drank tea", { isFinal: false });
    const p2 = st3.update("now I drank tea", { isFinal: false }); // commits ["now", "I", "drank"] without anchor match -> clears anchorTail
    assert.deepEqual(p2.newWords, ["now", "I", "drank"]);
    const p3 = st3.update("now I drank coffee", { isFinal: true });
    assert.deepEqual(p3.newWords, ["coffee"]);
  });

  it("Fix #11, #12, #13: SpeechController resumes suspended AudioContext, prevents track leak on AudioContext throw, stops Web Speech network loop, guards setEngine race, stale READY, and empty isFinal", async () => {
    // Fix #11a: _createAudioContext() throwing synchronously must stop stream tracks (0 leaks)
    let tracksStopped = 0;
    const mockWorker1 = makeMockWorker();
    const ctrl1 = new SpeechController(
      {},
      {
        createWorker: () => mockWorker1,
        getUserMedia: async () => ({
          getTracks: () => [{ stop: () => { tracksStopped++; } }],
        }),
        createAudioContext: () => {
          throw new Error("AudioContext blocked");
        },
        createPcmCapture: async () => ({ stop: () => {} }),
      }
    );
    await assert.rejects(() => ctrl1.start(), /AudioContext blocked/);
    assert.equal(tracksStopped, 1, "MediaStream tracks must be stopped when _createAudioContext throws");

    // Fix #11b: suspended AudioContext must be resumed before capture starts
    let resumedCount = 0;
    let closedCount = 0;
    const mockWorker2 = makeMockWorker();
    const ctrl2 = new SpeechController(
      {},
      {
        createWorker: () => mockWorker2,
        getUserMedia: async () => ({ getTracks: () => [{ stop: () => {} }] }),
        createAudioContext: () => ({
          state: "suspended",
          resume: async function () {
            resumedCount++;
            this.state = "running";
          },
          close: async () => {
            closedCount++;
          },
        }),
        createPcmCapture: async () => ({ stop: () => {} }),
      }
    );
    await ctrl2.start();
    assert.equal(resumedCount, 1, "Suspended AudioContext must be resumed");
    assert.equal(ctrl2.isListening, true);
    ctrl2.stop();
    assert.equal(closedCount, 1);

    // Fix #12: Web Speech network error >= 2 stops listening and preserves fallback error banner across onend
    const statuses = [];
    const srInstances = [];
    class MockSR {
      constructor() {
        srInstances.push(this);
      }
      start() {}
      abort() {
        this.aborted = true;
      }
    }
    const ctrl3 = new SpeechController(
      { onStatus: (text, state) => statuses.push({ text, state }) },
      { getSpeechRecognition: () => MockSR }
    );
    ctrl3.engine = "webspeech";
    await ctrl3.start();
    const rec = srInstances[0];
    rec.onerror({ error: "network" });
    assert.equal(ctrl3.isListening, true);
    rec.onerror({ error: "network" });
    assert.equal(ctrl3.isListening, false, "2nd network error must stop listening");
    rec.onend();
    assert.equal(ctrl3.webSpeechRestartTimer, null, "No reconnect timer may be scheduled after 2nd network error");
    assert.match(statuses[statuses.length - 1].text, /Moonshine Offline WebAI/);
    assert.equal(statuses[statuses.length - 1].state, "error");

    // Fix #13a: concurrent setEngine("whisper") late rejection must NOT clobber newer live webspeech session
    let rejectSlowWhisperGum;
    const mockWorker4 = makeMockWorker();
    const statuses4 = [];
    const ctrl4 = new SpeechController(
      { onStatus: (text, state) => statuses4.push({ text, state }) },
      {
        createWorker: () => mockWorker4,
        getUserMedia: () =>
          ctrl4.engine === "whisper"
            ? new Promise((_, rej) => { rejectSlowWhisperGum = rej; })
            : Promise.resolve({ getTracks: () => [{ stop: () => {} }] }),
        createAudioContext: () => ({ state: "running", close: async () => {} }),
        createPcmCapture: async () => ({ stop: () => {} }),
        getSpeechRecognition: () => MockSR,
      }
    );
    await ctrl4.start(); // listening on moonshine
    const pWhisper = ctrl4.setEngine("whisper").catch((e) => e);
    await ctrl4.setEngine("webspeech"); // supersedes whisper
    assert.equal(ctrl4.engine, "webspeech");
    assert.equal(ctrl4.isListening, true);
    rejectSlowWhisperGum(new Error("old whisper mic aborted"));
    await pWhisper;
    assert.equal(ctrl4.engine, "webspeech");
    assert.equal(ctrl4.isListening, true, "Superseded setEngine rejection must not clobber live webspeech session");
    ctrl4.stop();

    // Fix #13b & #13c: stale READY for wrong modelKey ignored; empty-tail INTERIM with isFinal increments utteranceId
    const interims5 = [];
    const statuses5 = [];
    const mockWorker5 = makeMockWorker();
    const ctrl5 = new SpeechController(
      {
        onInterim: (text, meta) => interims5.push({ text, meta }),
        onStatus: (text, state) => statuses5.push({ text, state }),
      },
      {
        createWorker: () => mockWorker5,
        getUserMedia: async () => ({ getTracks: () => [{ stop: () => {} }] }),
        createAudioContext: () => ({ state: "running", close: async () => {} }),
        createPcmCapture: async () => ({ stop: () => {} }),
      }
    );
    ctrl5.engine = "whisper";
    await ctrl5.start();
    const statusCountBefore = statuses5.length;
    mockWorker5.emit({ type: "READY", modelKey: "moonshine", device: "wasm" });
    assert.equal(ctrl5.loadedModelKey, null, "Stale READY for wrong modelKey must not overwrite loadedModelKey");
    assert.equal(statuses5.length, statusCountBefore, "Stale READY must not overwrite status");

    const uttBefore = ctrl5._utteranceCounter;
    mockWorker5.emit({
      type: "INTERIM",
      sessionId: ctrl5.sessionId,
      text: "",
      isFinal: true,
      speechTimestampMs: 1234,
    });
    assert.equal(interims5.length, 1);
    assert.equal(interims5[0].meta.isFinal, true);
    assert.equal(interims5[0].meta.isCumulative, false);
    assert.equal(interims5[0].meta.utteranceId, uttBefore);
    assert.equal(ctrl5._utteranceCounter, uttBefore + 1);
    ctrl5.stop();
  });
});

describe("whisper.worker.js concurrency, error recovery & empty-tail finalization (Fixes #8, #9, #10)", () => {
  it("guards concurrent INIT model switches, resets on isFinal error storm, and emits isFinal:true on empty-tail finalize", async () => {
    const { registerHooks } = await import("node:module");

    // Shared mock controller exposed via globalThis for the hooked @huggingface/transformers import
    let pipelineImpl = null;
    const HOOK_KEY = "__OPENASL_TEST_TRANSFORMERS_MOCK__";
    globalThis[HOOK_KEY] = {
      pipeline: (...args) => pipelineImpl(...args),
    };

    const hooks = registerHooks({
      resolve(specifier, context, nextResolve) {
        if (specifier === "@huggingface/transformers") {
          return { url: "mock://transformers", shortCircuit: true };
        }
        return nextResolve(specifier, context);
      },
      load(url, context, nextLoad) {
        if (url === "mock://transformers") {
          return {
            format: "module",
            shortCircuit: true,
            source: `
              export const env = { allowLocalModels: true };
              export const pipeline = (...args) => globalThis.${HOOK_KEY}.pipeline(...args);
            `,
          };
        }
        return nextLoad(url, context);
      },
    });

    const posted = [];
    const prevSelf = globalThis.self;
    globalThis.self = {
      postMessage(msg) {
        posted.push(msg);
      },
      onmessage: null,
    };

    try {
      const workerUrl = new URL(`../src/stt/whisper.worker.js?test=${Date.now()}`, import.meta.url);
      await import(workerUrl.href);
      const send = (data) => globalThis.self.onmessage({ data });

      // Fix #9: Concurrent INIT(moonshine) -> INIT(whisper) where moonshine is mid-download when whisper starts,
      // whisper finishes first, and stale moonshine resolves late
      let resolveMoonshine;
      let markMoonshineStarted;
      const moonshineStarted = new Promise((r) => { markMoonshineStarted = r; });
      let moonshineDisposed = 0;
      let whisperDisposed = 0;
      let activeCalls = [];

      const moonshinePipe = Object.assign(
        async () => {
          activeCalls.push("moonshine");
          return { text: "from moonshine" };
        },
        { dispose: async () => { moonshineDisposed++; } }
      );
      const whisperPipe = Object.assign(
        async () => {
          activeCalls.push("whisper");
          return { text: "hello world" };
        },
        { dispose: async () => { whisperDisposed++; } }
      );

      pipelineImpl = async (_task, modelId) => {
        if (modelId.includes("moonshine")) {
          markMoonshineStarted();
          return new Promise((r) => { resolveMoonshine = () => r(moonshinePipe); });
        }
        return whisperPipe;
      };

      const pInitMoon = send({ type: "INIT", modelKey: "moonshine", sessionId: 1 });
      await moonshineStarted;
      const pInitWhisp = send({ type: "INIT", modelKey: "whisper", sessionId: 1 });
      await pInitWhisp;

      // Late resolution of stale moonshine pipeline MUST dispose moonshine and NOT post READY(moonshine)
      resolveMoonshine();
      await pInitMoon;

      const readyMsgs = posted.filter((m) => m.type === "READY");
      assert.deepEqual(readyMsgs.map((m) => m.modelKey), ["whisper"]);
      assert.equal(moonshineDisposed, 1, "Stale moonshine pipeline must be disposed");
      assert.equal(whisperDisposed, 0, "Active whisper pipeline must remain live");

      // Verify subsequent inference actually runs on whisper (not overwritten by stale moonshine)
      for (let i = 0; i < 4; i++) {
        await send({
          type: "AUDIO_CHUNK",
          sessionId: 1,
          pcm: makeChunk(80, 0.1),
          rms: 0.1,
          captureTimestampMs: 500 + (i + 1) * 80,
        });
      }
      await send({ type: "FLUSH", sessionId: 1 });
      await new Promise((r) => setTimeout(r, 30));
      assert.deepEqual(activeCalls, ["whisper"], "Active pipeline must be whisper, not stale moonshine");

      // Fix #10: Empty-tail isFinal emits INTERIM { text: "", isFinal: true }
      posted.length = 0;
      pipelineImpl = async () => Object.assign(async () => ({ text: "" }), { dispose: async () => {} });
      await send({ type: "INIT", modelKey: "moonshine", sessionId: 2 });
      posted.length = 0;

      // Push 320ms speech + explicit FLUSH -> decode returns "" -> must emit INTERIM with isFinal: true
      for (let i = 0; i < 4; i++) {
        await send({
          type: "AUDIO_CHUNK",
          sessionId: 2,
          pcm: makeChunk(80, 0.1),
          rms: 0.1,
          captureTimestampMs: 1000 + (i + 1) * 80,
        });
      }
      await send({ type: "FLUSH", sessionId: 2 });
      await new Promise((r) => setTimeout(r, 30));

      const emptyFinalInterims = posted.filter(
        (m) => m.type === "INTERIM" && m.text === "" && m.isFinal === true
      );
      assert.equal(emptyFinalInterims.length, 1, "Empty-tail finalization must emit INTERIM with isFinal: true");

      // Fix #8: isFinal decode error resets utterance and does NOT trigger infinite 20ms error storm
      let throwCount = 0;
      pipelineImpl = async () =>
        Object.assign(
          async () => {
            throwCount++;
            throw new Error("WASM runtime failure");
          },
          { dispose: async () => {} }
        );
      await send({ type: "INIT", modelKey: "whisper", sessionId: 3 });
      posted.length = 0;
      throwCount = 0;

      for (let i = 0; i < 4; i++) {
        await send({
          type: "AUDIO_CHUNK",
          sessionId: 3,
          pcm: makeChunk(80, 0.1),
          rms: 0.1,
          captureTimestampMs: 2000 + (i + 1) * 80,
        });
      }
      await send({ type: "FLUSH", sessionId: 3 });
      await new Promise((r) => setTimeout(r, 120));

      const errMsgs = posted.filter((m) => m.type === "ERROR");
      assert.equal(throwCount, 1, `Expected single throw on isFinal error, got ${throwCount}`);
      assert.equal(errMsgs.length, 1, `Expected single ERROR message (no 20ms storm), got ${errMsgs.length}`);
    } finally {
      hooks.deregister();
      delete globalThis[HOOK_KEY];
      if (prevSelf === undefined) delete globalThis.self;
      else globalThis.self = prevSelf;
    }
  });
});
