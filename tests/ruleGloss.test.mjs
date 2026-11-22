import { describe, it } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import {
  translateRuleGloss,
  glossStreamingTokens,
  StreamingRuleGloss,
  createStreamingGloss,
} from "../src/nlp/ruleGloss.js";

const manifest = JSON.parse(fs.readFileSync(new URL("../public/assets/manifest.json", import.meta.url), "utf8"));
const knownGlosses = new Set([
  ...Object.keys(manifest.poses || {}),
  ...Object.keys(manifest.aliases || {}),
]);

describe("Batch translateRuleGloss (backward compatibility + contraction fix)", () => {
  it("preserves batch TTC behavior for preset phrases", () => {
    const p1 = translateRuleGloss("Where did you buy that coffee yesterday?");
    assert.equal(p1.mode, "ttc");
    assert.equal(p1.tokens[0], "YESTERDAY");
    assert.equal(p1.tokens[p1.tokens.length - 1], "WHERE");
    assert.ok(p1.tokens.includes("BUY"));
    assert.ok(p1.tokens.includes("COFFEE"));

    const p2 = translateRuleGloss("Hello, my name is Maya.");
    assert.deepEqual(p2.tokens, ["HELLO", "MY", "NAME", "MAYA"]);

    const p3 = translateRuleGloss("I am going to the library tomorrow.");
    assert.equal(p3.tokens[0], "TOMORROW");
    assert.ok(p3.tokens.includes("LIBRARY"));
    assert.ok(p3.tokens.includes("GO"));
  });

  it("fixes the ruleGloss.js:189 contraction-auxiliary bug for don't and didn't", () => {
    const r1 = translateRuleGloss("don't");
    assert.equal(r1.tokens.includes("DO"), false, "don't must not emit auxiliary DO");
    assert.ok(r1.tokens.includes("NOT"));

    const r2 = translateRuleGloss("didn't");
    assert.equal(r2.tokens.includes("DO"), false, "didn't must not emit auxiliary DO");
    assert.ok(r2.tokens.includes("NOT"));

    const r3 = translateRuleGloss("I don't like loud music.");
    assert.equal(r3.tokens.includes("DO"), false);
    assert.ok(r3.tokens.includes("NOT"));
    assert.ok(r3.tokens.includes("LIKE"));
  });

  it("supports options.mode === 'stream' via translateRuleGloss", () => {
    const res = translateRuleGloss("I am going to the library tomorrow", {
      mode: "stream",
      knownGlosses,
    });
    assert.equal(res.mode, "stream");
    assert.deepEqual(res.tokens, ["I", "GO", "LIBRARY", "TOMORROW"]);
  });
});

describe("glossStreamingTokens (zero-compromise PSE fast-path)", () => {
  it("expands contractions cleanly without spurious DO or PAST", () => {
    assert.deepEqual(glossStreamingTokens(["don't"]).tokens, ["NOT"]);
    assert.deepEqual(glossStreamingTokens(["didn't"]).tokens, ["NOT"]);
    assert.deepEqual(glossStreamingTokens(["I'm", "tired"]).tokens, ["I", "TIRED"]);
    assert.deepEqual(glossStreamingTokens(["it's", "good"]).tokens, ["GOOD"]);
    assert.deepEqual(glossStreamingTokens(["we're", "ready"]).tokens, ["WE", "READY"]);
    assert.deepEqual(glossStreamingTokens(["can't"]).tokens, ["CAN", "NOT"]);
    assert.deepEqual(glossStreamingTokens(["won't"]).tokens, ["FUTURE", "NOT"]);
    assert.deepEqual(glossStreamingTokens(["Maya’s", "book"]).tokens, ["MAYA", "BOOK"]);
  });

  it("normalizes numbers so digits are never silently dropped", () => {
    assert.deepEqual(glossStreamingTokens("3 apples").tokens, ["THREE", "APPLE"]);
    assert.deepEqual(glossStreamingTokens("0").tokens, ["ZERO"]);
    assert.deepEqual(glossStreamingTokens("10").tokens, ["TEN"]);
    assert.deepEqual(glossStreamingTokens("2026").tokens, ["TWO", "ZERO", "TWO", "SIX"]);
    assert.deepEqual(glossStreamingTokens("21").tokens, ["TWENTY", "ONE"]);
  });

  it("protects proper nouns and natural -s/-ed/-ing words from over-stemming", () => {
    assert.deepEqual(
      glossStreamingTokens("Hello, my name is Maya.", { knownGlosses }).tokens,
      ["HELLO", "MY", "NAME", "MAYA"]
    );
    assert.deepEqual(
      glossStreamingTokens("James went to Paris", { knownGlosses }).tokens,
      ["JAMES", "GO", "PARIS"]
    );
    assert.deepEqual(
      glossStreamingTokens("bus yes tired morning always", { knownGlosses }).tokens,
      ["BUS", "YES", "TIRED", "MORNING", "ALWAYS"]
    );
  });

  it("lemmatizes inflections and uses knownGlosses before synonym collapse", () => {
    assert.deepEqual(
      glossStreamingTokens("running studies bought making watches", { knownGlosses }).tokens,
      ["RUN", "STUDY", "BUY", "MAKE", "WATCH"]
    );
    assert.deepEqual(
      glossStreamingTokens("exhausted laptop exam bye", { knownGlosses }).tokens,
      ["TIRED", "COMPUTER", "TEST", "GOODBYE"]
    );

    // If Phase 2 adds an exact pose for EXHAUSTED, knownGlosses preserves it instead of collapsing to TIRED
    const expandedVocab = new Set([...knownGlosses, "EXHAUSTED"]);
    assert.deepEqual(
      glossStreamingTokens("exhausted", { knownGlosses: expandedVocab }).tokens,
      ["EXHAUSTED"]
    );
  });

  it("strips bracketed STT noise and punctuation cleanly", () => {
    assert.deepEqual(glossStreamingTokens("Hello, wait—what?!").tokens, ["HELLO", "WAIT", "WHAT"]);
    assert.deepEqual(glossStreamingTokens("well-known").tokens, ["WELL", "KNOW"]);
    assert.deepEqual(glossStreamingTokens("twenty-one").tokens, ["TWENTY", "ONE"]);
    assert.deepEqual(glossStreamingTokens("(music) [BLANK_AUDIO] uh um hmm").tokens, []);
  });
});

describe("StreamingRuleGloss state machine (hypotheses, deltas, split bigrams, benchmark)", () => {
  it("streams words incrementally on monotonic hypotheses without duplicates", () => {
    const sg = createStreamingGloss({ knownGlosses });
    const allTokens = [];

    const steps = [
      { text: "I", isFinal: false },
      { text: "I am", isFinal: false },
      { text: "I am going", isFinal: false },
      { text: "I am going home", isFinal: false },
      { text: "I am going home", isFinal: true },
    ];

    for (const s of steps) {
      const r = sg.ingestHypothesis(s.text, { isFinal: s.isFinal, utteranceId: 1 });
      allTokens.push(...r.tokens);
    }

    assert.deepEqual(allTokens, ["I", "GO", "HOME"]);
  });

  it("handles retracted/shrinking interims and mid-prefix STT revisions without re-signing", () => {
    const sg = new StreamingRuleGloss({ knownGlosses });

    const r1 = sg.ingestHypothesis("I want to go", { isFinal: false, utteranceId: 1 });
    assert.deepEqual(r1.tokens, ["I", "WANT"]);

    // Retracted / shrunk interim -> zero emissions, retracted = true
    const r2 = sg.ingestHypothesis("I want", { isFinal: false, utteranceId: 1 });
    assert.equal(r2.retracted, true);
    assert.deepEqual(r2.tokens, []);

    // Extended again -> emits only new words
    const r3 = sg.ingestHypothesis("I want to go home", { isFinal: true, utteranceId: 1 });
    assert.deepEqual(r3.tokens, ["GO", "HOME"]);

    // Mid-prefix revision in a fresh utterance ("I want two" -> "I want to buy")
    const r4 = sg.ingestHypothesis("I want two", { isFinal: false, utteranceId: 2 });
    assert.deepEqual(r4.tokens, ["I", "WANT"]);
    const r5 = sg.ingestHypothesis("I want to buy", { isFinal: true, utteranceId: 2 });
    assert.deepEqual(r5.tokens, ["BUY"]);
  });

  it("handles zero-wait split bigrams and 4-word micro-prefix buffer across separate emissions", () => {
    const sg = createStreamingGloss({ knownGlosses });

    // Split "thank" then "you" -> THANKYOU immediately, then "you" suppressed!
    const a1 = sg.ingestDelta("thank");
    assert.deepEqual(a1.tokens, ["THANKYOU"]);
    const a2 = sg.ingestDelta("you");
    assert.deepEqual(a2.tokens, []);

    // Split "excuse" then "me" -> EXCUSE, then "me" suppressed
    assert.deepEqual(sg.ingestDelta("excuse").tokens, ["EXCUSE"]);
    assert.deepEqual(sg.ingestDelta("me").tokens, []);

    // Split "work" then "on" -> WORK, then "on" suppressed
    assert.deepEqual(sg.ingestDelta("work").tokens, ["WORK"]);
    assert.deepEqual(sg.ingestDelta("on").tokens, []);

    // Micro-prefix "last" + "night" -> YESTERDAY NIGHT
    assert.deepEqual(sg.ingestDelta("last").tokens, []);
    assert.deepEqual(sg.ingestDelta("night").tokens, ["YESTERDAY", "NIGHT"]);

    // Micro-prefix "it" + "is" + "good" -> GOOD (dummy "it is" dropped)
    assert.deepEqual(sg.ingestDelta("it").tokens, []);
    assert.deepEqual(sg.ingestDelta("is").tokens, []);
    assert.deepEqual(sg.ingestDelta("good").tokens, ["GOOD"]);

    // Held micro-prefix flushed on flush()
    assert.deepEqual(sg.ingestDelta("this").tokens, []);
    assert.deepEqual(sg.flush().tokens, ["THIS"]);
  });

  it("deduplicates 1-3 word overlaps across ingestDelta boundaries", () => {
    const sg = createStreamingGloss({ knownGlosses });
    const out = [];
    out.push(...sg.ingestDelta("I want coffee").tokens);
    out.push(...sg.ingestDelta("want coffee tomorrow morning").tokens);
    out.push(...sg.ingestDelta("tomorrow morning please", { isFinal: true }).tokens);
    assert.deepEqual(out, ["I", "WANT", "COFFEE", "TOMORROW", "MORNING", "PLEASE"]);
  });

  it("meets ultra-low-latency streaming benchmark (< 0.25ms/word avg across 100 words)", () => {
    const sg = createStreamingGloss({ knownGlosses });
    const words = [
      "hello", "my", "name", "is", "Maya", "I", "am", "working", "on", "a",
      "big", "project", "today", "and", "don't", "want", "to", "miss", "3", "meetings",
    ];
    const t0 = performance.now();
    for (let rep = 0; rep < 5; rep++) {
      for (let i = 0; i < words.length; i++) {
        sg.ingestDelta(words[i], { isFinal: i === words.length - 1 });
      }
      sg.reset();
    }
    const elapsedMs = performance.now() - t0;
    const avgPerWord = elapsedMs / 100;
    assert.ok(
      avgPerWord < 0.25,
      `Expected < 0.25ms/word avg, got ${avgPerWord.toFixed(4)}ms/word (total ${elapsedMs.toFixed(2)}ms)`
    );
  });

  it("Fix #1: resets ingestDelta overlap and suppressNextLower across utteranceId and isFinal boundaries", () => {
    const sg = createStreamingGloss({ knownGlosses });

    // Repeated single-word greeting across separate utterances (with utteranceId)
    assert.deepEqual(sg.ingestDelta("Hello", { isFinal: true, utteranceId: 1 }).tokens, ["HELLO"]);
    assert.deepEqual(sg.ingestDelta("Hello", { isFinal: true, utteranceId: 2 }).tokens, ["HELLO"]);

    // Repeated greeting even when utteranceId is omitted (isFinal alone resets)
    assert.deepEqual(sg.ingestDelta("Hello", { isFinal: true }).tokens, ["HELLO"]);

    // Repeated bigram across utterances
    assert.deepEqual(sg.ingestDelta("Thank you", { isFinal: true, utteranceId: 3 }).tokens, ["THANKYOU"]);
    assert.deepEqual(sg.ingestDelta("Thank you", { isFinal: true, utteranceId: 4 }).tokens, ["THANKYOU"]);

    // Cross-utterance boundary word ("water" ending utt 5, starting utt 6)
    assert.deepEqual(sg.ingestDelta("I want water", { isFinal: true, utteranceId: 5 }).tokens, [
      "I",
      "WANT",
      "WATER",
    ]);
    assert.deepEqual(sg.ingestDelta("water please", { isFinal: true, utteranceId: 6 }).tokens, [
      "WATER",
      "PLEASE",
    ]);

    // suppressNextLower ("work" -> "on", "thank" -> "you") MUST NOT leak across isFinal
    assert.deepEqual(sg.ingestDelta("I go to work", { isFinal: true, utteranceId: 7 }).tokens, [
      "I",
      "GO",
      "WORK",
    ]);
    assert.deepEqual(sg.ingestDelta("on Monday", { isFinal: true, utteranceId: 8 }).tokens, [
      "ON",
      "MONDAY",
    ]);

    assert.deepEqual(sg.ingestDelta("thank", { isFinal: true, utteranceId: 9 }).tokens, ["THANKYOU"]);
    assert.deepEqual(sg.ingestDelta("you are nice", { isFinal: true, utteranceId: 10 }).tokens, [
      "YOU",
      "NICE",
    ]);

    // utteranceId jump flushes held pendingPrefix before processing new utterance
    assert.deepEqual(sg.ingestDelta("this", { isFinal: false, utteranceId: 11 }).tokens, []);
    assert.deepEqual(sg.ingestDelta("hello", { isFinal: true, utteranceId: 12 }).tokens, [
      "THIS",
      "HELLO",
    ]);
  });

  it("Fix #2: preserves proper noun protection on mid-utterance single-word streaming deltas", () => {
    const sg = createStreamingGloss({ knownGlosses });

    assert.deepEqual(sg.ingestDelta("I met", { isFinal: false, utteranceId: 1 }).tokens, ["I", "MEET"]);
    assert.deepEqual(sg.ingestDelta("Ahmed", { isFinal: true, utteranceId: 1 }).tokens, ["AHMED"]);

    assert.deepEqual(sg.ingestDelta("I saw", { isFinal: false, utteranceId: 2 }).tokens, ["I", "SEE"]);
    assert.deepEqual(sg.ingestDelta("Brooks", { isFinal: true, utteranceId: 2 }).tokens, ["BROOKS"]);

    // Utterance-start capitalized verb still stems normally
    assert.deepEqual(sg.ingestDelta("Running", { isFinal: true, utteranceId: 3 }).tokens, ["RUN"]);
  });

  it("Fix #3: handles ingestHypothesis anchored retraction, shrink->regrow divergence, pure shrink, and utteranceId jump", () => {
    // 3a: Anchored retraction where anchored < committedCount ("I want to go now" -> "I wanna go home")
    const sg1 = createStreamingGloss({ knownGlosses });
    const h1 = sg1.ingestHypothesis("I want to go now", { isFinal: false, utteranceId: 1 });
    assert.deepEqual(h1.tokens, ["I", "WANT", "GO"]);
    const h2 = sg1.ingestHypothesis("I wanna go home", { isFinal: false, utteranceId: 1 });
    assert.equal(h2.retracted, true);
    assert.deepEqual(h2.tokens, []);
    const h3 = sg1.ingestHypothesis("I wanna go home", { isFinal: true, utteranceId: 1 });
    assert.deepEqual(h3.tokens, ["HOME"]);

    // 3b: Pure transient shrink MUST NOT duplicate committed words
    const sg2 = createStreamingGloss({ knownGlosses });
    const p1 = sg2.ingestHypothesis("I want to go", { isFinal: false, utteranceId: 1 });
    const p2 = sg2.ingestHypothesis("I want to", { isFinal: false, utteranceId: 1 });
    const p3 = sg2.ingestHypothesis("I want to go home", { isFinal: true, utteranceId: 1 });
    assert.deepEqual([...p1.tokens, ...p2.tokens, ...p3.tokens], ["I", "WANT", "GO", "HOME"]);

    // 3c: Shrink -> regrow divergence emits genuinely new words without duplicating shared prefix
    const sg3 = createStreamingGloss({ knownGlosses });
    const d1 = sg3.ingestHypothesis("I want to go home", { isFinal: false, utteranceId: 1 });
    const d2 = sg3.ingestHypothesis("I want pizza", { isFinal: false, utteranceId: 1 });
    const d3 = sg3.ingestHypothesis("I want pizza right", { isFinal: false, utteranceId: 1 });
    const d4 = sg3.ingestHypothesis("I want pizza right now", { isFinal: true, utteranceId: 1 });
    assert.deepEqual([...d1.tokens, ...d2.tokens, ...d3.tokens, ...d4.tokens], [
      "I",
      "WANT",
      "GO",
      "PIZZA",
      "NOW",
    ]);

    // 3d: Shrink divergence directly on isFinal emits the new divergent word
    const sg4 = createStreamingGloss({ knownGlosses });
    const f1 = sg4.ingestHypothesis("I want to go home", { isFinal: false, utteranceId: 1 });
    const f2 = sg4.ingestHypothesis("I want pizza", { isFinal: true, utteranceId: 1 });
    assert.deepEqual([...f1.tokens, ...f2.tokens], ["I", "WANT", "GO", "PIZZA"]);

    // 3e: utteranceId jump flushes uncommitted held-back tail words from prior utterance
    const sg5 = createStreamingGloss({ knownGlosses });
    const j1 = sg5.ingestHypothesis("Hello friend", { isFinal: false, utteranceId: 1 });
    assert.deepEqual(j1.tokens, ["HELLO"]);
    const j2 = sg5.ingestHypothesis("Goodbye now", { isFinal: true, utteranceId: 2 });
    assert.deepEqual(j2.tokens, ["FRIEND", "GOODBYE", "NOW"]);
  });

  it("Fix #4: normalizes currency/decimal numbers and splits unspaced internal punctuation cleanly", () => {
    assert.deepEqual(
      glossStreamingTokens("hello,world wait...what good/bad").tokens,
      ["HELLO", "WORLD", "WAIT", "WHAT", "GOOD", "BAD"]
    );
    assert.deepEqual(
      glossStreamingTokens("$42.50").tokens,
      ["FORTY", "TWO", "POINT", "FIVE", "ZERO"]
    );
    assert.deepEqual(
      glossStreamingTokens("$1,250.99").tokens,
      ["ONE", "TWO", "FIVE", "ZERO", "POINT", "NINE", "NINE"]
    );
    assert.deepEqual(glossStreamingTokens("€9.5").tokens, ["NINE", "POINT", "FIVE"]);
    assert.deepEqual(glossStreamingTokens("£100").tokens, ["ONE", "ZERO", "ZERO"]);

    // Verify zero raw punctuation or digits leak into emitted tokens
    const mixed = glossStreamingTokens("It costs $42.50 or €1,250.99 good/bad").tokens;
    for (const tok of mixed) {
      assert.match(tok, /^[A-Z]+$/, `Expected pure alpha gloss token, got "${tok}"`);
    }
  });
});
