/**
 * Pure, DOM-free LocalAgreement-2 word stabilizer + overlap-anchor deduplicator
 * + safe hallucination filter for sliding-window Offline WebAI STT (Moonshine / Whisper).
 */

import { TARGET_SAMPLE_RATE } from "./audioRingBuffer.js";

/**
 * Fixes the verified @huggingface/transformers@4.3.2 `_call_moonshine` `<1.0s` 0-token bug
 * where default `Math.floor(aud.length / 16000) * 6` evaluates to 0 for audio < 1.0s.
 */
export function computeMaxNewTokens(audioLength, sampleRate = TARGET_SAMPLE_RATE, minTokens = 12) {
  const audioSec = Math.max(0, Number(audioLength) || 0) / (sampleRate || TARGET_SAMPLE_RATE);
  return Math.max(minTokens, Math.min(96, Math.ceil(audioSec * 8)));
}

export function normalizeWord(word) {
  return String(word || "")
    .replace(/[’‘`]/g, "'")
    .toLowerCase()
    .replace(/^[^a-z0-9']+|[^a-z0-9']+$/g, "");
}

// Strict boilerplate silence-hallucination phrases (NEVER include real conversational ASL signs
// like "thank you", "bye", "okay", "you", "hello", "yes", "no"!)
const SILENCE_BOILERPLATE = new Set([
  "thanks for watching",
  "thank you for watching",
  "please subscribe",
  "subscribe",
  "subtitles by",
  "transcribed by",
  "silence",
  "blank audio",
  "music",
  "applause",
]);

export function sanitizeSttText(rawText, { peakRms = null, activeSpeechMs = null } = {}) {
  if (!rawText || typeof rawText !== "string") return "";

  // Strip bracketed/parenthesized/angle-bracketed non-speech tokens ([BLANK_AUDIO], (music), <|en|>)
  const stripped = rawText
    .replace(/\[[^\]]*\]|\([^)]*\)|<[^>]*>/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  if (!stripped) return "";

  const normalizedFull = stripped
    .toLowerCase()
    .replace(/[^a-z0-9'\s]/g, "")
    .replace(/\s+/g, " ")
    .trim();

  if (!normalizedFull) return "";

  const isLowConfidenceAudio =
    peakRms === null ||
    activeSpeechMs === null ||
    peakRms < 0.035 ||
    activeSpeechMs < 400;

  // Reject pure non-speech hallucination boilerplate on low-RMS / short buffers
  if (isLowConfidenceAudio && SILENCE_BOILERPLATE.has(normalizedFull)) {
    return "";
  }

  // Only reject ultra-short single-character noise (allow "i", "a", and spoken single digits "0"-"9")
  if (
    normalizedFull.length < 2 &&
    normalizedFull !== "i" &&
    normalizedFull !== "a" &&
    !/^\d$/.test(normalizedFull)
  ) {
    return "";
  }

  // Reject suspected boilerplate prefixes when RMS is low or speech is brief
  if (
    isLowConfidenceAudio &&
    (normalizedFull.startsWith("thanks for watching") ||
      normalizedFull.startsWith("thank you for watching"))
  ) {
    return "";
  }

  return stripped;
}

function lcpLength(a, b) {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a[i] === b[i]) i++;
  return i;
}

export class StreamingStabilizer {
  constructor() {
    this.reset();
  }

  reset() {
    this.committedWords = []; // normalized committed words in current window
    this.prevHypothesis = []; // normalized words from previous decode tick
    this.anchorTail = []; // last 1-2 committed words preserved across compactLeft()
  }

  /**
   * Call immediately when StreamingAudioBuffer.compactLeft() slides the window right.
   * Preserves the last 2 committed words as an overlap anchor so re-decoded left-overlap
   * audio is cleanly skipped on subsequent ticks without duplicating or dropping words.
   */
  onWindowCompact() {
    if (this.committedWords.length > 0) {
      this.anchorTail = this.committedWords.slice(-2);
    }
    this.committedWords = [];
    this.prevHypothesis = [];
  }

  _stripAnchorPrefix(displayWords, normWords) {
    if (this.anchorTail.length === 0 || normWords.length === 0) {
      return { displayWords, normWords, matched: false };
    }

    const searchLimit = Math.min(normWords.length, 8);

    // Try 2-word anchor match first
    if (this.anchorTail.length >= 2) {
      const a0 = this.anchorTail[this.anchorTail.length - 2];
      const a1 = this.anchorTail[this.anchorTail.length - 1];
      for (let i = 0; i + 1 < searchLimit; i++) {
        if (normWords[i] === a0 && normWords[i + 1] === a1) {
          return {
            displayWords: displayWords.slice(i + 2),
            normWords: normWords.slice(i + 2),
            matched: true,
          };
        }
      }
    }

    // Fallback: 1-word tail anchor match only in the first 2 words of the window
    const lastAnchor = this.anchorTail[this.anchorTail.length - 1];
    const singleLimit = Math.min(normWords.length, 2);
    for (let i = 0; i < singleLimit; i++) {
      if (normWords[i] === lastAnchor) {
        return {
          displayWords: displayWords.slice(i + 1),
          normWords: normWords.slice(i + 1),
          matched: true,
        };
      }
    }

    return { displayWords, normWords, matched: false };
  }

  update(
    rawText,
    { isFinal = false, windowStartMs = 0, windowEndMs = 0, peakRms = 0.05, activeSpeechMs = 500 } = {}
  ) {
    const cleaned = sanitizeSttText(rawText, { peakRms, activeSpeechMs });
    if (!cleaned) {
      return {
        newWords: [],
        text: "",
        interimText: "",
        speechTimestampMs: windowEndMs,
        isFinal,
      };
    }

    const rawParts = cleaned.split(/\s+/).filter(Boolean);
    const rawDisplay = [];
    const rawNorm = [];
    for (const part of rawParts) {
      const norm = normalizeWord(part);
      if (norm) {
        rawDisplay.push(part);
        rawNorm.push(norm);
      }
    }

    const { displayWords, normWords, matched: anchorMatched } = this._stripAnchorPrefix(
      rawDisplay,
      rawNorm
    );
    if (normWords.length === 0) {
      return {
        newWords: [],
        text: "",
        interimText: "",
        speechTimestampMs: windowEndMs,
        isFinal,
      };
    }

    const lcpLen = lcpLength(this.prevHypothesis, normWords);
    const alreadyCommitted = this.committedWords.length;

    let commitUpTo;
    if (isFinal) {
      commitUpTo = normWords.length;
    } else {
      // Hold back the trailing 1 word while it may still be mid-syllable ("tom" -> "tomorrow"),
      // and require LocalAgreement-2 (agreed across 2 consecutive sliding-window decodes).
      const holdbackLimit = Math.max(0, normWords.length - 1);
      const stableCount = Math.min(lcpLen, holdbackLimit);
      commitUpTo = Math.max(alreadyCommitted, stableCount);
    }

    const safeCommit = Math.min(commitUpTo, normWords.length);
    const newWords =
      safeCommit > alreadyCommitted ? displayWords.slice(alreadyCommitted, safeCommit) : [];
    const interimWords = isFinal ? [] : displayWords.slice(safeCommit);

    // Proportional speech timestamp interpolation inside [windowStartMs, windowEndMs]
    let speechTimestampMs = windowEndMs;
    if (!isFinal && normWords.length > 0 && windowEndMs >= windowStartMs) {
      const ratio = Math.min(1, Math.max(0, safeCommit / normWords.length));
      speechTimestampMs = windowStartMs + (windowEndMs - windowStartMs) * ratio;
    }

    // Never shrink committedWords on transient shorter hypotheses
    if (safeCommit > alreadyCommitted) {
      this.committedWords = normWords.slice(0, safeCommit);
      if (!anchorMatched && this.anchorTail.length > 0) {
        this.anchorTail = [];
      }
    }
    this.prevHypothesis = normWords;

    return {
      newWords,
      text: newWords.join(" "),
      interimText: interimWords.join(" "),
      speechTimestampMs,
      isFinal,
    };
  }
}
