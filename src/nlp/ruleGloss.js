import nlp from "compromise";

/**
 * Pure English grammatical glue that does NOT exist as a sign in ASL.
 * IMPORTANT: Never put real ASL content words here (like, have, can, man, look, listen,
 * on, in, with, from, about, this, that, must, should, will, just, well, etc.).
 */
const PURE_STOP_WORDS = new Set([
  // English articles (ASL does not use a / an / the)
  "a", "an", "the",
  // Pure copula "to be" (ASL omits linking verbs: "I am tired" -> "I TIRED")
  "am", "is", "are", "was", "were", "be", "been", "being",
  // Pure infinitive / genitive glue
  "to", "of", "at",
  // Pure hesitation noises
  "uh", "um", "er", "ah", "hmm", "uhh", "umm", "hm", "mm", "oh",
]);

// Auxiliaries dropped ONLY when followed by a verb/negation (e.g. "do not", "did you buy"),
// or as filler auxiliaries in streaming PSE mode.
const AUX_DO = new Set(["do", "does", "did"]);

const INTENSIFIER_PROFANITY = new Set([
  "fucking", "fuckin", "freaking", "friggin", "damn", "hella",
]);

const TIME_WORDS = new Set([
  "today", "tomorrow", "yesterday", "now", "later", "soon", "before", "after",
  "always", "sometimes", "morning", "afternoon", "evening", "night", "tonight",
  "week", "month", "year", "weekend", "past", "future",
  "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
]);

const WH_WORDS = new Set(["who", "what", "where", "when", "why", "how", "which"]);
const NEG_WORDS = new Set(["not", "never", "no", "nothing", "none", "neither"]);

const PRONOUN_MAP = {
  i: "I",
  me: "I",
  my: "MY",
  mine: "MY",
  you: "YOU",
  your: "YOUR",
  yours: "YOUR",
  he: "HE",
  him: "HE",
  his: "HIS",
  she: "SHE",
  her: "HER",
  hers: "HER",
  we: "WE",
  us: "WE",
  our: "OUR",
  ours: "OUR",
  they: "THEY",
  them: "THEY",
  their: "THEIR",
  theirs: "THEIR",
};

const POSSESSIVES = new Set(["my", "your", "his", "her", "our", "their"]);

// Map English synonyms to canonical ASL glosses so we sign instead of fingerspelling
const ASL_SYNONYMS = {
  tough: "HARD",
  difficult: "HARD",
  challenging: "HARD",
  rough: "HARD",
  task: "WORK",
  assignment: "WORK",
  job: "WORK",
  gig: "WORK",
  awesome: "GOOD",
  amazing: "GOOD",
  great: "GOOD",
  wonderful: "GOOD",
  ok: "GOOD",
  okay: "GOOD",
  cool: "GOOD",
  awful: "BAD",
  terrible: "BAD",
  horrible: "BAD",
  huge: "BIG",
  massive: "BIG",
  giant: "BIG",
  tiny: "SMALL",
  little: "SMALL",
  quick: "FAST",
  rapid: "FAST",
  mad: "ANGRY",
  furious: "ANGRY",
  scared: "AFRAID",
  terrified: "AFRAID",
  frightened: "AFRAID",
  frighten: "AFRAID",
  glad: "HAPPY",
  excited: "HAPPY",
  upset: "SAD",
  kid: "CHILD",
  buddy: "FRIEND",
  pal: "FRIEND",
  begin: "START",
  done: "FINISH",
  finished: "FINISH",
  speak: "TALK",
  chat: "TALK",
  purchase: "BUY",
  hi: "HELLO",
  hey: "HELLO",
  bye: "GOODBYE",
  thanks: "THANKYOU",
  thank: "THANKYOU",
  thx: "THANKYOU",
  ty: "THANKYOU",
  pls: "PLEASE",
  plz: "PLEASE",
  mom: "MOTHER",
  mommy: "MOTHER",
  dad: "FATHER",
  daddy: "FATHER",
  yeah: "YES",
  yep: "YES",
  yup: "YES",
  nope: "NO",
  nah: "NO",
  bro: "BROTHER",
  dude: "MAN",
  guys: "PEOPLE",
  folks: "PEOPLE",
  everyone: "PEOPLE",
  gonna: "FUTURE",
  wanna: "WANT",
  gotta: "NEED",
  exhausted: "TIRED",
  sleepy: "TIRED",
  laptop: "COMPUTER",
  pc: "COMPUTER",
  cellphone: "PHONE",
  smartphone: "PHONE",
  exam: "TEST",
  quiz: "TEST",
  ill: "SICK",
  correct: "RIGHT",
  mistake: "WRONG",
  error: "WRONG",
};

// Multi-word English idioms & time expressions normalized before batch clause parsing
const PHRASE_REWRITES = [
  [/\blast\s+night\b/gi, "YESTERDAY NIGHT"],
  [/\bthis\s+morning\b/gi, "TODAY MORNING"],
  [/\btonight\b/gi, "TODAY NIGHT"],
  [/\bright\s+now\b/gi, "NOW"],
  [/\bnext\s+week\b/gi, "FUTURE WEEK"],
  [/\blast\s+week\b/gi, "PAST WEEK"],
  [/\bthank\s+you\b/gi, "THANKYOU"],
  [/\bexcuse\s+me\b/gi, "EXCUSE"],
  [/\bice\s+cream\b/gi, "ICECREAM"],
  [/\bwork\s+on\b/gi, "WORK"],
  // "so fucking tough" -> "REALLY tough" (keeps emphasis without dropping the adjective)
  [/\b(?:so\s+)?(?:fucking|fuckin|freaking|friggin|damn|hella)\b/gi, "REALLY"],
  // Dummy pronoun "it is / it was / it's" -> drop only the dummy "it is"
  [/\bit(?:'s|\s+is|\s+was|\s+were)\b/gi, ""],
];

// ---- Streaming Zero-Compromise PSE Tables ----

const CONTRACTIONS = {
  "don't": ["not"],
  "doesn't": ["not"],
  "didn't": ["not"],
  "isn't": ["not"],
  "aren't": ["not"],
  "wasn't": ["not"],
  "weren't": ["not"],
  "ain't": ["not"],
  "haven't": ["not"],
  "hasn't": ["not"],
  "hadn't": ["not"],
  "won't": ["future", "not"],
  "can't": ["can", "not"],
  cannot: ["can", "not"],
  "couldn't": ["can", "not"],
  "shouldn't": ["should", "not"],
  "wouldn't": ["not"],
  "i'm": ["i"],
  "you're": ["you"],
  "he's": ["he"],
  "she's": ["she"],
  "we're": ["we"],
  "they're": ["they"],
  "it's": [],
  "that's": ["that"],
  "there's": ["there"],
  "what's": ["what"],
  "who's": ["who"],
  "where's": ["where"],
  "how's": ["how"],
  "when's": ["when"],
  "let's": ["we"],
  "y'all": ["you", "all"],
};

const DIGIT_WORDS = [
  "ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE",
];

const TEEN_WORDS = {
  10: "TEN",
  11: "ELEVEN",
  12: "TWELVE",
  13: "THIRTEEN",
  14: "FOURTEEN",
  15: "FIFTEEN",
  16: "SIXTEEN",
  17: "SEVENTEEN",
  18: "EIGHTEEN",
  19: "NINETEEN",
};

const TENS_WORDS = {
  20: "TWENTY",
  30: "THIRTY",
  40: "FORTY",
  50: "FIFTY",
  60: "SIXTY",
  70: "SEVENTY",
  80: "EIGHTY",
  90: "NINETY",
};

const IRREGULAR_LEMMAS = {
  // Irregular verbs
  went: "go",
  gone: "go",
  going: "go",
  goes: "go",
  bought: "buy",
  buying: "buy",
  ate: "eat",
  eaten: "eat",
  eating: "eat",
  drank: "drink",
  drunk: "drink",
  drinking: "drink",
  slept: "sleep",
  sleeping: "sleep",
  ran: "run",
  running: "run",
  came: "come",
  coming: "come",
  saw: "see",
  seen: "see",
  seeing: "see",
  said: "say",
  saying: "say",
  told: "tell",
  telling: "tell",
  took: "take",
  taken: "take",
  taking: "take",
  gave: "give",
  given: "give",
  giving: "give",
  got: "get",
  gotten: "get",
  getting: "get",
  made: "make",
  making: "make",
  knew: "know",
  known: "know",
  thought: "think",
  thinking: "think",
  found: "find",
  finding: "find",
  left: "leave",
  leaving: "leave",
  felt: "feel",
  feeling: "feel",
  brought: "bring",
  bringing: "bring",
  wrote: "write",
  written: "write",
  writing: "write",
  read: "read",
  reading: "read",
  spoke: "speak",
  spoken: "speak",
  speaking: "speak",
  taught: "teach",
  teaching: "teach",
  learned: "learn",
  learnt: "learn",
  learning: "learn",
  understood: "understand",
  understanding: "understand",
  forgot: "forget",
  forgotten: "forget",
  forgetting: "forget",
  built: "build",
  building: "build",
  sent: "send",
  sending: "send",
  paid: "pay",
  paying: "pay",
  sold: "sell",
  selling: "sell",
  met: "meet",
  meeting: "meet",
  heard: "hear",
  hearing: "hear",
  drove: "drive",
  driven: "drive",
  driving: "drive",
  broke: "break",
  broken: "break",
  breaking: "break",
  lost: "lose",
  losing: "lose",
  put: "put",
  putting: "put",
  sat: "sit",
  sitting: "sit",
  stood: "stand",
  standing: "stand",
  began: "begin",
  begun: "begin",
  beginning: "begin",
  tried: "try",
  trying: "try",
  studied: "study",
  studies: "study",
  studying: "study",
  working: "work",
  worked: "work",
  living: "live",
  lived: "live",
  loving: "love",
  loved: "love",
  liking: "like",
  liked: "like",
  using: "use",
  used: "use",
  changing: "change",
  changed: "change",
  closing: "close",
  closed: "close",
  creating: "create",
  created: "create",
  continuing: "continue",
  continued: "continue",
  purchasing: "purchase",
  purchased: "purchase",
  stopping: "stop",
  stopped: "stop",
  planning: "plan",
  planned: "plan",
  // Irregular plurals & comparatives
  children: "child",
  men: "man",
  women: "woman",
  people: "people",
  persons: "person",
  better: "good",
  best: "good",
  worse: "bad",
  worst: "bad",
  bigger: "big",
  biggest: "big",
  smaller: "small",
  smallest: "small",
  faster: "fast",
  fastest: "fast",
  slower: "slow",
  slowest: "slow",
  older: "old",
  oldest: "old",
  younger: "young",
  youngest: "young",
  longer: "long",
  longest: "long",
  shorter: "short",
  shortest: "short",
  harder: "hard",
  hardest: "hard",
  easier: "easy",
  easiest: "easy",
  happier: "happy",
  happiest: "happy",
  sadder: "sad",
  saddest: "sad",
  angrier: "angry",
  angriest: "angry",
  hotter: "hot",
  hottest: "hot",
  colder: "cold",
  coldest: "cold",
  busier: "busy",
  busiest: "busy",
  tonight: "today",
};

// Words that end in -s/-ed/-ing naturally and must NOT be stripped by morphological rules
const PROTECTED_WORDS = new Set([
  "bus", "yes", "this", "his", "us", "class", "less", "news", "always",
  "sometimes", "morning", "evening", "nothing", "something", "everything",
  "anything", "thing", "bring", "sing", "ring", "king", "wing", "string",
  "spring", "tired", "excited", "scared", "afraid", "need", "bed", "red",
  "fed", "led", "wed", "speed", "seed", "feed", "good", "food", "wood",
  "mood", "blood", "flood", "cold", "old", "bold", "gold", "hold", "fold",
  "sold", "told", "world", "child", "wild", "mild", "friend", "end", "send",
  "spend", "lend", "bend", "mind", "find", "kind", "wind", "blind", "behind",
  "around", "sound", "ground", "pound", "round", "found", "stand", "hand",
  "land", "band", "sand", "hard", "card", "yard", "word", "bird", "third",
  "hundred", "thousand", "wednesday", "thanks", "christmas", "business",
  "address", "dress", "press", "stress", "guess", "mess", "bless", "cross",
  "boss", "loss", "toss", "glass", "grass", "pass", "mass", "gas", "has",
  "was", "plus", "thus", "focus", "campus", "status", "virus", "minus",
  "bonus", "genius", "serious", "curious", "nervous", "famous", "various",
  "previous", "obvious", "delicious", "dangerous", "jealous", "generous",
  "paris", "james", "chris", "thomas", "charles", "lucas", "marcus", "alexis",
  "luis", "carlos", "dennis", "lewis", "travis", "davis", "harris", "jones",
  "williams", "miles", "giles", "niles", "jules", "moses",
]);

const MICRO_PREFIXES = new Set(["it", "last", "next", "this"]);

const ZERO_WAIT_SUPPRESS_NEXT = {
  thank: "you",
  excuse: "me",
  work: "on",
};

const SAME_SLICE_BIGRAMS = new Map([
  ["last night", ["YESTERDAY", "NIGHT"]],
  ["this morning", ["TODAY", "MORNING"]],
  ["right now", ["NOW"]],
  ["next week", ["FUTURE", "WEEK"]],
  ["last week", ["PAST", "WEEK"]],
  ["thank you", ["THANKYOU"]],
  ["excuse me", ["EXCUSE"]],
  ["ice cream", ["ICECREAM"]],
  ["work on", ["WORK"]],
  ["it is", []],
  ["it was", []],
  ["it were", []],
  ["want to", ["WANT"]],
  ["have to", ["NEED"]],
]);

function normalizeRawToken(raw) {
  return String(raw || "")
    .replace(/[’‘`]/g, "'")
    .replace(/^[^a-zA-Z0-9'$€£]+|[^a-zA-Z0-9']+$/g, "")
    .replace(/^[$€£]+/, "");
}

export function tokenizeRaw(text) {
  if (!text || typeof text !== "string") return [];
  const cleaned = text
    .replace(/[’‘`]/g, "'")
    .replace(/\[[^\]]*\]|\([^)]*\)|<[^>]*>/g, " ")
    .replace(/[—–]/g, " ")
    .replace(/\.{2,}/g, " ")
    .replace(/\//g, " ")
    .replace(/(?<!\d),|,(?!\d)/g, " ")
    .trim();
  if (!cleaned) return [];

  const parts = cleaned.split(/\s+/);
  const tokens = [];
  for (const part of parts) {
    // Split hyphenated compounds (e.g. "twenty-one", "well-known")
    const subparts = part.includes("-") ? part.split("-") : [part];
    for (const sub of subparts) {
      const tok = normalizeRawToken(sub);
      if (tok) tokens.push(tok);
    }
  }
  return tokens;
}

function expandIntegerDigits(digitsStr) {
  const n = Number(digitsStr);
  if (!Number.isFinite(n)) return null;

  if (n >= 0 && n <= 9 && digitsStr.length === 1) {
    return [DIGIT_WORDS[n]];
  }
  if (n >= 10 && n <= 19 && digitsStr.length === 2) {
    return [TEEN_WORDS[n]];
  }
  if (n >= 20 && n <= 99 && digitsStr.length === 2) {
    const tens = Math.floor(n / 10) * 10;
    const ones = n % 10;
    return ones === 0 ? [TENS_WORDS[tens]] : [TENS_WORDS[tens], DIGIT_WORDS[ones]];
  }
  // Multi-digit numbers (e.g. 2026, 100, 05, 1250) -> digit words so nothing is dropped
  const out = [];
  for (let i = 0; i < digitsStr.length; i++) {
    const d = digitsStr.charCodeAt(i) - 48;
    if (d >= 0 && d <= 9) out.push(DIGIT_WORDS[d]);
  }
  return out;
}

function normalizeNumberToken(lower) {
  const cleaned = String(lower || "").replace(/^[$€£]+/, "");
  // Match pure integers, comma-grouped numbers, decimals, or ordinals
  const m = /^(\d+(?:,\d{3})*)(?:\.(\d+))?(?:st|nd|rd|th)?$/.exec(cleaned);
  if (!m) return null;

  const intDigits = m[1].replace(/,/g, "");
  const decDigits = m[2] || null;
  const intWords = expandIntegerDigits(intDigits);
  if (!intWords) return null;

  if (!decDigits) {
    return intWords;
  }

  const out = [...intWords, "POINT"];
  for (let i = 0; i < decDigits.length; i++) {
    const d = decDigits.charCodeAt(i) - 48;
    if (d >= 0 && d <= 9) out.push(DIGIT_WORDS[d]);
  }
  return out;
}

function expandContractionToken(rawToken) {
  const lower = rawToken.toLowerCase();
  if (Object.prototype.hasOwnProperty.call(CONTRACTIONS, lower)) {
    return CONTRACTIONS[lower].map((w) => ({ raw: w, lower: w, isProper: false }));
  }

  // Generic contraction / possessive suffixes
  if (lower.endsWith("'ve") && rawToken.length > 3) {
    const baseRaw = rawToken.slice(0, -3);
    return [{ raw: baseRaw, lower: baseRaw.toLowerCase(), isProper: isCapitalizedProper(baseRaw) }];
  }
  if (lower.endsWith("'d") && rawToken.length > 2) {
    const baseRaw = rawToken.slice(0, -2);
    return [{ raw: baseRaw, lower: baseRaw.toLowerCase(), isProper: isCapitalizedProper(baseRaw) }];
  }
  if (lower.endsWith("'ll") && rawToken.length > 3) {
    const baseRaw = rawToken.slice(0, -3);
    return [
      { raw: baseRaw, lower: baseRaw.toLowerCase(), isProper: isCapitalizedProper(baseRaw) },
      { raw: "future", lower: "future", isProper: false },
    ];
  }
  if (lower.endsWith("'s") && rawToken.length > 2) {
    const baseRaw = rawToken.slice(0, -2);
    return [{ raw: baseRaw, lower: baseRaw.toLowerCase(), isProper: isCapitalizedProper(baseRaw) }];
  }

  return null;
}

function isCapitalizedProper(raw, indexInSlice = 1) {
  if (!raw || raw.length < 2) return false;
  // Mid-sentence capitalized word (e.g. "James", "Paris", "Maya") is a strong proper-noun signal.
  // Also protect capitalized non-all-caps words at index 0 if they aren't common words.
  const startsUpper = raw[0] >= "A" && raw[0] <= "Z";
  const hasLower = /[a-z]/.test(raw.slice(1));
  if (!startsUpper || !hasLower) return false;
  if (indexInSlice > 0) return true;
  return false;
}

function stemMorphological(lower, knownGlosses) {
  if (PROTECTED_WORDS.has(lower) || lower.length <= 3) return lower;

  // -ies -> -y (studies -> study, stories -> story)
  if (lower.endsWith("ies") && lower.length > 4) {
    return `${lower.slice(0, -3)}y`;
  }

  // -ing
  if (lower.endsWith("ing") && lower.length > 5) {
    let stem = lower.slice(0, -3);
    // Double consonant (running -> run, stopping -> stop)
    if (
      stem.length > 2 &&
      stem[stem.length - 1] === stem[stem.length - 2] &&
      !"lsz".includes(stem[stem.length - 1])
    ) {
      stem = stem.slice(0, -1);
    }
    if (knownGlosses?.has(stem.toUpperCase())) return stem;
    if (knownGlosses?.has(`${stem}E`.toUpperCase())) return `${stem}e`;
    // Restore silent e for common consonant-vowel-consonant patterns (making -> make, taking -> take)
    if (/[aeiou][^aeiouwxy]$/.test(stem) && stem.length >= 3 && !knownGlosses?.has(stem.toUpperCase())) {
      const withE = `${stem}e`;
      if (ASL_SYNONYMS[withE] || knownGlosses?.has(withE.toUpperCase())) {
        return withE;
      }
    }
    return stem;
  }

  // -ied -> -y
  if (lower.endsWith("ied") && lower.length > 4) {
    return `${lower.slice(0, -3)}y`;
  }

  // -ed
  if (lower.endsWith("ed") && lower.length > 4) {
    let stem = lower.slice(0, -2);
    if (
      stem.length > 2 &&
      stem[stem.length - 1] === stem[stem.length - 2] &&
      !"lsz".includes(stem[stem.length - 1])
    ) {
      stem = stem.slice(0, -1);
    }
    if (knownGlosses?.has(stem.toUpperCase())) return stem;
    if (knownGlosses?.has(`${stem}E`.toUpperCase())) return `${stem}e`;
    const withE = `${stem}e`;
    if (ASL_SYNONYMS[withE]) return withE;
    return stem;
  }

  // -es after ch/sh/s/x/z/o (watches -> watch, boxes -> box)
  if (lower.endsWith("es") && lower.length > 4 && /(ch|sh|s|x|z|o)es$/.test(lower)) {
    return lower.slice(0, -2);
  }

  // Regular plural / 3rd-person -s (apples -> apple, works -> work)
  if (lower.endsWith("s") && lower.length > 3 && !/(ss|us|is|as|os)$/.test(lower)) {
    return lower.slice(0, -1);
  }

  return lower;
}

const CANONICAL_NORMALIZATIONS = {
  hi: "HELLO",
  hey: "HELLO",
  bye: "GOODBYE",
  thank: "THANKYOU",
  thanks: "THANKYOU",
  thx: "THANKYOU",
  ty: "THANKYOU",
  pls: "PLEASE",
  plz: "PLEASE",
  ok: "GOOD",
  okay: "GOOD",
  mom: "MOTHER",
  mommy: "MOTHER",
  dad: "FATHER",
  daddy: "FATHER",
  yeah: "YES",
  yep: "YES",
  yup: "YES",
  nope: "NO",
  nah: "NO",
  gonna: "FUTURE",
  wanna: "WANT",
  gotta: "NEED",
};

function mapSingleWordToGloss(raw, lower, isProper, knownGlosses) {
  if (!lower) return { tokens: [], dropped: false, source: "dropped" };

  if (PURE_STOP_WORDS.has(lower) || AUX_DO.has(lower)) {
    return { tokens: [], dropped: true, source: "dropped" };
  }

  if (INTENSIFIER_PROFANITY.has(lower)) {
    return { tokens: ["REALLY"], dropped: false, source: "synonym" };
  }

  const numTokens = normalizeNumberToken(lower);
  if (numTokens) {
    return { tokens: numTokens, dropped: false, source: "number" };
  }

  if (lower === "tonight") {
    return { tokens: ["TODAY", "NIGHT"], dropped: false, source: "phrase" };
  }

  if (PRONOUN_MAP[lower]) {
    return { tokens: [PRONOUN_MAP[lower]], dropped: false, source: "direct" };
  }

  if (CANONICAL_NORMALIZATIONS[lower]) {
    return { tokens: [CANONICAL_NORMALIZATIONS[lower]], dropped: false, source: "synonym" };
  }

  const upperRaw = lower.toUpperCase();

  // Protect mid-sentence proper nouns (e.g., James, Paris, Maya) from stemming/synonym distortion
  if (isProper && !knownGlosses?.has(upperRaw) && !IRREGULAR_LEMMAS[lower] && !ASL_SYNONYMS[lower]) {
    return { tokens: [upperRaw], dropped: false, source: "direct" };
  }

  // Irregular lemma check before alias/knownGlosses so "going"/"went" always canonicalize to "GO"
  if (IRREGULAR_LEMMAS[lower]) {
    const lemma = IRREGULAR_LEMMAS[lower];
    const lemmaUpper = lemma.toUpperCase();
    if (knownGlosses?.has(lemmaUpper)) {
      return { tokens: [lemmaUpper], dropped: false, source: "lemma" };
    }
    if (ASL_SYNONYMS[lemma]) {
      return { tokens: [ASL_SYNONYMS[lemma]], dropped: false, source: "synonym" };
    }
    return { tokens: [lemmaUpper], dropped: false, source: "lemma" };
  }

  // Direct vocabulary hit BEFORE general synonym collapse (preserves richer Phase 2 vocabulary too)
  if (knownGlosses?.has(upperRaw)) {
    return { tokens: [upperRaw], dropped: false, source: "direct" };
  }

  // Synonym map before morphological stemming
  if (ASL_SYNONYMS[lower]) {
    return { tokens: [ASL_SYNONYMS[lower]], dropped: false, source: "synonym" };
  }

  // Safe morphological stemmer (skip proper nouns)
  if (!isProper) {
    const stemmed = stemMorphological(lower, knownGlosses);
    if (stemmed !== lower) {
      const stemUpper = stemmed.toUpperCase();
      if (knownGlosses?.has(stemUpper)) {
        return { tokens: [stemUpper], dropped: false, source: "lemma" };
      }
      if (ASL_SYNONYMS[stemmed]) {
        return { tokens: [ASL_SYNONYMS[stemmed]], dropped: false, source: "synonym" };
      }
      return { tokens: [stemUpper], dropped: false, source: "lemma" };
    }
  }

  return { tokens: [upperRaw], dropped: false, source: "direct" };
}

/**
 * Pure single-token / short-chunk PSE gloss helper (<0.15ms, zero compromise call).
 */
export function glossStreamingTokens(rawTokens, options = {}) {
  const knownGlosses = options.knownGlosses || null;
  const sliceOffset =
    typeof options.sliceOffset === "number"
      ? options.sliceOffset
      : options.isUtteranceStart === false
        ? 1
        : 0;
  const inputList = Array.isArray(rawTokens)
    ? rawTokens
    : typeof rawTokens === "string"
      ? tokenizeRaw(rawTokens)
      : [];

  // Pre-normalize & expand contractions
  const expanded = [];
  for (let i = 0; i < inputList.length; i++) {
    const entry = inputList[i];
    const isObj = entry !== null && typeof entry === "object";
    const raw = normalizeRawToken(isObj ? entry.raw : entry);
    if (!raw) continue;
    const expandedParts = expandContractionToken(raw);
    if (expandedParts !== null) {
      for (const p of expandedParts) {
        expanded.push({ raw: p.raw, lower: p.lower, isProper: p.isProper, origRaw: raw });
      }
    } else {
      const lower = raw.toLowerCase();
      const isProper =
        isObj && typeof entry.isProper === "boolean"
          ? entry.isProper
          : isCapitalizedProper(raw, i + sliceOffset);
      expanded.push({
        raw,
        lower,
        isProper,
        origRaw: raw,
      });
    }
  }

  const tokens = [];
  const dropped = [];
  const meta = [];

  for (let i = 0; i < expanded.length; i++) {
    const cur = expanded[i];
    const next = i + 1 < expanded.length ? expanded[i + 1] : null;

    // Check same-slice bigrams first
    if (next) {
      const pairKey = `${cur.lower} ${next.lower}`;
      if (SAME_SLICE_BIGRAMS.has(pairKey)) {
        const mapped = SAME_SLICE_BIGRAMS.get(pairKey);
        if (mapped.length === 0) {
          dropped.push(cur.raw, next.raw);
          meta.push({ raw: pairKey, gloss: null, source: "dropped" });
        } else {
          for (const g of mapped) {
            tokens.push(g);
            meta.push({ raw: pairKey, gloss: g, source: "phrase" });
          }
        }
        i++; // consume next
        continue;
      }
      // "so fucking/freaking/damn..." -> REALLY
      if (cur.lower === "so" && INTENSIFIER_PROFANITY.has(next.lower)) {
        tokens.push("REALLY");
        meta.push({ raw: pairKey, gloss: "REALLY", source: "phrase" });
        i++;
        continue;
      }
    }

    const res = mapSingleWordToGloss(cur.raw, cur.lower, cur.isProper, knownGlosses);
    if (res.dropped) {
      dropped.push(cur.raw);
      meta.push({ raw: cur.raw, gloss: null, source: "dropped" });
    } else {
      for (const g of res.tokens) {
        tokens.push(g);
        meta.push({ raw: cur.raw, gloss: g, source: res.source });
      }
    }
  }

  return { tokens, dropped, meta };
}

function lcpLength(a, b) {
  const minLen = Math.min(a.length, b.length);
  let i = 0;
  while (i < minLen && a[i] === b[i]) i++;
  return i;
}

/**
 * Stateful ultra-low-latency word-by-word streaming glosser for live STT.
 * Supports BOTH cumulative hypotheses (`ingestHypothesis`) and pre-sliced deltas (`ingestDelta`).
 */
export class StreamingRuleGloss {
  constructor(options = {}) {
    this.knownGlosses = options.knownGlosses || null;
    this.holdbackWords =
      typeof options.holdbackWords === "number" ? Math.max(0, options.holdbackWords) : 1;
    this.enablePhraseLookahead = options.enablePhraseLookahead !== false;
    this.maxRecentHistory = Math.max(4, options.maxRecentHistory || 12);
    this.reset();
  }

  setKnownGlosses(glossSet) {
    this.knownGlosses = glossSet instanceof Set ? glossSet : glossSet ? new Set(glossSet) : null;
  }

  reset() {
    this.utteranceId = null;
    this.committedNorms = [];
    this.prevHypNorms = [];
    this.prevHypTokens = [];
    this.recentCommittedNorms = [];
    this.suppressNextLower = null;
    this.pendingPrefix = null; // { raw, lower, isProper }
  }

  _recordRecent(norms) {
    for (const n of norms) {
      this.recentCommittedNorms.push(n);
    }
    if (this.recentCommittedNorms.length > this.maxRecentHistory) {
      this.recentCommittedNorms.splice(
        0,
        this.recentCommittedNorms.length - this.maxRecentHistory
      );
    }
  }

  _processCommittedRawWords(rawWords, flushTail = false) {
    const outTokens = [];
    const outDropped = [];
    const hasPriorWords = this.committedNorms.length > 0 || this.recentCommittedNorms.length > 0;

    // Prepend any held 4-word micro-prefix from previous step
    const workList = [];
    if (this.pendingPrefix) {
      workList.push(this.pendingPrefix);
      this.pendingPrefix = null;
    }

    for (let i = 0; i < rawWords.length; i++) {
      const raw = normalizeRawToken(rawWords[i]);
      if (!raw) continue;
      const lower = raw.toLowerCase();

      // Zero-wait split-bigram suppression (e.g. "thank" emitted THANKYOU; suppress next "you")
      if (this.suppressNextLower && workList.length === 0 && lower === this.suppressNextLower) {
        this.suppressNextLower = null;
        outDropped.push(raw);
        continue;
      }
      this.suppressNextLower = null;

      workList.push({
        raw,
        lower,
        isProper: isCapitalizedProper(
          raw,
          i > 0 || workList.length > 0 || hasPriorWords ? 1 : 0
        ),
      });
    }

    if (workList.length === 0) {
      if (flushTail) {
        this.suppressNextLower = null;
      }
      return { tokens: outTokens, dropped: outDropped };
    }

    // If last token is one of the 4 micro-prefixes and we're NOT flushing, hold it 1 step
    let heldPrefix = null;
    if (
      this.enablePhraseLookahead &&
      !flushTail &&
      workList.length > 0 &&
      MICRO_PREFIXES.has(workList[workList.length - 1].lower)
    ) {
      heldPrefix = workList.pop();
    }

    if (workList.length > 0) {
      // Check zero-wait split-bigram suppression trigger on the last token of workList
      const lastLower = workList[workList.length - 1].lower;
      const res = glossStreamingTokens(workList, {
        knownGlosses: this.knownGlosses,
        isUtteranceStart: !hasPriorWords,
      });
      outTokens.push(...res.tokens);
      outDropped.push(...res.dropped);

      if (!flushTail && !heldPrefix && ZERO_WAIT_SUPPRESS_NEXT[lastLower]) {
        // Only suppress next if it wasn't already consumed as a same-slice bigram
        const prevLower = workList.length >= 2 ? workList[workList.length - 2].lower : "";
        const lastPair = `${prevLower} ${lastLower}`;
        if (!SAME_SLICE_BIGRAMS.has(lastPair)) {
          this.suppressNextLower = ZERO_WAIT_SUPPRESS_NEXT[lastLower];
        }
      }
    }

    if (heldPrefix) {
      this.pendingPrefix = heldPrefix;
    }

    if (flushTail) {
      this.suppressNextLower = null;
    }

    return { tokens: outTokens, dropped: outDropped };
  }

  ingestHypothesis(text, meta = {}) {
    const t0 = performance.now();
    const isFinal = Boolean(meta.isFinal);
    const uttId = meta.utteranceId !== undefined ? meta.utteranceId : this.utteranceId;

    const emittedTokens = [];
    const emittedDropped = [];
    let retracted = false;

    // Handle utterance boundary change: flush any held-back tail & micro-prefix from prior utterance
    if (uttId !== this.utteranceId) {
      const uncommittedTail =
        this.prevHypTokens.length > this.committedNorms.length
          ? this.prevHypTokens.slice(this.committedNorms.length)
          : [];
      if (uncommittedTail.length > 0 || this.pendingPrefix) {
        const flushed = this._processCommittedRawWords(uncommittedTail, true);
        emittedTokens.push(...flushed.tokens);
        emittedDropped.push(...flushed.dropped);
      }
      this.utteranceId = uttId;
      this.committedNorms = [];
      this.prevHypNorms = [];
      this.prevHypTokens = [];
      this.recentCommittedNorms = [];
      this.suppressNextLower = null;
    }

    const hypTokens = tokenizeRaw(text);
    const hypNorms = hypTokens.map((w) => w.toLowerCase());

    if (hypTokens.length === 0) {
      if (isFinal && this.pendingPrefix) {
        const flushed = this._processCommittedRawWords([], true);
        emittedTokens.push(...flushed.tokens);
        emittedDropped.push(...flushed.dropped);
      }
      if (isFinal) {
        this.suppressNextLower = null;
      }
      const glossDoneTsMs = performance.now();
      return {
        tokens: emittedTokens,
        dropped: emittedDropped,
        committedWords: [],
        pendingPreview: [],
        glossMs: glossDoneTsMs - t0,
        retracted: false,
        trace: {
          speechTsMs: meta.speechTsMs ?? meta.speechTimestampMs ?? null,
          sttEmitTsMs: meta.sttEmitTsMs ?? meta.emitTimestampMs ?? null,
          glossDoneTsMs,
        },
      };
    }

    const agreeLen = lcpLength(this.prevHypNorms, hypNorms);
    const committedCount = this.committedNorms.length;

    let targetCommitLen;
    if (isFinal) {
      targetCommitLen = hypNorms.length;
    } else {
      targetCommitLen = Math.max(agreeLen, Math.max(0, hypNorms.length - this.holdbackWords));
    }

    const commonLen = lcpLength(this.committedNorms, hypNorms);
    const prefixMatches = commonLen === committedCount;

    // Pure transient shrink (hypNorms is a strict prefix of committedNorms): do not duplicate
    if (hypNorms.length < committedCount && commonLen === hypNorms.length) {
      retracted = true;
      this.prevHypNorms = hypNorms;
      this.prevHypTokens = hypTokens;
      if (isFinal && this.pendingPrefix) {
        const flushed = this._processCommittedRawWords([], true);
        emittedTokens.push(...flushed.tokens);
        emittedDropped.push(...flushed.dropped);
      }
      if (isFinal) {
        this.suppressNextLower = null;
      }
      const glossDoneTsMs = performance.now();
      return {
        tokens: emittedTokens,
        dropped: emittedDropped,
        committedWords: [],
        pendingPreview: [],
        glossMs: glossDoneTsMs - t0,
        retracted,
        trace: {
          speechTsMs: meta.speechTsMs ?? meta.speechTimestampMs ?? null,
          sttEmitTsMs: meta.sttEmitTsMs ?? meta.emitTimestampMs ?? null,
          glossDoneTsMs,
        },
      };
    }

    let startIdx = committedCount;
    let anchored = -1;
    let diverged = false;

    if (!prefixMatches && committedCount > 0) {
      retracted = true;
      // Attempt tail-anchor alignment around committedCount
      const tail2 = this.committedNorms.slice(-2);
      const tail1 = this.committedNorms.slice(-1);
      const searchStart = Math.max(0, Math.min(committedCount - 2, hypNorms.length - 2));
      const searchEnd = Math.min(hypNorms.length, committedCount + 2);

      if (tail2.length === 2) {
        for (let pos = searchStart; pos + 2 <= searchEnd; pos++) {
          if (hypNorms[pos] === tail2[0] && hypNorms[pos + 1] === tail2[1]) {
            anchored = pos + 2;
            break;
          }
        }
      }
      if (anchored === -1 && tail1.length === 1) {
        for (let pos = searchStart; pos + 1 <= searchEnd; pos++) {
          if (hypNorms[pos] === tail1[0]) {
            anchored = pos + 1;
            break;
          }
        }
      }

      if (anchored !== -1) {
        startIdx = anchored;
      } else if (
        commonLen < hypNorms.length &&
        hypNorms[commonLen] !== this.committedNorms[commonLen]
      ) {
        // Genuine divergence after shared prefix with no tail anchor match
        diverged = true;
        startIdx = commonLen;
      } else {
        startIdx = committedCount;
      }
    }

    const newRawSlice =
      targetCommitLen > startIdx ? hypTokens.slice(startIdx, targetCommitLen) : [];
    const newNormSlice =
      targetCommitLen > startIdx ? hypNorms.slice(startIdx, targetCommitLen) : [];

    if (newRawSlice.length > 0 || (isFinal && this.pendingPrefix)) {
      const processed = this._processCommittedRawWords(newRawSlice, isFinal);
      emittedTokens.push(...processed.tokens);
      emittedDropped.push(...processed.dropped);
    }

    let nextCommittedCount;
    if (anchored !== -1) {
      nextCommittedCount = startIdx + newRawSlice.length;
      this.committedNorms = hypNorms.slice(0, nextCommittedCount);
    } else if (diverged) {
      if (newRawSlice.length > 0) {
        nextCommittedCount = startIdx + newRawSlice.length;
        this.committedNorms = hypNorms.slice(0, nextCommittedCount);
      } else {
        // Divergent tail is still unconfirmed/held back: preserve committedNorms until confirmed
        nextCommittedCount = commonLen;
      }
    } else {
      nextCommittedCount = Math.max(committedCount, startIdx + newRawSlice.length);
      this.committedNorms = hypNorms.slice(0, nextCommittedCount);
    }

    if (isFinal) {
      this.suppressNextLower = null;
    }

    this._recordRecent(newNormSlice);
    this.prevHypNorms = hypNorms;
    this.prevHypTokens = hypTokens;

    const pendingPreview = isFinal ? [] : hypTokens.slice(nextCommittedCount);
    const glossDoneTsMs = performance.now();

    return {
      tokens: emittedTokens,
      dropped: emittedDropped,
      committedWords: newRawSlice,
      pendingPreview,
      glossMs: glossDoneTsMs - t0,
      retracted,
      trace: {
        speechTsMs: meta.speechTsMs ?? meta.speechTimestampMs ?? null,
        sttEmitTsMs: meta.sttEmitTsMs ?? meta.emitTimestampMs ?? null,
        glossDoneTsMs,
      },
    };
  }

  ingestDelta(deltaText, meta = {}) {
    const t0 = performance.now();
    const isFinal = Boolean(meta.isFinal);
    const emittedTokens = [];
    const emittedDropped = [];

    // Utterance boundary change: flush any held micro-prefix and reset overlap/suppression state
    if (meta.utteranceId !== undefined && meta.utteranceId !== this.utteranceId) {
      if (this.pendingPrefix) {
        const flushed = this._processCommittedRawWords([], true);
        emittedTokens.push(...flushed.tokens);
        emittedDropped.push(...flushed.dropped);
      }
      this.utteranceId = meta.utteranceId;
      this.recentCommittedNorms = [];
      this.suppressNextLower = null;
    }

    const rawTokens = tokenizeRaw(deltaText);
    const chunkNorms = rawTokens.map((w) => w.toLowerCase());

    // Strip 1..3-word overlap with recently committed words within the same utterance
    let overlap = 0;
    const maxK = Math.min(3, this.recentCommittedNorms.length, chunkNorms.length);
    for (let k = maxK; k >= 1; k--) {
      let match = true;
      for (let j = 0; j < k; j++) {
        if (
          this.recentCommittedNorms[this.recentCommittedNorms.length - k + j] !== chunkNorms[j]
        ) {
          match = false;
          break;
        }
      }
      if (match) {
        overlap = k;
        break;
      }
    }

    const dedupedRaw = rawTokens.slice(overlap);
    const dedupedNorms = chunkNorms.slice(overlap);

    if (dedupedRaw.length > 0 || (isFinal && this.pendingPrefix)) {
      const processed = this._processCommittedRawWords(dedupedRaw, isFinal);
      emittedTokens.push(...processed.tokens);
      emittedDropped.push(...processed.dropped);
    }

    if (dedupedNorms.length > 0) {
      this._recordRecent(dedupedNorms);
    }

    if (isFinal) {
      this.recentCommittedNorms = [];
      this.suppressNextLower = null;
    }

    const glossDoneTsMs = performance.now();
    return {
      tokens: emittedTokens,
      dropped: emittedDropped,
      committedWords: dedupedRaw,
      pendingPreview: [],
      glossMs: glossDoneTsMs - t0,
      retracted: false,
      trace: {
        speechTsMs: meta.speechTsMs ?? meta.speechTimestampMs ?? null,
        sttEmitTsMs: meta.sttEmitTsMs ?? meta.emitTimestampMs ?? null,
        glossDoneTsMs,
      },
    };
  }

  flush(meta = {}) {
    const t0 = performance.now();
    const uncommittedTail =
      this.prevHypTokens.length > this.committedNorms.length
        ? this.prevHypTokens.slice(this.committedNorms.length)
        : [];
    const uncommittedNorms = uncommittedTail.map((w) => w.toLowerCase());

    let tokens = [];
    let dropped = [];
    if (uncommittedTail.length > 0 || this.pendingPrefix) {
      const processed = this._processCommittedRawWords(uncommittedTail, true);
      tokens = processed.tokens;
      dropped = processed.dropped;
    }

    if (uncommittedNorms.length > 0) {
      this.committedNorms = [...this.committedNorms, ...uncommittedNorms];
    }
    this.recentCommittedNorms = [];
    this.suppressNextLower = null;

    const glossDoneTsMs = performance.now();
    return {
      tokens,
      dropped,
      committedWords: uncommittedTail,
      pendingPreview: [],
      glossMs: glossDoneTsMs - t0,
      retracted: false,
      trace: {
        speechTsMs: meta.speechTsMs ?? meta.speechTimestampMs ?? null,
        sttEmitTsMs: meta.sttEmitTsMs ?? meta.emitTimestampMs ?? null,
        glossDoneTsMs,
      },
    };
  }
}

export function createStreamingGloss(options = {}) {
  return new StreamingRuleGloss(options);
}

// ---- Existing Batch Time-Topic-Comment Engine (100% backward compatible + contraction fix) ----

function translateClause(clauseText, knownGlosses = null) {
  const doc = nlp(clauseText);
  doc.contractions().expand();

  const hasPastVerb = doc.has("#PastTense");
  const hasFutureModal = /\b(will|shall|FUTURE)\b/i.test(clauseText);

  doc.match("#Verb").not("#Modal").not("#Auxiliary").verbs().toInfinitive();
  doc.nouns().toSingular();

  const sentences = doc.json();
  const terms = [];
  for (const s of sentences) {
    for (const t of s.terms || []) {
      terms.push(t);
    }
  }

  // Check expanded terms for main verb or negation (fixes "don't"/"didn't" contraction bug)
  const hasOtherMainVerb = terms.some((t) => {
    const w = (t.text || "").toLowerCase().replace(/[^a-z0-9_']/g, "");
    const tags = new Set(t.tags || []);
    return tags.has("Verb") && !tags.has("Copula") && !AUX_DO.has(w) && !PURE_STOP_WORDS.has(w);
  });
  const hasExpandedNegation = terms.some((t) => {
    const w = (t.text || "").toLowerCase().replace(/[^a-z0-9_']/g, "");
    return w === "not" || w === "never";
  });

  const timeSlot = [];
  const topicSlot = [];
  const commentSlot = [];
  const negSlot = [];
  const whSlot = [];
  const dropped = [];

  let seenVerb = false;
  let seenSubject = false;

  for (let i = 0; i < terms.length; i++) {
    const term = terms[i];
    const rawText = (term.text || "").trim();
    if (!rawText) continue;

    let normalized = rawText.toLowerCase().replace(/[^a-z0-9_']/g, "");
    if (!normalized) continue;

    if (PURE_STOP_WORDS.has(normalized)) {
      dropped.push(rawText);
      continue;
    }

    // Drop auxiliary do/does/did ONLY when another verb or negation is present
    if (AUX_DO.has(normalized) && (hasOtherMainVerb || hasExpandedNegation)) {
      dropped.push(rawText);
      continue;
    }

    if (INTENSIFIER_PROFANITY.has(normalized)) {
      commentSlot.push("REALLY");
      continue;
    }

    const numTokens = normalizeNumberToken(normalized);
    if (numTokens) {
      commentSlot.push(...numTokens);
      continue;
    }

    const tags = new Set(term.tags || []);

    if (tags.has("Gerund") && normalized.endsWith("ing") && normalized.length > 4) {
      normalized = normalized.slice(0, -3);
      const last = normalized[normalized.length - 1];
      if (normalized.length > 3 && normalized.endsWith(last + last)) {
        normalized = normalized.slice(0, -1);
      }
    }

    const upperCandidate = normalized.toUpperCase();
    const gloss =
      PRONOUN_MAP[normalized] ||
      (knownGlosses?.has(upperCandidate) ? upperCandidate : null) ||
      ASL_SYNONYMS[normalized] ||
      upperCandidate;

    if (WH_WORDS.has(normalized)) {
      whSlot.push(gloss);
      continue;
    }

    if (NEG_WORDS.has(normalized)) {
      negSlot.push(normalized === "no" ? "NO" : "NOT");
      continue;
    }

    if (TIME_WORDS.has(normalized) || tags.has("Date")) {
      if (!timeSlot.includes(gloss)) timeSlot.push(gloss);
      continue;
    }

    if (POSSESSIVES.has(normalized) && i + 1 < terms.length) {
      if (seenVerb) topicSlot.push(gloss);
      else commentSlot.push(gloss);
      continue;
    }

    const isVerb = tags.has("Verb") && !tags.has("Copula");
    const isNounOrProper = (tags.has("Noun") || tags.has("ProperNoun")) && !tags.has("Pronoun");

    if (seenSubject && seenVerb && isNounOrProper) {
      topicSlot.push(gloss);
    } else {
      commentSlot.push(gloss);
      if (tags.has("Pronoun") || isNounOrProper) seenSubject = true;
      if (isVerb) seenVerb = true;
    }
  }

  if (hasPastVerb && timeSlot.length === 0) {
    timeSlot.unshift("PAST");
  } else if (hasFutureModal && timeSlot.length === 0) {
    timeSlot.unshift("FUTURE");
  }

  return {
    tokens: [...timeSlot, ...topicSlot, ...commentSlot, ...negSlot, ...whSlot],
    time: timeSlot,
    topic: topicSlot,
    comment: commentSlot,
    neg: negSlot,
    wh: whSlot,
    dropped,
  };
}

/**
 * Deterministic English -> ASL Gloss converter.
 * - Default mode ("ttc"): Time -> Topic -> Comment -> Negation -> WH batch conversion.
 * - Streaming mode ({ mode: "stream" | "pse" }): Ultra-low-latency (<0.15ms) word-by-word PSE.
 */
export function translateRuleGloss(inputText, options = {}) {
  const mode = options.mode === "stream" || options.mode === "pse" ? "stream" : "ttc";
  const knownGlosses = options.knownGlosses || null;
  const trimmed = (inputText || "").trim();

  if (!trimmed) {
    return {
      tokens: [],
      glossString: "",
      engine: "rule",
      mode,
      breakdown: { time: [], topic: [], comment: [], neg: [], wh: [], dropped: [] },
    };
  }

  if (mode === "stream") {
    const res = glossStreamingTokens(tokenizeRaw(trimmed), { knownGlosses });
    return {
      tokens: res.tokens,
      glossString: res.tokens.join(" "),
      engine: "rule",
      mode: "stream",
      breakdown: {
        time: [],
        topic: [],
        comment: [...res.tokens],
        neg: [],
        wh: [],
        dropped: res.dropped,
      },
    };
  }

  let preprocessed = trimmed;
  for (const [pattern, replacement] of PHRASE_REWRITES) {
    preprocessed = preprocessed.replace(pattern, replacement);
  }

  const clauses = preprocessed
    .split(/[,.;!?]+/)
    .map((c) => c.trim())
    .filter(Boolean);

  const tokens = [];
  const breakdown = { time: [], topic: [], comment: [], neg: [], wh: [], dropped: [] };

  for (const clause of clauses) {
    const res = translateClause(clause, knownGlosses);
    tokens.push(...res.tokens);
    breakdown.time.push(...res.time);
    breakdown.topic.push(...res.topic);
    breakdown.comment.push(...res.comment);
    breakdown.neg.push(...res.neg);
    breakdown.wh.push(...res.wh);
    breakdown.dropped.push(...res.dropped);
  }

  return {
    tokens,
    glossString: tokens.join(" "),
    engine: "rule",
    mode: "ttc",
    breakdown,
  };
}
