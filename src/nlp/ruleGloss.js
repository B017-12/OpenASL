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
  "uh", "um", "er", "ah", "hmm", "uhh", "umm",
]);

// Auxiliaries dropped ONLY when followed by a verb/negation (e.g. "do not", "did you buy"),
// NEVER when used as the main verb ("What do you do?" -> keep DO).
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
  glad: "HAPPY",
  excited: "HAPPY",
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
  thanks: "THANKYOU",
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
  gonna: "FUTURE",
  wanna: "WANT",
  gotta: "NEED",
};

// Multi-word English idioms & time expressions normalized before clause parsing
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

function translateClause(clauseText) {
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

  // Check if clause has a non-auxiliary main verb so we know whether "do/did" is auxiliary or main verb
  const hasOtherMainVerb = terms.some((t) => {
    const w = (t.text || "").toLowerCase().replace(/[^a-z0-9_']/g, "");
    const tags = new Set(t.tags || []);
    return tags.has("Verb") && !tags.has("Copula") && !AUX_DO.has(w) && !PURE_STOP_WORDS.has(w);
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
    if (AUX_DO.has(normalized) && (hasOtherMainVerb || /\bnot\b/i.test(clauseText))) {
      dropped.push(rawText);
      continue;
    }

    if (INTENSIFIER_PROFANITY.has(normalized)) {
      commentSlot.push("REALLY");
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

    const gloss =
      PRONOUN_MAP[normalized] ||
      ASL_SYNONYMS[normalized] ||
      normalized.toUpperCase();

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
 * Deterministic English -> ASL Gloss converter (~2ms in-browser).
 * Preserves all meaningful content words while converting English grammar into
 * ASL Time -> Topic -> Comment -> Negation -> WH order.
 */
export function translateRuleGloss(inputText) {
  const trimmed = (inputText || "").trim();
  if (!trimmed) {
    return {
      tokens: [],
      glossString: "",
      engine: "rule",
      breakdown: { time: [], topic: [], comment: [], neg: [], wh: [], dropped: [] },
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
    const res = translateClause(clause);
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
    breakdown,
  };
}
