import nlp from "compromise";

const STOP_WORDS = new Set([
  // Articles & determiners
  "a", "an", "the", "that", "this", "these", "those",
  // Copulas & auxiliaries
  "am", "is", "are", "was", "were", "be", "been", "being",
  "do", "does", "did", "have", "has", "had",
  "will", "would", "shall", "should", "can", "could", "may", "might", "must",
  // Prepositions that drop in ASL spatial grammar
  "to", "of", "at", "by", "for", "from", "with", "into", "onto", "on", "in", "about", "as",
  // Conversational fillers & discourse markers
  "uh", "um", "er", "ah", "oh", "ugh", "like", "just", "well", "yeah", "yep",
  "actually", "basically", "literally", "honestly", "anyway", "so",
  // Intensifier profanity (expressed non-manually / via REALLY in ASL, never fingerspelled)
  "fucking", "fuckin", "freaking", "friggin", "damn", "hella", "totally", "super", "extremely", "very"
]);

// Words dropped only when used as sentence/clause-initial discourse vocatives ("man i went...", "bro...")
const CLAUSE_INITIAL_FILLERS = new Set(["man", "bro", "dude", "yo", "guys", "listen", "look"]);

const INTENSIFIERS = new Set([
  "so", "very", "really", "super", "extremely", "fucking", "fuckin", "freaking", "hella", "totally"
]);

const TIME_WORDS = new Set([
  "today", "tomorrow", "yesterday", "now", "later", "soon", "before", "after",
  "always", "sometimes", "morning", "afternoon", "evening", "night", "tonight",
  "week", "month", "year", "weekend", "past", "future",
  "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
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

// ASL maps many English synonyms to a single core sign (prevents unnecessary fingerspelling)
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
  fine: "GOOD",
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
};

// Multi-word English phrases normalized before clause parsing
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
  // Strip intensifier profanity before compromise lemmatizes "fucking" -> "fuck"
  [/\b(?:so\s+)?(?:fucking|fuckin|freaking|friggin|damn|hella)\b/gi, "REALLY"],
  // Dummy pronoun "it is / it was / it's" -> drop "it"
  [/\bit(?:'s|\s+is|\s+was|\s+were)\b/gi, ""],
];

function translateClause(clauseText) {
  const doc = nlp(clauseText);
  doc.contractions().expand();

  const hasPastVerb = doc.has("#PastTense");
  const hasFutureModal = /\b(will|shall|FUTURE)\b/i.test(clauseText);

  doc.verbs().toInfinitive();
  doc.nouns().toSingular();

  const sentences = doc.json();
  const terms = [];
  for (const s of sentences) {
    for (const t of s.terms || []) {
      terms.push(t);
    }
  }

  const timeSlot = [];
  const topicSlot = [];
  const commentSlot = [];
  const negSlot = [];
  const whSlot = [];
  const dropped = [];

  let seenVerb = false;
  let seenSubject = false;
  let hasIntensifier = false;
  let isClauseStart = true;

  for (let i = 0; i < terms.length; i++) {
    const term = terms[i];
    const rawText = (term.text || "").trim();
    if (!rawText) continue;

    let normalized = rawText.toLowerCase().replace(/[^a-z0-9_']/g, "");
    if (!normalized) continue;

    // Strip clause-initial vocative fillers ("man", "bro", "dude")
    if (isClauseStart && CLAUSE_INITIAL_FILLERS.has(normalized)) {
      dropped.push(rawText);
      continue;
    }
    isClauseStart = false;

    // Collapse intensifier chains ("so fucking tough" -> intensifier flag -> REALLY)
    if (INTENSIFIERS.has(normalized)) {
      hasIntensifier = true;
      dropped.push(rawText);
      continue;
    }

    if (STOP_WORDS.has(normalized) || normalized === "it") {
      dropped.push(rawText);
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

  // Express collapsed intensifiers cleanly at the end of the comment (ASL non-manual + REALLY)
  if (hasIntensifier && commentSlot.length > 0 && !commentSlot.includes("REALLY")) {
    commentSlot.push("REALLY");
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
 * Follows Time -> Topic (Object/Location) -> Comment (Subject + Verb) -> Negation -> WH order.
 * Collapses conversational fillers, multi-word time phrases, and English synonyms to core ASL glosses.
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
