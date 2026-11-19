import * as webllm from "@mlc-ai/web-llm";
import { translateRuleGloss } from "./ruleGloss.js";

export const WEBLLM_MODELS = [
  {
    id: "SmolLM2-360M-Instruct-q4f16_1-MLC",
    label: "SmolLM2 360M (Fast, ~250 MB)",
  },
  {
    id: "Llama-3.2-1B-Instruct-q4f16_1-MLC",
    label: "Llama 3.2 1B (Accurate, ~700 MB)",
  },
];

const SYSTEM_PROMPT = `You convert spoken/written English into American Sign Language (ASL) Gloss.
Rules:
1. Output ONLY uppercase gloss words separated by single spaces. No punctuation, no explanations.
2. Follow ASL Time-Topic-Comment grammar:
   - TIME words first (TOMORROW, YESTERDAY, NOW, PAST, FUTURE).
   - TOPIC / OBJECT second.
   - COMMENT (SUBJECT + ROOT VERB) third.
   - NEGATION (NOT, NEVER) and WH-WORDS (WHO, WHAT, WHERE, WHEN, WHY, HOW) at the end.
3. Drop English articles (a, an, the) and linking verbs (am, is, are, was, were, do, does, did).
4. Convert verbs to base root form (went -> GO, buying -> BUY).

Examples:
English: "I am going to the library tomorrow." -> TOMORROW LIBRARY I GO
English: "Where did you buy that coffee yesterday?" -> YESTERDAY COFFEE YOU BUY WHERE
English: "I do not like loud music." -> LOUD MUSIC I LIKE NOT`;

export async function checkWebGPUSupport() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) {
    return { supported: false, reason: "WebGPU is not exposed by this browser." };
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      return { supported: false, reason: "No compatible WebGPU GPU adapter found." };
    }
    return { supported: true, reason: "WebGPU ready" };
  } catch (err) {
    return { supported: false, reason: err?.message || "WebGPU check failed." };
  }
}

export class WebLLMGlossEngine {
  constructor() {
    this.engine = null;
    this.loadedModelId = null;
    this.loading = false;
  }

  get isReady() {
    return Boolean(this.engine) && !this.loading;
  }

  async load(modelId, onProgress) {
    if (this.loading) return;
    if (this.engine && this.loadedModelId === modelId) return;

    this.loading = true;
    try {
      this.engine = await webllm.CreateMLCEngine(modelId, {
        initProgressCallback: (report) => {
          if (onProgress) onProgress(report);
        },
      });
      this.loadedModelId = modelId;
    } finally {
      this.loading = false;
    }
  }

  async translate(englishText) {
    const trimmed = (englishText || "").trim();
    if (!trimmed) {
      return translateRuleGloss("");
    }
    if (!this.engine) {
      return translateRuleGloss(trimmed);
    }

    const reply = await this.engine.chat.completions.create({
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: trimmed },
      ],
      temperature: 0.0,
      max_tokens: 48,
    });

    const raw = reply.choices?.[0]?.message?.content?.trim() || "";
    const tokens = raw
      .toUpperCase()
      .replace(/[^A-Z0-9_\s]/g, " ")
      .split(/\s+/)
      .filter(Boolean);

    if (tokens.length === 0) {
      return translateRuleGloss(trimmed);
    }

    const ruleRef = translateRuleGloss(trimmed);
    return {
      tokens,
      glossString: tokens.join(" "),
      engine: "webllm",
      modelId: this.loadedModelId,
      breakdown: ruleRef.breakdown,
    };
  }
}
