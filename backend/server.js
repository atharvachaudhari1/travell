// ══════════════════════════════════════════════════════════════
// ARKA AI Proxy — Dual-Model Backend
// Gemini primary → Groq fallback chain (3 models) on 429/503
// Keys live ONLY here in env vars. Browser never sees them.
// ══════════════════════════════════════════════════════════════
import dotenv from "dotenv";
dotenv.config();

import express from "express";
import cors from "cors";

const app = express();
app.use(express.json({ limit: "1mb" }));

app.use(
  cors({
    origin: process.env.ALLOWED_ORIGIN || "*",
    methods: ["GET", "POST", "OPTIONS"],
  })
);

// ── Constants ──────────────────────────────────────────────────
const GEMINI_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent`;
const GROQ_URL   = `https://api.groq.com/openai/v1/chat/completions`;
const sleep      = (ms) => new Promise(r => setTimeout(r, ms));

// Groq fallback chain — lighter models first to avoid TPM limits.
// Each model has a lower token budget to stay within free-tier limits.
const GROQ_CHAIN = [
  { model: "openai/gpt-oss-20b",  maxTokens: 2048 }, // lightest, fastest
  { model: "qwen/qwen3.6-27b",    maxTokens: 2048 }, // multilingual
  { model: "groq/compound",       maxTokens: 2048 }, // most capable
];

// ── Gemini ─────────────────────────────────────────────────────
async function callGemini(prompt, isJson) {
  const key = process.env.GEMINI_API_KEY;
  if (!key) throw Object.assign(new Error("Missing GEMINI_API_KEY"), { status: 500 });

  const r = await fetch(`${GEMINI_URL}?key=${encodeURIComponent(key)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: {
        temperature: 0.7,
        maxOutputTokens: 8192,
        responseMimeType: isJson ? "application/json" : "text/plain",
      },
    }),
  });

  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    throw Object.assign(
      new Error(e?.error?.message || `Gemini error ${r.status}`),
      { status: r.status }
    );
  }

  const data = await r.json();
  const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!text) throw Object.assign(new Error("Empty Gemini response"), { status: 502 });
  return { text, model: "gemini-flash-latest" };
}

// ── Clean Groq reasoning artifacts ────────────────────────────
// Reasoning models (qwen, compound) prepend <think>...</think> or
// **Reasoning** / **Answer** blocks. Strip all of it so the
// frontend's parseJSON() gets clean text or valid JSON.
function cleanGroqResponse(raw, isJson) {
  let t = raw;

  // 1. Strip <think>...</think> reasoning blocks (qwen / deepseek style)
  t = t.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();

  // 2. Strip **Reasoning** ... **Answer** section pattern (groq/compound)
  const reasoningIdx = t.search(/\*\*Reasoning\*\*/i);
  const answerIdx    = t.search(/\*\*Answer\*\*/i);
  if (reasoningIdx !== -1 && answerIdx !== -1 && answerIdx > reasoningIdx) {
    t = t.slice(answerIdx + "**Answer**".length).trim();
  }

  // 3. Strip any remaining leading **Bold Header**
  t = t.replace(/^\*\*[A-Za-z ]+\*\*\s*/m, "").trim();

  // 4. Strip markdown code fences (```json ... ``` or ``` ... ```)
  t = t.replace(/^```(?:json)?\r?\n?/i, "").replace(/\r?\n?```\s*$/, "").trim();

  // 5. For JSON mode: extract first { } or [ ] block if extra text remains
  if (isJson) {
    const objMatch = t.match(/\{[\s\S]*\}/);
    if (objMatch) {
      try { JSON.parse(objMatch[0]); return objMatch[0]; } catch {}
    }
    const arrMatch = t.match(/\[[\s\S]*\]/);
    if (arrMatch) {
      try { JSON.parse(arrMatch[0]); return arrMatch[0]; } catch {}
    }
  }

  return t;
}

// ── Single Groq model attempt ─────────────────────────────────
async function callGroqModel(prompt, isJson, model, maxTokens) {
  const key = process.env.GROQ_API_KEY;
  if (!key) throw Object.assign(new Error("Missing GROQ_API_KEY"), { status: 500 });

  const systemMsg = isJson
    ? "You are a travel AI assistant. Output ONLY a raw JSON object or array. No <think> tags, no markdown fences, no explanation, no headers — start your response directly with { or [."
    : "You are a helpful travel AI assistant. Answer directly and concisely. Do NOT use <think> tags or reasoning blocks.";

  const r = await fetch(GROQ_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${key}`,
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: "system", content: systemMsg },
        { role: "user",   content: prompt },
      ],
      temperature: 0.7,
      max_tokens: maxTokens,
    }),
  });

  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    throw Object.assign(
      new Error(e?.error?.message || `Groq ${model} error ${r.status}`),
      { status: r.status }
    );
  }

  const data = await r.json();
  const raw  = data?.choices?.[0]?.message?.content;
  if (!raw) throw Object.assign(new Error(`Empty response from ${model}`), { status: 502 });

  const text = cleanGroqResponse(raw, isJson);
  console.log(`   ✅ ${model} responded (${raw.length} → ${text.length} chars after cleaning)`);
  return { text, model };
}

// ── Groq fallback chain — tries all models, respects TPM waits ─
async function callGroqChain(prompt, isJson) {
  if (!process.env.GROQ_API_KEY) {
    throw Object.assign(new Error("No GROQ_API_KEY configured"), { status: 500 });
  }

  for (const { model, maxTokens } of GROQ_CHAIN) {
    try {
      console.log(`   ⚡ Groq trying: ${model}`);
      return await callGroqModel(prompt, isJson, model, maxTokens);
    } catch (err) {
      if (err.status === 429) {
        // Parse the retry wait from Groq's error message ("try again in 7.66s")
        const m = err.message?.match(/try again in ([\d.]+)s/i);
        const waitMs = m ? Math.ceil(parseFloat(m[1]) * 1000) + 1000 : 9000;
        console.warn(`   ⏳ ${model} TPM limited. Waiting ${waitMs}ms then trying next model...`);
        await sleep(waitMs);
        continue; // move to next model in chain
      }
      // Any other error — log and try next model
      console.warn(`   ⚠️  ${model} failed (${err.status || "?"}): ${err.message}`);
    }
  }

  // All Groq models exhausted — return a graceful message instead of an error
  console.error("All Groq models rate-limited. Returning graceful fallback.");
  return {
    text: "AI is currently experiencing high demand. Please wait a moment and try again.",
    model: "fallback-message",
  };
}

// ── Smart AI router ────────────────────────────────────────────
// Gemini first. On 429/503/529 → Groq chain. Never propagates
// a rate-limit error to the browser.
const RATE_LIMIT_STATUSES = new Set([429, 503, 529]);

async function callAI(prompt, isJson) {
  try {
    return await callGemini(prompt, isJson);
  } catch (err) {
    if (RATE_LIMIT_STATUSES.has(err.status) && process.env.GROQ_API_KEY) {
      console.warn(`⚡ Gemini ${err.status} — switching to Groq chain`);
      return await callGroqChain(prompt, isJson);
    }
    throw err; // non-rate-limit error: propagate
  }
}

// ── Routes ─────────────────────────────────────────────────────

// Health check
app.get("/", (_req, res) => res.json({
  ok: true,
  gemini: !!process.env.GEMINI_API_KEY,
  groq:   !!process.env.GROQ_API_KEY,
  groq_chain: GROQ_CHAIN.map(m => m.model),
}));

// Debug status
app.get("/status", (_req, res) => res.json({
  gemini_key:  process.env.GEMINI_API_KEY ? "set" : "missing",
  groq_key:    process.env.GROQ_API_KEY   ? "set" : "missing",
  groq_models: GROQ_CHAIN.map(m => m.model),
  fallback_on: [...RATE_LIMIT_STATUSES],
}));

// Main AI route
app.post("/api/ai", async (req, res) => {
  const { prompt, isJson } = req.body || {};
  if (typeof prompt !== "string" || prompt.length === 0 || prompt.length > 20000) {
    return res.status(400).json({ error: "A valid 'prompt' string is required." });
  }
  try {
    const { text, model } = await callAI(prompt, isJson);
    return res.json({ text, model });
  } catch (err) {
    const status = err.status || 502;
    console.error(`/api/ai failed [${status}]:`, err.message);
    return res.status(status).json({ error: err.message || "AI request failed." });
  }
});

// Alias route (/gemini)
app.post("/gemini", async (req, res) => {
  const { prompt, isJson } = req.body || {};
  if (typeof prompt !== "string" || prompt.length === 0 || prompt.length > 20000) {
    return res.status(400).json({ error: "A valid 'prompt' string is required." });
  }
  try {
    const { text, model } = await callAI(prompt, isJson);
    return res.json({ text, model });
  } catch (err) {
    const status = err.status || 502;
    console.error(`/gemini failed [${status}]:`, err.message);
    return res.status(status).json({ error: err.message || "AI request failed." });
  }
});

// ── Start ──────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`✅ AI proxy listening on :${PORT}`);
  console.log(`   Primary:  Gemini (gemini-flash-latest)`);
  console.log(`   Fallback: Groq chain → ${GROQ_CHAIN.map(m => m.model).join(" → ")}`);
  console.log(`   Triggers: HTTP ${[...RATE_LIMIT_STATUSES].join(", ")} from Gemini`);
});
