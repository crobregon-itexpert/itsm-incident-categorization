// Evaluación automatizada del agente (paper, Sección VI): usuario simulado + juez LLM.
//
// POST /api/eval/case  {case_id, oos, title, resolution, category_id, unseen}
//   - Solo activo si EVAL_ALLOWED_EMAILS está definido. Requiere un ID token de Google cuyo
//     email esté en esa lista y cuyo aud sea EVAL_AUDIENCE (por defecto "itsm-agent-eval").
//   - Usuario simulado (Gemini, EVAL_USER_MODEL): convierte el título del ticket en un reporte en
//     lenguaje natural y responde las preguntas del agente con datos plausibles. No conoce la solución.
//   - Juez (EVAL_JUDGE_MODEL, por defecto gemini-2.5-pro, distinto del agente): compara los pasos
//     propuestos con la solución real y con los casos recuperados. El usuario simulado responde
//     "sí" a "¿se resolvió?" solo si el juez considera que los pasos resolverían el problema.
//   - Los tickets llevan la etiqueta eval_auto. Se usa el mismo agente que atiende Teams.
import crypto from "crypto";
import { GoogleGenAI } from "@google/genai";
import { handleMessage } from "./agent.js";

const ALLOWED = () => (process.env.EVAL_ALLOWED_EMAILS || "").split(",").map((s) => s.trim()).filter(Boolean);
const AUD = () => process.env.EVAL_AUDIENCE || "itsm-agent-eval";
const USER_MODEL = () => process.env.EVAL_USER_MODEL || "gemini-2.5-flash";
// Juez: modelo Pro distinto del agente. Los Gemini 3 Pro solo están en la ubicación "global"; se prueba
// la lista en orden y se registra el primero disponible.
const JUDGE_CANDIDATES = () => [process.env.EVAL_JUDGE_MODEL, "gemini-3.1-pro-preview", "gemini-3-pro-preview", "gemini-3-pro", "gemini-2.5-pro"].filter(Boolean);
let judgeModel = null;
const JUDGE_MODEL = () => judgeModel || JUDGE_CANDIDATES()[0];
const clients = {};
const clientAt = (loc) => (clients[loc] ||= new GoogleGenAI({ vertexai: true, project: process.env.GCP_PROJECT, location: loc }));
const client = () => clientAt(process.env.GCP_LOCATION || "us-central1");

async function judgeGen(system, user) {
  const loc = process.env.EVAL_JUDGE_LOCATION || "global";
  for (const m of judgeModel ? [judgeModel] : JUDGE_CANDIDATES()) {
    try {
      const r = await clientAt(loc).models.generateContent({ model: m, contents: user,
        config: { systemInstruction: system, temperature: 0, responseMimeType: "application/json" } });
      judgeModel = m;
      return JSON.parse((r.text || "").replace(/```json|```/g, "").trim());
    } catch (e) {
      if (judgeModel || !/404|NOT_FOUND|not found/i.test(String(e.message))) throw e;
      console.error(`judge model ${m} unavailable at ${loc}`);
    }
  }
  throw new Error("no judge model available");
}

async function gen(model, system, user, json) {
  if (process.env.MOCK === "true") return json ? { would_solve: 1, grounded: 1, hallucination: 0, rationale: "mock" } : "No sale la boleta en la caja 1, desde hoy";
  const config = { systemInstruction: system, temperature: json ? 0 : 0.4 };
  if (json) config.responseMimeType = "application/json";
  const r = await client().models.generateContent({ model, contents: user, config });
  const t = (r.text || "").replace(/```json|```/g, "").trim();
  return json ? JSON.parse(t) : t;
}

export async function authorized(req) {
  const emails = ALLOWED();
  if (!emails.length) return false;
  const h = req.headers.authorization || "";
  if (!h.startsWith("Bearer ")) return false;
  const r = await fetch(`https://oauth2.googleapis.com/tokeninfo?id_token=${encodeURIComponent(h.slice(7))}`);
  if (!r.ok) return false;
  const t = await r.json();
  return t.aud === AUD() && t.email_verified === "true" && emails.includes(t.email);
}

const USER_SYS = (c) => `Eres personal de una tienda (restaurante) que escribe a la mesa de ayuda de TI por chat.
Tu situación, tal como quedó registrada en el ticket original: "${c.title}".
Reglas: escribe en español, breve (1-2 frases), como un usuario real, sin códigos de ticket ni "XXX".
Si la situación es un aviso o informativo, repórtalo como tal. Responde las preguntas del asistente con datos
plausibles y consistentes (local, caja, desde cuándo, qué intentaste: como máximo reiniciar el equipo).
Nunca propongas tú la solución técnica. Devuelve solo el texto del mensaje.`;

const JUDGE_SYS = `Eres un evaluador experto de mesas de ayuda de TI. Recibes: el problema, la solución real aplicada por el
técnico, los casos recuperados por el agente y los pasos que el agente propuso al usuario de tienda.
Evalúa con criterio estricto y responde SOLO JSON:
{"would_solve":0|1, "grounded":0|1, "hallucination":0|1, "rationale":"<1-2 frases>"}
- would_solve=1 si, siguiendo los pasos, el problema quedaría resuelto según la solución real (o si los pasos
  coinciden con la acción esencial de la solución real). Si la solución real requiere a un técnico o acceso remoto
  y los pasos no la cubren, would_solve=0.
- grounded=1 si cada paso propuesto está respaldado por al menos uno de los casos recuperados.
- hallucination=1 si los pasos mencionan sistemas, comandos o acciones que no aparecen en los casos ni en el problema.`;

export async function runCase(c) {
  const trace = {};
  const user = { id: `eval-${c.case_id}-${crypto.randomUUID().slice(0, 8)}`, name: `Evaluación ${c.case_id}`,
                 extraLabels: ["eval_auto"], trace };
  const t0 = Date.now();
  const transcript = [];
  let msg = c.oos ? c.title : await gen(USER_MODEL(), USER_SYS(c), "Escribe tu primer mensaje a la mesa de ayuda.", false);
  let judge = null;
  for (let turn = 0; turn < 5; turn++) {
    transcript.push({ role: "user", text: msg });
    const reply = await handleMessage(user, msg);
    transcript.push({ role: "agent", text: reply });
    if (trace.state === "done") break;
    if (trace.state === "await_confirm") {
      if (!judge) {
        const cases = (trace.cls?.cases || []).map((x, i) => `${i + 1}. ${x.titulo} -> ${x.solucion}`).join("\n");
        judge = process.env.MOCK === "true" ? await gen(null, JUDGE_SYS, "", true) : await judgeGen(JUDGE_SYS,
          `Problema: ${c.title}\nSolución real: ${c.resolution || "(fuera de alcance)"}\n\nCasos recuperados:\n${cases || "(ninguno)"}\n\nPasos propuestos:\n${trace.proposal}`, true);
      }
      msg = judge.would_solve ? "sí, ya funciona" : "no, sigue igual";
      continue;
    }
    const conv = transcript.map((m) => `${m.role === "user" ? "Tú" : "Asistente"}: ${m.text}`).join("\n");
    msg = await gen(USER_MODEL(), USER_SYS(c), `Conversación hasta ahora:\n${conv}\n\nEscribe tu siguiente mensaje.`, false);
  }
  const cls = trace.cls?.classification;
  return {
    case_id: c.case_id, oos: !!c.oos, unseen: c.unseen, true_category_id: c.category_id,
    outcome: trace.outcome || "incomplete", ticket: trace.ticket || null, questions: trace.questions || 0,
    clf_category_id: cls?.category_id ?? null, clf_probability: cls?.probability ?? null, clf_confident: !!cls?.confident,
    clf_correct: c.oos ? null : cls?.category_id === c.category_id,
    filter_category_id: trace.cls?.filter_category_id ?? null, case_ids: (trace.cls?.cases || []).map((x) => x.id),
    proposal: trace.proposal || null, judge,
    seconds_to_proposal: trace.seconds_to_proposal ?? trace.seconds_to_decision ?? null,
    classifier_roundtrip_ms: trace.classifier_roundtrip_ms ?? null, seconds_total: (Date.now() - t0) / 1000,
    judge_model: JUDGE_MODEL(), user_model: USER_MODEL(), transcript,
  };
}
