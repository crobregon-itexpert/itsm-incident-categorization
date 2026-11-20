// Agente de primer contacto (usuario de tienda), usado por Microsoft Teams.
// Flujo (paper, Sección VI):
//   1. INTAKE: Gemini hace como máximo 2 preguntas (o detecta fuera de alcance).
//   2. CLASIFICAR + RECUPERAR: servicio itsm-classifier (/assist). El filtro por categoría se
//      aplica solo si p >= 0.8; si no, se busca en toda la base y la categoría queda "Sin clasificar".
//   3. PROPONER: Gemini propone pasos seguros para el usuario basados solo en los casos recuperados.
//   4. CONFIRMAR: "¿Se resolvió?" -> sí: ticket creado y cerrado como resuelto por el agente;
//      no (o sin pasos): ticket para el técnico con categoría, probabilidad, conversación y casos.
// Cada decisión se registra como un log estructurado (event=agent_case) para la evaluación.
import crypto from "crypto";
import { intakeSystem, selfHelpSystem, CATEGORIES, UNCLASSIFIED } from "./prompts.js";
import { generateJSON, generateText } from "./gemini.js";
import { buildLabels } from "./labels.js";
import { assist } from "./classifier.js";
import * as jira from "./jira.js";

const SESSIONS = new Map();
const TTL_MS = 2 * 60 * 60 * 1000;
const todayISO = () => new Date().toISOString().slice(0, 10);
// \b de JS no reconoce letras acentuadas: se usa un lookahead explícito
const END = "(?![a-záéíóúñ])";
const YES = new RegExp(`^\\s*(s[ií]${END}|ya${END}|listo|resuelto|solucionado|funcion[oó]|ok${END}|okay|perfecto)`, "i");
const NO = new RegExp(`^\\s*(no${END}|nop|sigue|a[uú]n no|todav[ií]a|tampoco|persiste|igual)`, "i");

const hash = (s) => crypto.createHash("sha256").update(String(s)).digest("hex").slice(0, 12);
const log = (o) => console.log(JSON.stringify({ event: "agent_case", ...o }));

function session(userId) {
  const s = SESSIONS.get(userId);
  if (s && Date.now() - s.updated < TTL_MS) return s;
  const n = { id: crypto.randomUUID(), state: "intake", history: [], started: Date.now(), updated: Date.now() };
  SESSIONS.set(userId, n);
  return n;
}

const userText = (s) => s.history.filter((h) => h.startsWith("Usuario: ")).map((h) => h.slice(9)).join(". ");

function describeCases(cases) {
  return cases.map((c, i) => `${i + 1}. [${c.id}] ${c.titulo} -> Solución: ${c.solucion}`).join("\n");
}

async function createTicket(s, user, { resolvedByAgent, kind = "incident" }) {
  const c = s.cls?.classification;
  const confident = !!c?.confident;
  const category = confident ? c.category : UNCLASSIFIED;
  const descr =
    `Reportado por: ${user.name} (vía Microsoft Teams)\n` +
    `Categoría (clasificador): ${category}` + (c?.probability != null ? ` — p=${c.probability}` : "") + "\n" +
    (c?.top3?.length ? `Top-3: ${c.top3.map((t) => `${t.category} (${t.probability})`).join("; ")}\n` : "") +
    (s.intake?.category ? `Categoría sugerida por Gemini (referencial): ${s.intake.category}\n` : "") +
    `\nResumen: ${s.intake?.summary || userText(s)}\n` +
    (s.proposal ? `\nPasos propuestos por el agente:\n${s.proposal}\n` : "") +
    `\nTipo: ${kind === "request" ? "Solicitud" : kind === "informative" ? "Aviso informativo" : "Incidente"}\n` +
    (kind === "incident" ? `Resultado con el usuario: ${resolvedByAgent ? "RESUELTO por el usuario con los pasos propuestos" : "NO resuelto — requiere técnico"}\n` : "") +
    (s.cls?.cases?.length ? `\nCasos similares recuperados:\n${describeCases(s.cls.cases)}\n` : "") +
    `\nConversación:\n${s.history.join("\n")}`;
  const labels = buildLabels({
    category: confident ? category : null, sede: "Teams", empresa: "Corporativo", repDate: todayISO(),
    extra: [confident ? "clf_confiable" : "clf_sin_clasificar", kind === "incident" ? (resolvedByAgent ? "resuelto_por_agente" : "requiere_tecnico") : `tipo_${kind}`, ...(user.extraLabels || [])],
  });
  const ticket = await jira.createIssue({ summary: `[Teams] ${s.intake?.title || "Incidente"}`, descriptionText: descr, labels });
  if (resolvedByAgent && process.env.JIRA_TRANSITION_CLOSE_ID) {
    try {
      await jira.addComment(ticket.key, "[Resuelto en primer contacto por el agente — confirmado por el usuario]");
      await jira.transition(ticket.key, process.env.JIRA_TRANSITION_CLOSE_ID);
    } catch (e) { console.error("No se pudo cerrar el ticket resuelto por el agente:", e.message); }
  }
  return { ...ticket, url: jira.browseUrl(ticket.key), category };
}

// Procesa un mensaje del usuario y devuelve el texto a responder.
export async function handleMessage(user, text) {
  if (/^\s*(reiniciar|reset|nuevo caso)\s*$/i.test(text)) {
    SESSIONS.delete(user.id);
    return "Listo, empecemos de nuevo. ¿Qué problema tienes?";
  }
  const s = session(user.id);
  s.updated = Date.now();
  s.history.push(`Usuario: ${text}`);
  const base = { conversation: s.id, user: hash(user.id) };

  if (s.state === "await_confirm") {
    const yes = YES.test(text) && !NO.test(text);
    const ticket = await createTicket(s, user, { resolvedByAgent: yes });
    log({ ...base, step: "outcome", resolved_by_agent: yes, ticket: ticket.key, category: ticket.category,
          seconds_total: Math.round((Date.now() - s.started) / 1000) });
    if (user.trace) Object.assign(user.trace, { state: "done", outcome: yes ? "resolved_by_agent" : "ticket_for_technician", ticket: ticket.key });
    SESSIONS.delete(user.id);
    return yes
      ? `¡Excelente! Dejé registrado el caso como resuelto: [${ticket.key}](${ticket.url}).`
      : `Entendido. Creé el ticket [${ticket.key}](${ticket.url}) para un técnico, con la información y los pasos que ya probaste.`;
  }

  // 1. Intake
  const out = await generateJSON(intakeSystem(), s.history.join("\n"));
  if (out.action === "out_of_scope") {
    log({ ...base, step: "out_of_scope" });
    if (user.trace) Object.assign(user.trace, { state: "done", outcome: "out_of_scope" });
    SESSIONS.delete(user.id);
    return out.message;
  }
  if (out.action !== "ready") {
    if (user.trace) Object.assign(user.trace, { state: "intake", questions: (user.trace.questions || 0) + 1 });
    s.history.push(`Asistente: ${out.message}`);
    return out.message;
  }
  s.intake = { ...out, category: CATEGORIES.includes(out.category) ? out.category : null };

  // 2. Clasificar + recuperar
  const query = `${userText(s)}. ${out.title || ""}`;
  const tc = Date.now();
  try {
    s.cls = await assist(query, 5);
    s.roundtrip_ms = Date.now() - tc;
  } catch (e) {
    console.error("itsm-classifier no disponible:", e.message);
    s.cls = null;
  }
  const c = s.cls?.classification;

  // Solicitudes y avisos: se registran sin proponer pasos de autoservicio.
  const kind = ["request", "informative"].includes(out.kind) ? out.kind : "incident";
  if (kind !== "incident") {
    const ticket = await createTicket(s, user, { resolvedByAgent: false, kind });
    log({ ...base, step: "outcome", kind, ticket: ticket.key, category: ticket.category,
          clf_category_id: c?.category_id ?? null, clf_probability: c?.probability ?? null, clf_confident: !!c?.confident,
          classifier_roundtrip_ms: s.roundtrip_ms ?? null, seconds_total: Math.round((Date.now() - s.started) / 1000) });
    if (user.trace) Object.assign(user.trace, { state: "done", outcome: `${kind}_ticket`, ticket: ticket.key, cls: s.cls, intake: s.intake,
                                               seconds_to_decision: (Date.now() - s.started) / 1000, classifier_roundtrip_ms: s.roundtrip_ms });
    SESSIONS.delete(user.id);
    return `${out.message}\n\nRegistré tu ${kind === "request" ? "solicitud" : "aviso"} en el ticket [${ticket.key}](${ticket.url}).`;
  }

  // 3. Proponer pasos (solo con casos recuperados)
  let proposal = "SIN_PASOS";
  if (s.cls?.cases?.length) {
    const ctx = `Incidente: ${out.summary}\n\nCasos resueltos similares:\n${describeCases(s.cls.cases)}`;
    proposal = (await generateText(selfHelpSystem(), ctx)).trim();
  }
  log({ ...base, step: "proposal", seconds_to_proposal: Math.round((Date.now() - s.started) / 1000),
        clf_category_id: c?.category_id ?? null, clf_probability: c?.probability ?? null, clf_confident: !!c?.confident,
        gemini_category: s.intake.category, filter_category_id: s.cls?.filter_category_id ?? null,
        case_ids: (s.cls?.cases || []).map((x) => x.id), latency_ms: s.cls?.latency_ms ?? null, classifier_roundtrip_ms: s.roundtrip_ms ?? null,
        has_steps: !/SIN_PASOS/.test(proposal), version: s.cls?.version ?? null });

  if (/SIN_PASOS/.test(proposal)) {
    const ticket = await createTicket(s, user, { resolvedByAgent: false });
    log({ ...base, step: "outcome", resolved_by_agent: false, no_steps: true, ticket: ticket.key, category: ticket.category,
          seconds_total: Math.round((Date.now() - s.started) / 1000) });
    if (user.trace) Object.assign(user.trace, { state: "done", outcome: "no_steps_ticket", ticket: ticket.key, cls: s.cls, intake: s.intake,
                                               seconds_to_decision: (Date.now() - s.started) / 1000 });
    SESSIONS.delete(user.id);
    return `${out.message}\n\nEste caso necesita un técnico. Creé el ticket [${ticket.key}](${ticket.url}).`;
  }
  s.proposal = proposal;
  s.state = "await_confirm";
  if (user.trace) Object.assign(user.trace, { state: "await_confirm", proposal, cls: s.cls, intake: s.intake,
                                             seconds_to_proposal: (Date.now() - s.started) / 1000, classifier_roundtrip_ms: s.roundtrip_ms });
  s.history.push(`Asistente: ${proposal}`);
  return `${proposal}\n\n¿Se resolvió el problema? Responde **sí** o **no**.`;
}
