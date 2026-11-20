// Cliente del servicio itsm-classifier (Cloud Run): clasificación + recuperación de casos resueltos.
// En Cloud Run se autentica con un ID token del metadata server (servicio privado);
// en local llama sin token. Si CLASSIFIER_URL no está definido, devuelve null y el agente
// sigue sin clasificador (modo demo / MOCK).
const URL_ = () => process.env.CLASSIFIER_URL;
let tok = { value: null, exp: 0 };

async function idToken(audience) {
  if (!process.env.K_SERVICE) return null; // fuera de Cloud Run
  if (tok.value && Date.now() < tok.exp) return tok.value;
  const r = await fetch(
    `http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience=${encodeURIComponent(audience)}`,
    { headers: { "Metadata-Flavor": "Google" } }
  );
  if (!r.ok) return null;
  tok = { value: await r.text(), exp: Date.now() + 50 * 60 * 1000 };
  return tok.value;
}

async function call(path, body) {
  const base = URL_();
  if (!base) return null;
  const t = await idToken(base);
  const headers = { "Content-Type": "application/json" };
  if (t) headers.Authorization = `Bearer ${t}`;
  const r = await fetch(`${base}${path}`, { method: "POST", headers, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`classifier ${path} ${r.status}: ${await r.text()}`);
  return await r.json();
}

// Clasifica y recupera k casos; filtra por categoría solo si p >= umbral (lo decide el servicio).
export const assist = (text, k = 5) => call("/assist", { text, k });
// Recuperación explícita (consola del técnico): categoryId null = sin filtro.
export const retrieveCases = (text, categoryId = null, k = 5) => call("/retrieve", { text, category_id: categoryId, k });
