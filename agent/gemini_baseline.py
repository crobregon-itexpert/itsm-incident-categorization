"""External Gemini baseline (zero/few-shot) and LLM label audit for the ITSM paper.

Reads gemini_inputs.json (anonymized; from the private bucket), calls Gemini on Vertex AI and writes
gemini_results.json. Titles are classified in batches of 25 per request; temperature 0.
  zero-shot : category names + definitions only
  few-shot  : + 3 training titles per category
  audit     : for each of 300 audit tickets (title + resolution), is the cluster label correct?
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from google import genai
from google.genai import types

MODEL = os.environ.get("BASELINE_MODEL", "gemini-2.5-flash")
client = genai.Client(vertexai=True, project=os.environ["PROJECT_ID"], location=os.environ.get("LOCATION", "us-central1"),
                      http_options=types.HttpOptions(timeout=120_000))  # ms: no request may hang the run
D = json.load(open(sys.argv[1], encoding="utf-8"))
cats = "\n".join(f'{c["id"]}: {c["name"]} — {c["definition"]}' for c in D["categories"])
few = "\n".join(f'- "{f["title"]}" -> {f["category_id"]}' for f in D["fewshot"])


def call(system, prompt, schema):
    for a in range(5):
        try:
            r = client.models.generate_content(model=MODEL, contents=prompt, config=types.GenerateContentConfig(
                system_instruction=system, temperature=0, response_mime_type="application/json", response_schema=schema,
                thinking_config=types.ThinkingConfig(thinking_budget=0)))
            return json.loads(r.text)
        except Exception as e:
            print("retry", a, e, flush=True); time.sleep(5 * (a + 1))
    return None


SYS_CLS = ("Clasificas tickets de mesa de ayuda de TI de una cadena de restaurantes en una de 10 categorías, usando solo el "
           "título del ticket (anonimizado: XXX y dígitos en 0 son marcadores). Categorías:\n" + cats)
LIST = {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {"i": {"type": "INTEGER"}, "category_id": {"type": "INTEGER"}},
                                   "required": ["i", "category_id"]}}


def classify(mode):
    sysmsg = SYS_CLS + ("\n\nEjemplos de entrenamiento:\n" + few if mode == "fewshot" else "")
    out = {}
    B = D["baseline"]

    def job(s):
        chunk = B[s:s + 25]
        prompt = "Clasifica cada título. Devuelve un arreglo JSON con {i, category_id} para cada uno.\n" + \
                 "\n".join(f'{s + j}: "{x["title"]}"' for j, x in enumerate(chunk))
        return call(sysmsg, prompt, LIST) or []
    with ThreadPoolExecutor(8) as ex:
        for k, r in enumerate(ex.map(job, range(0, len(B), 25))):
            for o in r:
                if 0 <= o["i"] < len(B):
                    out[B[o["i"]]["tid"]] = o["category_id"]
            print(mode, "batch", k + 1, flush=True)
    return out


AUD = {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
    "audit_id": {"type": "STRING"}, "label_correct": {"type": "STRING", "enum": ["si", "no", "dudoso"]},
    "best_category_id": {"type": "INTEGER"}}, "required": ["audit_id", "label_correct", "best_category_id"]}}
SYS_AUD = ("Eres auditor de una taxonomía de incidentes de TI de una cadena de restaurantes. Para cada ticket (título y nota "
           "de resolución, anonimizados) decide si la categoría asignada es correcta según las definiciones: 'si', 'no' o "
           "'dudoso', e indica la categoría que mejor le corresponde. Categorías:\n" + cats)


def audit():
    res = []
    A = D["audit"]

    def job(s):
        chunk = A[s:s + 20]
        prompt = "Audita cada ticket. Devuelve un arreglo JSON.\n" + "\n".join(
            f'{x["audit_id"]} | categoría asignada: {x["cluster_label"]} | título: "{x["title"]}" | resolución: "{x["resolution"]}"'
            for x in chunk)
        return call(SYS_AUD, prompt, AUD) or []
    with ThreadPoolExecutor(8) as ex:
        for k, r in enumerate(ex.map(job, range(0, len(A), 20))):
            res += r
            print("audit batch", k + 1, flush=True)
    return res


t0 = time.time()
result = {"model": MODEL, "zeroshot": classify("zeroshot"), "fewshot": classify("fewshot"), "audit": audit()}
result["seconds"] = time.time() - t0
json.dump(result, open("gemini_results.json", "w", encoding="utf-8"), ensure_ascii=False)
print("done", {k: len(v) for k, v in result.items() if isinstance(v, (list, dict))})
