"""Incident classifier + retrieval service for the service desk agent (Cloud Run).

Assets (agent_assets.joblib) are built offline by itsm.agent_assets in the ITSM-paper repository
and stored in a private GCS bucket; they are never committed. Set ASSETS_URI=gs://bucket/path or
ASSETS_PATH=/local/file.

Endpoints
  GET  /health
  POST /classify  {"text": "..."}                          -> category, probability, top-3
  POST /retrieve  {"text": "...", "category_id": 3|null, "k": 5}
  POST /assist    {"text": "...", "k": 5}  -> classify, then retrieve; the category filter is
                  applied only when probability >= threshold (0.8), as in the paper's evaluation
"""
from __future__ import annotations

import os
import re
import tempfile
import time
import unicodedata

import joblib
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)


def _load():
    path = os.environ.get("ASSETS_PATH")
    if not path:
        uri = os.environ["ASSETS_URI"]
        from google.cloud import storage
        bucket, blob = uri[5:].split("/", 1)
        path = os.path.join(tempfile.gettempdir(), "agent_assets.joblib")
        storage.Client().bucket(bucket).blob(blob).download_to_filename(path)
    return joblib.load(path)


A = _load()
THRESHOLD = float(os.environ.get("THRESHOLD", A["threshold"]))
CATS = A["categories"]

# ---- text normalization: same steps as itsm.text.TextNormalizer + plural folding + domain stop words
SPANISH_STOPWORDS = set("""
de la que el en y a los del se las por un para con no una su al lo como mas pero sus le ya o este si porque esta
entre cuando muy sin sobre tambien me hasta hay donde quien desde todo nos durante todos uno les ni contra otros ese
eso ante ellos e esto mi antes algunos que unos yo otro otras otra el tanto esa estos mucho quienes nada muchos cual
poco ella estar estas algunas algo nosotros mi mis tu te ti tu tus ellas nosotras vosotros vosotras os mio mia mios
mias tuyo tuya tuyos tuyas suyo suya suyos suyas nuestro nuestra nuestros nuestras vuestro vuestra vuestros vuestras
esos esas estoy estas esta estamos estais estan este estes estemos esteis esten estare estaras estara estaremos
estareis estaran estaria estarias estariamos estariais estarian estaba estabas estabamos estabais estaban estuve
estuviste estuvo estuvimos estuvisteis estuvieron he has ha hemos habeis han haya hayas hayamos hayais hayan habre
habras habra habremos habreis habran habria habrias habriamos habriais habrian habia habias habiamos habiais habian
hube hubiste hubo hubimos hubisteis hubieron soy eres es somos sois son sea seas seamos seais sean sere seras sera
seremos sereis seran seria serias seriamos seriais serian era eras eramos erais eran fui fuiste fue fuimos fuisteis
fueron tengo tienes tiene tenemos teneis tienen tenga tengas tengamos tengais tengan tendre tendras tendra tendremos
tendreis tendran tendria tendrias tendriamos tendriais tendrian tenia tenias teniamos teniais tenian tuve tuviste
tuvo tuvimos tuvisteis tuvieron tened siendo sido tenido tenida tenidos tenidas
""".split())
_C = A["text_cfg"]
_URL = re.compile(r"(https?://|www\.)\S+")
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_PHRASES = [re.compile(rf"\b{re.escape(p)}\b") for p in sorted(_C["boilerplate_phrases"], key=len, reverse=True)]
_ABBR = {k.lower(): v for k, v in (_C.get("abbreviations") or {}).items()}
_DOMAIN = set(_C.get("domain_stopwords") or [])
_PM = A["plural_map"]


def clean(text: str) -> str:
    t = _URL.sub(" ", text or "").lower()
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    t = _NON_ALNUM.sub(" ", t)
    for p in _PHRASES:
        t = p.sub(" ", t)
    out = []
    for tok in t.split():
        tok = _ABBR.get(tok, tok)
        if any(ch.isdigit() for ch in tok) or len(tok) < _C["min_token_len"] or tok in ("rn", "xxx"):
            continue
        if tok in SPANISH_STOPWORDS:
            continue
        tok = _PM.get(tok, tok)
        if tok in _DOMAIN:
            continue
        out.append(tok)
    return " ".join(out)


def classify(text: str) -> dict:
    c = clean(text)
    X = A["vectorizer"].transform([c])
    if X.nnz == 0:   # no known vocabulary: the model would only return the prior
        return {"clean_text": c, "category_id": None, "category": "Sin clasificar", "probability": 0.0,
                "confident": False, "top3": [], "reason": "no_known_terms"}
    p = A["model"].predict_proba(X)[0]
    order = np.argsort(-p)[:3]
    best = int(A["model"].classes_[order[0]])
    return {"clean_text": c, "category_id": best, "category": CATS[best]["name_es"],
            "probability": round(float(p[order[0]]), 4), "confident": bool(p[order[0]] >= THRESHOLD),
            "top3": [{"category_id": int(A["model"].classes_[i]), "category": CATS[int(A["model"].classes_[i])]["name_es"],
                      "probability": round(float(p[i]), 4)} for i in order]}


def retrieve(text: str, category_id=None, k: int = 5) -> list[dict]:
    q = A["kb_vectorizer"].transform([clean(text)])
    s = (q @ A["kb_matrix"].T).toarray()[0]
    kb = A["kb"]
    if category_id is not None:
        s = np.where(kb["category_id"].values == category_id, s, -1.0)
    s = s + np.random.RandomState(0).uniform(0, 1e-6, size=len(s))   # fixed tie-break, as offline
    top = np.argsort(-s)[:k]
    return [{"id": kb.at[i, "id"], "titulo": kb.at[i, "titulo"], "categoria": kb.at[i, "categoria"],
             "solucion": kb.at[i, "solucion"], "score": round(float(max(s[i], 0)), 4)} for i in top if s[i] > 0]


@app.get("/health")
def health():
    return jsonify(ok=True, version=A["version"], kb_size=len(A["kb"]), threshold=THRESHOLD)


@app.post("/classify")
def r_classify():
    t0 = time.perf_counter()
    out = classify((request.get_json(force=True) or {}).get("text", ""))
    out["latency_ms"] = round(1000 * (time.perf_counter() - t0), 2)
    return jsonify(out)


@app.post("/retrieve")
def r_retrieve():
    b = request.get_json(force=True) or {}
    return jsonify(cases=retrieve(b.get("text", ""), b.get("category_id"), int(b.get("k", 5))))


@app.post("/assist")
def r_assist():
    b = request.get_json(force=True) or {}
    t0 = time.perf_counter()
    c = classify(b.get("text", ""))
    t1 = time.perf_counter()
    filt = c["category_id"] if c["confident"] else None
    # no known vocabulary -> likely out of scope: return no cases rather than spurious matches
    cases = [] if c["category_id"] is None else retrieve(b.get("text", ""), filt, int(b.get("k", 5)))
    t2 = time.perf_counter()
    return jsonify(classification=c, filter_category_id=filt, cases=cases, version=A["version"],
                   latency_ms={"classify": round(1000 * (t1 - t0), 2), "retrieve": round(1000 * (t2 - t1), 2)})
