"""Build the assets of the deployed agent's classifier/retrieval service (Phase 2).

    python -m itsm.agent_assets --config configs/phase1_v2.yaml --seed 0

Output (local only, git-ignored; uploaded by hand to a private GCS bucket):
  data/agent/agent_assets.joblib
    - classifier: word TF-IDF + MLP (same features, grid and validation-based epoch selection as
      itsm.train_classical, trained on the seed-0 training split), probabilities over the 10
      categories, confidence threshold 0.8 (Table V)
    - text normalizer config + plural map restricted to the classifier vocabulary, so a user's
      free-text message is cleaned exactly like the training titles (no name lists are shipped)
    - knowledge base: seed-0 training incidents with a resolution note, deduplicated on
      (title, note); title and note anonymized with the same rules as the shared dataset
      (names and store slots masked, digits -> 0, words in fewer than 5 notes -> XXX); test
      tickets are never included
    - retriever: TF-IDF (1-2 grams) on the cleaned anonymized title, as in itsm.retrieval_eval
  results/summary/<exp>/agent_assets.json  -> sizes, sha256, test accuracy of the shipped model
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

from .anonymize import doc_freq, mask_raw, mask_rare_raw
from .prepare import load_raw
from .runlog import ROOT, file_sha256, load_config, run_context, write_json
from .text import TextNormalizer, apply_map, name_stopwords, plural_map, strip_solution_template
from .train_classical import MODELS, fit_mlp_epochs


def build(cfg, seed=0):
    k_anon = cfg["text"]["anonymize"]["min_ticket_freq"]
    sh = pd.read_csv(ROOT / cfg["data"]["shared_csv"], keep_default_na=False)
    part = sh[f"split_seed{seed}"].values
    tr, va, te = (part == "train"), (part == "val"), (part == "test")
    y = sh["category_id"].astype(int).values
    labels = list(range(10))

    # ---- classifier (identical to the paper's MLP, seed 0) ----
    w = cfg["classification"]["tfidf_word"]
    vec = TfidfVectorizer(ngram_range=tuple(w["ngram_range"]), min_df=w["min_df"],
                          max_features=w["max_features"], sublinear_tf=w["sublinear_tf"])
    X = sh["title_clean"].values
    Xtr = vec.fit_transform(X[tr]); Xva = vec.transform(X[va]); Xte = vec.transform(X[te])
    factory, grid, _ = MODELS["mlp"]
    best = None
    for params in grid:
        m, curve = fit_mlp_epochs(factory, params, seed, Xtr, y[tr], Xva, y[va], labels)
        f1 = f1_score(y[va], m.predict(Xva), labels=labels, average="macro", zero_division=0)
        if best is None or f1 > best[0]:
            best = (f1, params, m, len(curve))
    _, params, mlp, n_epochs = best
    p_te = mlp.predict(Xte)
    test = {"accuracy": float(accuracy_score(y[te], p_te)),
            "macro_f1": float(f1_score(y[te], p_te, labels=labels, average="macro", zero_division=0))}
    P = mlp.predict_proba(Xte); conf = P.max(1); yp = mlp.classes_[P.argmax(1)]; yt = y[te]
    test["threshold_0.8"] = {"coverage": float((conf >= 0.8).mean()),
                             "accuracy": float((yt[conf >= 0.8] == yp[conf >= 0.8]).mean())}
    b = np.minimum((conf * 10).astype(int), 9)
    test["ece_10bins"] = float(sum((b == i).mean() * abs((yt[b == i] == yp[b == i]).mean() - conf[b == i].mean())
                                   for i in range(10) if (b == i).any()))

    # ---- inference-time normalizer: same steps as itsm.prepare, without the private name list ----
    raw = load_raw(cfg)
    d = cfg["data"]
    raw = raw[raw[d["type_col"]].isin(d["ticket_types"])].copy()
    names = name_stopwords(raw, cfg["text"].get("name_cols"), cfg["text"].get("name_token_keep"))
    norm_private = TextNormalizer(cfg["text"], extra_stopwords=names)
    titles_n = raw[d["title_col"]].fillna("").astype(str).map(norm_private)
    vocab_tokens = {t for f in vec.get_feature_names_out() for t in f.split()}
    pm_full = plural_map(titles_n)
    pm = {a: b for a, b in pm_full.items() if b in vocab_tokens}   # only maps onto model vocabulary
    text_cfg = {kk: cfg["text"][kk] for kk in ("lowercase", "strip_accents", "remove_stopwords",
                                               "drop_tokens_with_digits", "min_token_len",
                                               "boilerplate_phrases", "abbreviations", "domain_stopwords")}

    # ---- knowledge base: training split only ----
    idmap = pd.read_csv(ROOT / "data/share/private_id_map.csv", dtype={"ticket_id": str})
    t = pd.read_pickle(ROOT / cfg["data"]["interim_path"])
    kb = sh.loc[tr, ["tid", "title_bert", "title_clean", "category_id", "category_es"]].merge(idmap, on="tid").merge(
        t[["ticket_id", "solution_raw", "solution", "solution_missing"]], on="ticket_id")
    kb = kb[~kb.solution_missing & (kb.solution.str.len() > 0)].copy()
    tcfg = cfg["text"].get("solution_template")
    from .text import _RN_GLUED
    sol = kb["solution_raw"].map(lambda v: mask_raw(_RN_GLUED.sub(" ", strip_solution_template(v, tcfg)), names))
    dfs = doc_freq(sol)
    kb["solucion"] = sol.map(lambda v: mask_rare_raw(v, dfs, k_anon))
    # notes with fewer than 4 informative words ("tk informativo", "Informativo") propose nothing
    kb = kb[kb["solution"].str.split().str.len() >= 4]
    kb = kb.drop_duplicates(["title_bert", "solucion"]).reset_index(drop=True)
    kb = kb.rename(columns={"title_bert": "titulo", "category_es": "categoria"})[
        ["titulo", "title_clean", "category_id", "categoria", "solucion"]]
    kb.insert(0, "id", [f"KB-{i:05d}" for i in range(1, len(kb) + 1)])
    rvec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit(kb.title_clean)

    cats = yaml.safe_load(open(ROOT / "configs" / "categories.yaml", encoding="utf-8"))["categories"]
    assets = {
        "version": f"{cfg['experiment_name']}-seed{seed}",
        "threshold": 0.8,
        "categories": [{"id": i, "name_es": cats[i]["name_es"], "name_en": cats[i]["name_en"],
                        "definition_es": cats[i].get("definition_es", "")} for i in range(10)],
        "text_cfg": text_cfg, "plural_map": pm,
        "vectorizer": vec, "model": mlp, "model_params": {**params, "epochs": n_epochs},
        "kb": kb[["id", "titulo", "title_clean", "category_id", "categoria", "solucion"]],
        "kb_vectorizer": rvec, "kb_matrix": rvec.transform(kb.title_clean),
    }
    out = ROOT / "data" / "agent" / "agent_assets.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(assets, out, compress=3)
    kb.drop(columns="title_clean").sample(40, random_state=0).to_csv(out.parent / "kb_sample_review.csv", index=False)
    meta = {**run_context(cfg), "seed": seed, "file": out.name, "sha256": file_sha256(out),
            "size_mb": round(out.stat().st_size / 2**20, 2), "kb_size": len(kb),
            "classifier": "word TF-IDF + MLP", "params": assets["model_params"], "test": test,
            "plural_map_entries": len(pm), "threshold": 0.8}
    write_json(meta, ROOT / cfg["output"]["summary_dir"] / cfg["experiment_name"] / "agent_assets.json")
    print(meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1_v2.yaml")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(load_config(a.config), a.seed)
