"""Latency and size of the deployable classifiers (reviewer comment 8).

    python -m itsm.benchmark --config configs/phase1.yaml

For each classical model: refit on the seed-0 training split with the hyperparameters
selected in that run, then time the full path a new ticket takes in production
(raw title -> cleaning -> TF-IDF -> prediction), one ticket at a time, on CPU.
"""
from __future__ import annotations

import argparse
import json
import pickle
import platform
import time

import numpy as np
import pandas as pd

from .evaluate import run_dir, setting_name
from .runlog import ROOT, load_config, run_context, write_json
from .train_classical import MODELS, load_task, split_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--n", type=int, default=1000)
    cfg = load_config(ap.parse_args().config)
    n = ap.parse_args().n
    c = cfg["classification"]
    k = cfg["clustering"]["selected_k"]
    setting = setting_name(c["target"], c["input_fields"], k)
    df, _ = load_task(cfg, c["target"], k)
    part = split_for(cfg, df, 0)
    tr, te = np.where(part == "train")[0], np.where(part == "test")[0]
    text = df[c["input_fields"]].fillna("").agg(" ".join, axis=1).str.strip().values

    from .text import TextNormalizer
    norm = TextNormalizer(cfg["text"])          # cleaning cost is part of the path
    raw = (df["title_raw"] if "title_raw" in df else df["title_bert"]).values[te][:n]

    rows = []
    for name in c["models"]:
        f = run_dir(cfg, setting, name, 0) / "metrics.json"
        if not f.exists():
            continue
        params = json.loads(f.read_text(encoding="utf-8"))["best_params"]
        factory, _, kind = MODELS[name]
        # Build vectorizers on train (same code path as training)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from scipy.sparse import hstack
        w = c["tfidf_word"]
        wv = TfidfVectorizer(ngram_range=tuple(w["ngram_range"]), min_df=w["min_df"],
                             max_features=w["max_features"], sublinear_tf=w["sublinear_tf"]).fit(text[tr])
        vecs = [wv]
        if kind == "word_char":
            ch = c["tfidf_char"]
            vecs.append(TfidfVectorizer(analyzer=ch["analyzer"], ngram_range=tuple(ch["ngram_range"]),
                                        min_df=ch["min_df"], max_features=ch["max_features"],
                                        sublinear_tf=ch["sublinear_tf"]).fit(text[tr]))

        def vectorize(t):
            mats = [v.transform(t) for v in vecs]
            return mats[0] if len(mats) == 1 else hstack(mats).tocsr()

        Xtr = vectorize(text[tr])
        if name == "mlp":
            from .train_classical import fit_mlp_epochs
            va = np.where(part == "val")[0]
            m, _ = fit_mlp_epochs(factory, params, 0, Xtr, df["y"].values[tr], vectorize(text[va]),
                                  df["y"].values[va], sorted(df["y"].unique()))
        else:
            m = factory(params, 0).fit(Xtr, df["y"].values[tr])
        for t in raw[:20]:                       # warm-up
            m.predict(vectorize([norm(t)]))
        lat = []
        for t in raw:
            t0 = time.perf_counter()
            m.predict(vectorize([norm(t)]))
            lat.append(1000 * (time.perf_counter() - t0))
        size_mb = len(pickle.dumps((vecs, m))) / 1e6
        rows.append({"model": name, "params": params, "n_tickets": len(lat),
                     "latency_ms_median": float(np.median(lat)), "latency_ms_p95": float(np.percentile(lat, 95)),
                     "model_size_mb": size_mb})
        print(rows[-1], flush=True)

    out = ROOT / cfg["output"]["summary_dir"] / cfg["experiment_name"] / "latency_cpu.json"
    write_json({**run_context(cfg), "cpu": platform.processor() or platform.machine(),
                "note": "single ticket, raw title -> cleaning -> TF-IDF -> predict, CPU, 1 process",
                "results": rows}, out)
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()
