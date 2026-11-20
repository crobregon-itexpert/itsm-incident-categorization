"""Step 4: classical models on TF-IDF features, one run per (setting, model, seed).

    python -m itsm.train_classical --config configs/phase1.yaml               # selected k, all models
    python -m itsm.train_classical --k 6 8 12 --models logreg linear_svm      # k sensitivity
    python -m itsm.train_classical --inputs title solution                    # upper-bound ablation
    python -m itsm.train_classical --target grupo                             # human routing label

Hyperparameters are chosen on the validation split (macro-F1) from a small grid;
the grid and all validation scores are logged. The test split is scored once.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from .evaluate import setting_name, write_run, run_dir
from .runlog import ROOT, load_config
from .splits import labels_path, load_split

# name -> (factory(params, seed), grid, feature set)
MODELS = {
    "logreg": (lambda p, s: LogisticRegression(C=p["C"], class_weight="balanced", max_iter=3000, random_state=s),
               [{"C": c} for c in (0.25, 1.0, 4.0, 16.0)], "word"),
    "linear_svm": (lambda p, s: LinearSVC(C=p["C"], class_weight="balanced", random_state=s),
                   [{"C": c} for c in (0.03, 0.1, 0.5, 1.0)], "word"),
    "svm_word_char": (lambda p, s: LinearSVC(C=p["C"], class_weight="balanced", random_state=s),
                      [{"C": c} for c in (0.1, 0.5, 1.0)], "word_char"),
    "complement_nb": (lambda p, s: ComplementNB(alpha=p["alpha"]),
                      [{"alpha": a} for a in (0.03, 0.1, 0.3, 1.0)], "word"),
    "random_forest": (lambda p, s: RandomForestClassifier(n_estimators=p["n_estimators"], class_weight="balanced_subsample",
                                                          min_samples_leaf=1, n_jobs=-1, random_state=s),
                      [{"n_estimators": 200}], "word"),
    # Trained epoch by epoch; the epoch is chosen on our validation split (see fit_mlp_epochs).
    # sklearn's built-in early stopping samples a random 10% of train, which shares duplicate
    # titles with train and stops on a leaked score.
    "mlp": (lambda p, s: MLPClassifier(hidden_layer_sizes=(p["hidden"],), alpha=p["alpha"], batch_size=256,
                                       learning_rate_init=1e-3, max_iter=1, warm_start=True, random_state=s),
            [{"hidden": 256, "alpha": 1e-4}, {"hidden": 256, "alpha": 1e-2}], "word"),
}


def split_for(cfg, df, seed):
    if f"split_seed{seed}" in df.columns:
        return df[f"split_seed{seed}"].values
    sp = load_split(cfg, seed)
    return df[["ticket_id"]].merge(sp, on="ticket_id", how="left")["split"].values


def fit_mlp_epochs(factory, params, seed, Xtr, ytr, Xva, yva, labels, max_epochs=20, patience=3):
    import copy
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    m = factory(params, seed)
    best, bad, curve = None, 0, []
    for epoch in range(1, max_epochs + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            m.fit(Xtr, ytr)
        f1 = f1_score(yva, m.predict(Xva), labels=labels, average="macro", zero_division=0)
        curve.append({"epoch": epoch, "train_loss": float(m.loss_), "val_macro_f1": float(f1)})
        if best is None or f1 > best[0]:
            best, bad = (f1, copy.deepcopy(m)), 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best[1], curve


def load_shared(cfg, target):
    """v2: everything comes from the shared anonymized CSV (itsm.export_dataset)."""
    d = pd.read_csv(ROOT / cfg["data"]["shared_csv"], keep_default_na=False)
    df = pd.DataFrame({"ticket_id": d["tid"].astype(str), "title": d["title_clean"], "title_bert": d["title_bert"],
                       "group": d["title_clean"], "ext_grupo_asignado": d["resolver_group"].replace("", None)})
    for c in d.filter(like="split_seed").columns:
        df[c] = d[c]
    if target == "cluster":
        df["y"] = d["category_id"].astype(int)
        names = d.drop_duplicates("category_id").set_index("category_id")["category_en"].to_dict()
        return df, [names[i] for i in sorted(names)]
    return df, None


def load_task(cfg, target, k, inputs=("title",)):
    # The shared CSV holds only the selected-k labels and the title; other k values and the
    # title+solution ablation read the internal (anonymized) table with the same splits.
    use_shared = bool(cfg["data"].get("shared_csv")) and set(inputs) <= {"title"} and (
        target == "grupo" or k == cfg["clustering"]["selected_k"])
    if use_shared:
        df, names = load_shared(cfg, target)
        if target == "cluster":
            return df, names
    else:
        df = pd.read_pickle(ROOT / cfg["data"]["interim_path"])
    if target == "cluster":   # pickle source (v1)
        lab = pd.read_csv(labels_path(cfg, k), dtype={"ticket_id": str})
        df = df.merge(lab, on="ticket_id", validate="one_to_one")
        df["y"] = df["cluster"].astype(int)
        names_file = ROOT / "configs" / f"cluster_names_k{k}.yaml"
        import yaml
        names = yaml.safe_load(open(names_file, encoding="utf-8")) if names_file.exists() else {}
        label_names = [str(names.get(i, i)) for i in sorted(df["y"].unique())]
    elif target == "grupo":
        col = "ext_grupo_asignado"
        vc = df[col].value_counts()
        keep = vc[vc >= cfg["classification"]["min_class_count"]].index
        df = df[df[col].isin(keep)].copy()
        cats = sorted(keep)
        df["y"] = df[col].map({c: i for i, c in enumerate(cats)})
        label_names = cats
    else:
        raise ValueError(target)
    return df, label_names


def features(cfg, kind, tr, va, te):
    c = cfg["classification"]
    w = c["tfidf_word"]
    wv = TfidfVectorizer(ngram_range=tuple(w["ngram_range"]), min_df=w["min_df"],
                         max_features=w["max_features"], sublinear_tf=w["sublinear_tf"])
    Xs = [wv.fit_transform(tr), wv.transform(va), wv.transform(te)]
    if kind == "word_char":
        ch = c["tfidf_char"]
        cv = TfidfVectorizer(analyzer=ch["analyzer"], ngram_range=tuple(ch["ngram_range"]), min_df=ch["min_df"],
                             max_features=ch["max_features"], sublinear_tf=ch["sublinear_tf"])
        C = [cv.fit_transform(tr), cv.transform(va), cv.transform(te)]
        Xs = [hstack([a, b]).tocsr() for a, b in zip(Xs, C)]
    return Xs


def run(cfg, target, inputs, ks, models, seeds, skip_existing=True):
    for k in ks:
        df, label_names = load_task(cfg, target, k, inputs)
        labels = list(range(len(label_names)))
        text = df[inputs].fillna("").agg(" ".join, axis=1).str.strip()
        setting = setting_name(target, inputs, k)
        for seed in seeds:
            part = split_for(cfg, df, seed)
            idx = {s: np.where(part == s)[0] for s in ("train", "val", "test")}
            y = df["y"].values
            feats = {}
            for name in models:
                if skip_existing and (run_dir(cfg, setting, name, seed) / "metrics.json").exists():
                    print(f"skip {setting} {name} seed{seed}")
                    continue
                factory, grid, kind = MODELS[name]
                if kind not in feats:
                    t0 = time.perf_counter()
                    feats[kind] = features(cfg, kind, text.values[idx["train"]], text.values[idx["val"]],
                                           text.values[idx["test"]])
                    feats[kind + "_sec"] = time.perf_counter() - t0
                Xtr, Xva, Xte = feats[kind]
                ytr, yva, yte = y[idx["train"]], y[idx["val"]], y[idx["test"]]
                grid_log, best = [], None
                for params in grid:
                    t0 = time.perf_counter()
                    if name == "mlp":
                        m, curve = fit_mlp_epochs(factory, params, seed, Xtr, ytr, Xva, yva, labels)
                    else:
                        m, curve = factory(params, seed).fit(Xtr, ytr), None
                    fit_s = time.perf_counter() - t0
                    f1 = f1_score(yva, m.predict(Xva), labels=labels, average="macro", zero_division=0)
                    grid_log.append({"params": params, "val_macro_f1": f1, "fit_seconds": fit_s, "epochs": curve})
                    if best is None or f1 > best[0]:
                        best = (f1, params, m, fit_s)
                _, params, m, fit_s = best
                t0 = time.perf_counter()
                p_te = m.predict(Xte)
                pred_ms = 1000 * (time.perf_counter() - t0) / len(yte)
                extra = {"target": target, "inputs": inputs, "k": k if target == "cluster" else None,
                         "features": kind, "n_features": Xtr.shape[1], "best_params": params,
                         "grid": grid_log, "fit_seconds": fit_s, "feature_seconds": feats[kind + "_sec"],
                         "predict_ms_per_ticket": pred_ms,
                         "n_train": len(ytr), "n_val": len(yva), "n_test": len(yte)}
                rec = write_run(cfg, setting=setting, model=name, seed=seed, labels=labels,
                                label_names=label_names, ids_test=df["ticket_id"].values[idx["test"]],
                                y_test=yte, p_test=p_te, y_val=yva, p_val=m.predict(Xva), extra=extra,
                                groups_test=df["group"].values[idx["test"]],
                                groups_train=df["group"].values[idx["train"]])
                print(f"{setting} {name:14s} seed{seed} acc={rec['test']['accuracy']:.4f} "
                      f"macroF1={rec['test']['macro_f1']:.4f} params={params} fit={fit_s:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1_v2.yaml")
    ap.add_argument("--target", default=None)
    ap.add_argument("--inputs", nargs="+", default=None)
    ap.add_argument("--k", nargs="+", type=int, default=None)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    c = cfg["classification"]
    run(cfg, a.target or c["target"], a.inputs or c["input_fields"],
        a.k or [cfg["clustering"]["selected_k"]], a.models or c["models"],
        a.seeds or cfg["split"]["seeds"], skip_existing=not a.force)
