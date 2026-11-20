"""Step 2: category discovery with K-means.

    python -m itsm.clustering sweep  --config configs/phase1.yaml   # choose k
    python -m itsm.clustering fit    --config configs/phase1.yaml   # write labels for selected k + k_sensitivity

Representation: TF-IDF (uni+bigrams) of title + solution -> truncated SVD (LSA) ->
L2 normalization -> K-means. Internal validity (silhouette, Davies-Bouldin, inertia),
stability (ARI between K-means seeds) and external validity (NMI/ARI against the
human fields Nivel 1, Nivel 2, Grupo Asignado) are logged for every k.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.preprocessing import Normalizer

from .runlog import ROOT, Timer, load_config, run_context, write_json


def load_tickets(cfg) -> pd.DataFrame:
    return pd.read_pickle(ROOT / cfg["data"]["interim_path"])


def cluster_text(df, fields) -> pd.Series:
    return df[list(fields)].fillna("").agg(" ".join, axis=1).str.strip()


def embed(cfg, df):
    c = cfg["clustering"]
    tf = c["tfidf"]
    vec = TfidfVectorizer(ngram_range=tuple(tf["ngram_range"]), min_df=tf["min_df"], max_df=tf["max_df"],
                          max_features=tf["max_features"], sublinear_tf=tf["sublinear_tf"])
    X_tfidf = vec.fit_transform(cluster_text(df, c["fields"]))
    svd = TruncatedSVD(n_components=c["svd_components"], random_state=0)
    X = Normalizer().fit_transform(svd.fit_transform(X_tfidf))
    return vec, svd, X_tfidf, X, float(svd.explained_variance_ratio_.sum())


def ext_cols(df):
    return [c for c in df.columns if c.startswith("ext_")]


def sweep(cfg):
    c = cfg["clustering"]
    df = load_tickets(cfg)
    _, _, _, X, evr = embed(cfg, df)
    rng = np.random.RandomState(0)
    sample = rng.choice(len(df), size=min(c["silhouette_sample"], len(df)), replace=False)
    rows = []
    for k in c["k_values"]:
        labels_by_seed = {}
        with Timer() as t:
            for s in c["stability_seeds"]:
                km = KMeans(n_clusters=k, n_init=c["n_init"], random_state=s).fit(X)
                labels_by_seed[s] = (km.labels_, km.inertia_)
        lab0, inertia0 = labels_by_seed[c["stability_seeds"][0]]
        pair_ari = [adjusted_rand_score(labels_by_seed[a][0], labels_by_seed[b][0])
                    for a, b in itertools.combinations(c["stability_seeds"], 2)]
        sizes = np.bincount(lab0, minlength=k)
        row = {
            "k": k,
            "inertia": inertia0,
            "silhouette": silhouette_score(X[sample], lab0[sample], metric="cosine"),
            "davies_bouldin": davies_bouldin_score(X, lab0),
            "stability_ari_mean": float(np.mean(pair_ari)),
            "stability_ari_std": float(np.std(pair_ari)),
            "largest_cluster_pct": 100 * sizes.max() / sizes.sum(),
            "smallest_cluster_pct": 100 * sizes.min() / sizes.sum(),
            "seconds": t.seconds,
        }
        for col in ext_cols(df):
            m = df[col].notna().values
            row[f"nmi_{col[4:]}"] = normalized_mutual_info_score(df.loc[m, col], lab0[m])
            row[f"ari_{col[4:]}"] = adjusted_rand_score(df.loc[m, col], lab0[m])
        rows.append(row)
        print({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in row.items()})

    out_dir = ROOT / cfg["output"]["summary_dir"] / cfg["experiment_name"]
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "cluster_sweep.csv", index=False)
    write_json({**run_context(cfg), "clustering_config": c, "svd_explained_variance": evr,
                "n_tickets": len(df), "results": rows}, out_dir / "cluster_sweep.json")
    return res


def top_terms(vec, X_tfidf, labels, k, n):
    terms = np.array(vec.get_feature_names_out())
    out = {}
    for j in range(k):
        m = labels == j
        mean = np.asarray(X_tfidf[m].mean(axis=0)).ravel()
        out[j] = terms[np.argsort(mean)[::-1][:n]].tolist()
    return out


def fit(cfg):
    c = cfg["clustering"]
    df = load_tickets(cfg)
    vec, svd, X_tfidf, X, evr = embed(cfg, df)
    ks = sorted(set(cfg["classification"]["k_sensitivity"]) | {c["selected_k"]})
    summary_dir = ROOT / cfg["output"]["summary_dir"] / cfg["experiment_name"]
    labels_dir = ROOT / "results" / "labels" / cfg["experiment_name"]
    labels_dir.mkdir(parents=True, exist_ok=True)
    for k in ks:
        km = KMeans(n_clusters=k, n_init=c["n_init"], random_state=0).fit(X)
        lab = km.labels_
        # Relabel clusters by size (0 = largest) so ids are stable and readable.
        order = np.argsort(-np.bincount(lab, minlength=k))
        remap = np.empty(k, dtype=int)
        remap[order] = np.arange(k)
        lab = remap[lab]
        pd.DataFrame({"ticket_id": df["ticket_id"], "cluster": lab}).to_csv(
            labels_dir / f"labels_k{k}.csv.gz", index=False)

        terms = top_terms(vec, X_tfidf, lab, k, c["top_terms"])
        info = []
        for j in range(k):
            m = lab == j
            item = {"cluster": j, "size": int(m.sum()), "pct": round(100 * m.mean(), 2),
                    "top_terms": terms[j],
                    "solution_missing_pct": round(100 * df.loc[m, "solution_missing"].mean(), 1)}
            for col in ext_cols(df):
                vc = df.loc[m, col].value_counts(normalize=True).head(3)
                item[f"top_{col[4:]}"] = {str(a): round(float(b), 3) for a, b in vc.items()}
            info.append(item)
        write_json({**run_context(cfg), "k": k, "clusters": info}, summary_dir / f"clusters_k{k}.json")

        names_path = ROOT / "configs" / f"cluster_names_k{k}.yaml"
        if not names_path.exists():
            stub = {j: " / ".join(terms[j][:3]) for j in range(k)}
            with open(names_path, "w", encoding="utf-8") as f:
                f.write(f"# Human-readable names for clusters at k={k} (ids ordered by size, 0 = largest).\n"
                        "# Auto-generated from top terms; edit the names, keep the ids.\n")
                yaml.safe_dump(stub, f, allow_unicode=True, sort_keys=True)

        if k == c["selected_k"]:
            (ROOT / "models").mkdir(exist_ok=True)
            joblib.dump({"vectorizer": vec, "svd": svd, "kmeans": km, "remap": remap},
                        ROOT / "models" / f"cluster_model_k{k}.joblib")
            write_audit_sample(cfg, df, lab, k)
        print(f"k={k}: sizes {np.bincount(lab).tolist()}")


def write_audit_sample(cfg, df, lab, k):
    """Random sample for two analysts to relabel (estimates pseudo-label noise)."""
    n = cfg["clustering"]["audit_sample_size"]
    s = df.assign(cluster=lab).sample(n=n, random_state=2026)
    names = yaml.safe_load(open(ROOT / "configs" / f"cluster_names_k{k}.yaml", encoding="utf-8"))
    audit = pd.DataFrame({
        "ticket_id": s["ticket_id"], "title": s["title_raw"], "solution": s["solution_raw"],
        "cluster_id": s["cluster"], "cluster_name": s["cluster"].map(names),
        "rater1_agrees(1/0)": "", "rater1_correct_cluster_id": "",
        "rater2_agrees(1/0)": "", "rater2_correct_cluster_id": "", "notes": "",
    })
    p = ROOT / "data" / "audit" / f"label_audit_k{k}.xlsx"
    p.parent.mkdir(parents=True, exist_ok=True)
    audit.to_excel(p, index=False)
    print(f"audit sample -> {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sweep", "fit"])
    ap.add_argument("--config", default="configs/phase1.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    (sweep if a.step == "sweep" else fit)(cfg)
