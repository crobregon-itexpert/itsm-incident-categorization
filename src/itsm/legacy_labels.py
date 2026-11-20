"""Reproduce the categories of the submitted manuscript (Fig. 1) exactly.

    python -m itsm.legacy_labels --config configs/phase1.yaml

Faithful port of legacy/notebooks/AAI800_NLP_v5.ipynb, cells 2-6 (Colab, pandas 2):
incidents only, titles containing "solicitud"/"consulta" dropped, channel phrases removed,
TF-IDF (1-2 grams, 30k features, min_df 3, max_df 0.8) of title + solution, K-means k=10,
random_state=42. Output: 57,249 tickets with cluster sizes
[5624, 4331, 1675, 677, 12294, 5597, 25164, 445, 283, 1159] (asserted).

Used only to compare the submitted taxonomy with the revised one; the revision trains on
results/labels/phase1_v1.
"""
from __future__ import annotations

import argparse
import re
import unicodedata

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from .prepare import load_raw
from .runlog import ROOT, load_config, run_context, write_json

NAMES = {
    0: "POS Device / Interface Issues", 1: "Data Entry Issues / Misreported Tickets",
    2: "PixelPoint / App Startup Issues", 3: "Boleta Printing Failures",
    4: "Application Startup (Server / App Closed)", 5: "General App / Comandas Issues",
    6: "Operational / Informational Notices", 7: "Open Orders / Account Handling",
    8: "SAP Access / User Blocked", 9: "General Printing Issues (Boletas / Comandas)",
}
EXPECTED = [5624, 4331, 1675, 677, 12294, 5597, 25164, 445, 283, 1159]


def _s(series):
    # pandas 2 `astype(str)` turned NaN into the string "nan"; keep that behaviour
    return series.map(str)


def _col(c):
    c = unicodedata.normalize("NFKD", c.lower()).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", c).strip("_")


def _norm(s):
    s = unicodedata.normalize("NFKD", s.lower()).encode("ascii", "ignore").decode("utf-8", "ignore")
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", s)
    s = re.sub(r"\b(id|ticket|case|incidente|folio)\s*[:#]?\s*\d+\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\b\d{6,}\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def legacy_labels(cfg):
    df = load_raw(cfg)
    df.columns = [_col(c) for c in df.columns]
    df = df[df["tipo"].str.lower() == "incidente"].copy()
    df = df[~_s(df["titulo"]).str.contains("solicitud|consulta", case=False, na=False)].copy()
    noise = ["informativo", "escalado", "ticket escalado", "soporte remoto", "soporte tecnico"]

    def clean_noise(t):
        t = t.lower()
        for p in noise:
            t = re.sub(rf"\b{re.escape(p)}\b", "", t, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", t).strip()

    df["titulo"] = _s(df["titulo"]).apply(clean_noise)
    df = df[df["titulo"].str.strip() != ""].copy()
    df["clean_text"] = (_s(df["titulo"]) + " " + _s(df["solucion"])).str.strip().apply(_norm)
    df = df[df["clean_text"].str.strip() != ""]
    X = TfidfVectorizer(max_features=30000, min_df=3, max_df=0.8, ngram_range=(1, 2),
                        strip_accents="unicode", sublinear_tf=True).fit_transform(df["clean_text"])
    df["cluster"] = KMeans(n_clusters=10, random_state=42, n_init=10).fit_predict(X)
    sizes = df["cluster"].value_counts().sort_index().tolist()
    assert sizes == EXPECTED, f"does not reproduce the manuscript: {sizes}"

    out_dir = ROOT / "results" / "labels" / "legacy_v0"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticket_id": df["id"].astype(str), "cluster": df["cluster"],
                  "category": df["cluster"].map(NAMES)}).to_csv(out_dir / "labels_k10.csv.gz", index=False)
    write_json({**run_context(cfg), "source": "legacy/notebooks/AAI800_NLP_v5.ipynb cells 2-6",
                "n_tickets": len(df), "cluster_sizes": sizes, "names": NAMES}, out_dir / "labels_k10.json")
    print(len(df), sizes, "reproduced")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    legacy_labels(load_config(ap.parse_args().config))
