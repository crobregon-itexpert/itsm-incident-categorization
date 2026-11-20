"""Offline evaluation of the retrieval step of the agent (H1: category filter vs no filter).

    python -m itsm.retrieval_eval --config configs/phase1_v2.yaml --seed 0

Knowledge base = training-split incidents of the given seed with a recorded resolution note
(duplicates of title + note removed); queries = test-split incidents with a recorded note.
Retriever = TF-IDF cosine on the cleaned anonymized title (a transparent proxy for the agent's
embedding search). For each query the top-5 cases are retrieved (a) without a filter, (b) filtered
to the category predicted by fine-tuned BETO, (c) filtered to the true category (oracle).
(d) filtered to the predicted category only when its probability is >= 0.8.
Metrics: category hit@5 (a retrieved case has the query's true category) and resolution
similarity (max cosine between the query's actual resolution note and the retrieved notes; TF-IDF
fitted on the knowledge base). Reported for all queries and for unseen titles.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .runlog import ROOT, load_config, run_context, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1_v2.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    cfg = load_config(a.config)
    t = pd.read_pickle(ROOT / cfg["data"]["interim_path"])
    idmap = pd.read_csv(ROOT / "data/share/private_id_map.csv", dtype={"ticket_id": str})
    sh = pd.read_csv(ROOT / cfg["data"]["shared_csv"], keep_default_na=False)
    df = sh[["tid", "category_id", f"split_seed{a.seed}"]].merge(idmap, on="tid").merge(
        t[["ticket_id", "title", "solution", "solution_missing"]], on="ticket_id")
    pred = pd.read_csv(ROOT / f"results/runs/{cfg['experiment_name']}/cluster__title__k10/bert_beto_tuned/seed{a.seed}/predictions_test.csv.gz")
    pred["tid"] = pred["ticket_id"].astype(int)
    pred["conf"] = pred[[f"p{i}" for i in range(10)]].max(axis=1)
    df = df.merge(pred[["tid", "y_pred", "conf"]], on="tid", how="left")
    split = df[f"split_seed{a.seed}"]
    kb = df[(split == "train") & ~df.solution_missing & (df.solution.str.len() > 0)].drop_duplicates(["title", "solution"])
    q = df[(split == "test") & ~df.solution_missing & (df.solution.str.len() > 0)].copy()
    seen = set(df.loc[split == "train", "title"])
    q["unseen"] = ~q.title.isin(seen)

    tv = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit(kb.title)
    sv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(kb.solution)
    Kt, Ks = tv.transform(kb.title), sv.transform(kb.solution)
    kb_cat = kb.category_id.values
    rows = []
    B = 2000
    # Many knowledge-base cases share an identical templated title (equal scores). Ties are broken
    # by a fixed random jitter so the top-k does not depend on file order.
    jitter = np.random.RandomState(0).uniform(0, 1e-6, size=len(kb))
    for s in range(0, len(q), B):
        qb = q.iloc[s:s + B]
        S = (tv.transform(qb.title) @ Kt.T).toarray() + jitter
        Qs = sv.transform(qb.solution)
        for i, r in enumerate(qb.itertuples()):
            out = {"tid": r.tid, "unseen": r.unseen}
            conf_cat = r.y_pred if r.conf >= 0.8 else None      # filter only when the classifier is confident
            for mode, cat in [("none", None), ("predicted", r.y_pred), ("predicted_conf0.8", conf_cat),
                              ("oracle", r.category_id)]:
                sc = S[i].copy()
                if cat is not None:
                    sc[kb_cat != cat] = -1
                top = np.argpartition(-sc, a.k)[:a.k]
                out[f"hit_{mode}"] = bool((kb_cat[top] == r.category_id).any())
                out[f"res_sim_{mode}"] = float((Qs[i] @ Ks[top].T).toarray().max())
            rows.append(out)
    res = pd.DataFrame(rows)
    summ = {}
    for view, m in [("all", np.ones(len(res), bool)), ("unseen_titles", res.unseen.values)]:
        summ[view] = {"n_queries": int(m.sum())}
        for mode in ["none", "predicted", "predicted_conf0.8", "oracle"]:
            summ[view][f"category_hit@{a.k}_{mode}"] = float(res.loc[m, f"hit_{mode}"].mean())
            summ[view][f"resolution_similarity_{mode}"] = float(res.loc[m, f"res_sim_{mode}"].mean())
    out_dir = ROOT / cfg["output"]["summary_dir"] / cfg["experiment_name"]
    write_json({**run_context(cfg), "seed": a.seed, "k": a.k, "kb_size": len(kb), "summary": summ},
               out_dir / f"retrieval_eval_seed{a.seed}.json")
    print(len(kb), "KB cases;", summ)




def table(cfg, seeds=(0, 1, 2, 3, 4), k=5):
    """paper/tables/table_retrieval.tex from the per-seed JSON files."""
    from .report import latex_table
    out_dir = ROOT / cfg["output"]["summary_dir"] / cfg["experiment_name"]
    R = [__import__("json").loads((out_dir / f"retrieval_eval_seed{s}.json").read_text())["summary"] for s in seeds]
    names = {"none": "No filter", "predicted": "Predicted category (always)",
             "predicted_conf0.8": "Predicted category if $p\\ge0.8$", "oracle": "True category (oracle)"}
    rows = []
    for m, label in names.items():
        row = [label]
        for v in ("all", "unseen_titles"):
            h = [r[v][f"category_hit@{k}_{m}"] for r in R]
            q = [r[v][f"resolution_similarity_{m}"] for r in R]
            row += [f"{np.mean(h):.3f} $\\pm$ {np.std(h, ddof=1):.3f}", f"{np.mean(q):.3f} $\\pm$ {np.std(q, ddof=1):.3f}"]
        rows.append(row)
    n_all = int(np.mean([r["all"]["n_queries"] for r in R])); n_un = int(np.mean([r["unseen_titles"]["n_queries"] for r in R]))
    latex_table(ROOT / cfg["output"]["tables_dir"] / "table_retrieval.tex",
                f"Offline evaluation of the retrieval step (top-{k} similar resolved incidents from the training split; "
                f"TF-IDF retriever as a proxy for the agent's search). Hit: a retrieved case has the query's true category. "
                f"Resolution similarity: max cosine between the query's actual resolution note and the retrieved notes. "
                f"Mean $\\pm$ std over 5 splits; about {n_all:,} queries (all) and {n_un:,} (unseen titles) per split.",
                "tab:retrieval", ["Retrieval filter", "Hit@5 (all)", "Res. sim. (all)", "Hit@5 (unseen)", "Res. sim. (unseen)"],
                rows, "lcccc", wide=True)


if __name__ == "__main__":
    import sys
    if "--table" in sys.argv:
        table(load_config("configs/phase1_v2.yaml"))
    else:
        main()
