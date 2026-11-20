"""External Gemini baseline and LLM label audit: metrics, table and summary.

    python -m itsm.gemini_report results/agent_eval/gemini_results.json

Compares Gemini zero/few-shot with BETO (tuned) and the MLP on the same 1,000 seed-0 test tickets
(stratified sample from itsm.gemini_inputs), on all of them and on those whose cleaned title never
appears in training. Audit: share of the 300 cluster labels Gemini judges correct, and agreement
(accuracy, Cohen's kappa) between Gemini's preferred category and the cluster label.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from .report import latex_table
from .runlog import ROOT, load_config, write_json

RUN = ROOT / "results/runs/phase1_v2/cluster__title__k10"


def main(path):
    cfg = load_config("configs/phase1_v2.yaml")
    G = json.load(open(path, encoding="utf-8"))
    D = json.load(open(ROOT / "data/agent/gemini_inputs.json", encoding="utf-8"))
    sh = pd.read_csv(ROOT / cfg["data"]["shared_csv"], keep_default_na=False)
    idmap = pd.read_csv(ROOT / "data/share/private_id_map.csv", dtype={"ticket_id": str})
    base = pd.DataFrame(D["baseline"]).merge(sh[["tid", "title_clean"]], on="tid").merge(idmap, on="tid")
    train_titles = set(sh.loc[sh.split_seed0 == "train", "title_clean"])
    base["unseen"] = ~base.title_clean.isin(train_titles)
    for m in ("zeroshot", "fewshot"):
        base[f"gemini_{m}"] = base.tid.astype(str).map(G[m]).fillna(-1).astype(int)
    for name, d in (("beto", "bert_beto_tuned"), ("mlp", "mlp")):
        # in the v2 runs, the predictions' "ticket_id" column holds the shared pseudonymous tid
        p = pd.read_csv(RUN / d / "seed0" / "predictions_test.csv.gz").rename(columns={"ticket_id": "tid", "y_pred": name})
        base = base.merge(p[["tid", name]], on="tid", how="left")
        assert base[name].notna().all(), name
    labels = list(range(10))
    rows, summ = [], {"n": len(base), "n_unseen": int(base.unseen.sum()), "model": G["model"],
                      "missing": {m: int((base[f"gemini_{m}"] < 0).sum()) for m in ("zeroshot", "fewshot")}}
    names = [("gemini_zeroshot", f"Gemini zero-shot"), ("gemini_fewshot", "Gemini few-shot (3/class)"),
             ("mlp", "MLP (TF-IDF)"), ("beto", "BETO (tuned)")]
    for col, label in names:
        r = {}
        for view, sub in (("all", base), ("unseen", base[base.unseen])):
            r[f"{view}_acc"] = float(accuracy_score(sub.y_true, sub[col]))
            r[f"{view}_f1"] = float(f1_score(sub.y_true, sub[col], labels=labels, average="macro", zero_division=0))
        summ[col] = r
        rows.append([label, f"{100 * r['all_acc']:.1f}", f"{r['all_f1']:.3f}", f"{100 * r['unseen_acc']:.1f}", f"{r['unseen_f1']:.3f}"])

    A = pd.DataFrame(G["audit"]).merge(pd.DataFrame(D["audit"])[["audit_id", "cluster_label"]], on="audit_id")
    summ["audit"] = {
        "n": len(A),
        "label_correct": A.label_correct.value_counts(normalize=True).round(4).to_dict(),
        "agreement_best_vs_cluster": float((A.best_category_id == A.cluster_label).mean()),
        "kappa_best_vs_cluster": float(cohen_kappa_score(A.cluster_label, A.best_category_id)),
        "label_correct_by_category": A.assign(ok=A.label_correct == "si").groupby("cluster_label").ok.mean().round(3).to_dict(),
    }
    write_json(summ, ROOT / cfg["output"]["summary_dir"] / cfg["experiment_name"] / "gemini_baseline.json")
    latex_table(ROOT / cfg["output"]["tables_dir"] / "table_gemini_baseline.tex",
                f"External LLM baseline ({G['model']}, temperature 0, no thinking) against the fine-tuned and TF-IDF "
                f"models on the same {len(base)} seed-0 test tickets (stratified sample; {int(base.unseen.sum())} with "
                f"titles unseen in training).",
                "tab:gemini", ["Model", "Acc. (\\%)", "Macro-F1", "Unseen acc. (\\%)", "Unseen macro-F1"], rows, "lcccc")
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
