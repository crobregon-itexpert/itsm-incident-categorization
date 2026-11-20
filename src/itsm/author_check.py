"""Author's blind check: 60 audit tickets (6 per category) and 8 agent cases judged by the LLM judge.

    python -m itsm.author_check data/audit/revision_autor_60_8_filled.xlsx
"""
from __future__ import annotations

import json
import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .runlog import ROOT, load_config, write_json

ALERTS = {1, 3, 9}   # automatic monitoring-alert categories


def main(path):
    cfg = load_config("configs/phase1_v2.yaml")
    a = pd.read_excel(path, sheet_name="Auditoria60")
    a["author"] = a.iloc[:, 3].astype(str).str.extract(r"^(\d+)")[0]
    a["note"] = a.iloc[:, 4] if a.shape[1] > 4 else None
    k = pd.read_csv(ROOT / "data/audit/revision_autor_60_key.csv")
    g = pd.DataFrame(json.load(open(ROOT / "results/agent_eval/gemini_results.json"))["audit"])
    m = a.merge(k, on="audit_id").merge(g[["audit_id", "best_category_id"]], on="audit_id")
    m = m[m.author.notna()].copy()
    m["author"] = m.author.astype(int)
    alert = lambda s: s.where(~s.isin(ALERTS), 99)
    non_alert = m[~m.cluster_label.isin(ALERTS)]
    out = {
        "n_labeled": len(m),
        "author_vs_cluster": {"agreement": float((m.author == m.cluster_label).mean()),
                              "kappa": float(cohen_kappa_score(m.author, m.cluster_label))},
        "author_vs_gemini": {"agreement": float((m.author == m.best_category_id).mean()),
                             "kappa": float(cohen_kappa_score(m.author, m.best_category_id))},
        "gemini_vs_cluster_same_tickets": float((m.best_category_id == m.cluster_label).mean()),
        "author_vs_cluster_by_category": m.assign(ok=m.author == m.cluster_label).groupby("cluster_label").ok.mean().round(3).to_dict(),
        "author_vs_cluster_non_alert_categories": float((non_alert.author == non_alert.cluster_label).mean()),
        "post_hoc_alerts_merged": {"agreement": float((alert(m.author) == alert(m.cluster_label)).mean()),
                                   "kappa": float(cohen_kappa_score(alert(m.author), alert(m.cluster_label)))},
        "confusion": pd.crosstab(m.cluster_label, m.author).to_dict(),
        "notes": m.loc[m.note.notna(), ["audit_id", "cluster_label", "author", "note"]].to_dict("records"),
    }
    j = pd.read_excel(path, sheet_name="Juez8").iloc[:, [0, 4, 5]]
    j.columns = ["case_id", "author_would_solve", "author_hallucination"]
    j["author_hallucination"] = j.author_hallucination.astype(str).str.replace("o", "0").astype(int)
    j = j.merge(pd.read_csv(ROOT / "data/audit/revision_juez8_key.csv"), on="case_id")
    out["judge_check"] = {
        "n": len(j),
        "author_would_solve": int(j.author_would_solve.sum()), "judge_would_solve": int(j.judge_would_solve.sum()),
        "agreement_would_solve": float((j.author_would_solve == j.judge_would_solve).mean()),
        "author_hallucination": int(j.author_hallucination.sum()), "judge_hallucination": int(j.judge_hallucination.sum()),
        "agreement_hallucination": float((j.author_hallucination == j.judge_hallucination).mean()),
        "cases": j.to_dict("records"),
    }
    write_json(out, ROOT / cfg["output"]["summary_dir"] / cfg["experiment_name"] / "author_check.json")
    print(json.dumps({k: v for k, v in out.items() if k not in ("confusion", "cases")}, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
