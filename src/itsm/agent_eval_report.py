"""Summarize the automated 35-case agent evaluation (simulated user + LLM judge).

    python -m itsm.agent_eval_report results/agent_eval/results_<stamp>.jsonl

Input: results.jsonl from the agent-eval workflow (techteampeer/ServiceDesk-Demo). Output:
results/summary/<exp>/agent_eval.json and paper/tables/table_agent_eval.tex.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from .report import latex_table
from .runlog import ROOT, load_config, write_json


def summarize(rows):
    ins = [r for r in rows if not r.get("oos")]
    oos = [r for r in rows if r.get("oos")]
    prop = [r for r in ins if r.get("proposal")]
    pct = lambda xs: float(np.mean(xs)) if xs else float("nan")
    def block(rs):
        p = [r for r in rs if r.get("proposal")]
        return {
            "n": len(rs),
            "classifier_accuracy": pct([bool(r.get("clf_correct")) for r in rs]),
            "confident_share": pct([bool(r.get("clf_confident")) for r in rs]),
            "confident_accuracy": pct([bool(r.get("clf_correct")) for r in rs if r.get("clf_confident")]),
            "wrongly_declined": int(sum(r.get("outcome") == "out_of_scope" for r in rs)),
            "request_or_informative_tickets": int(sum(str(r.get("outcome", "")).startswith(("request", "informative")) for r in rs)),
            "proposal_share": pct([bool(r.get("proposal")) for r in rs]),
            "would_solve_of_proposals": pct([r["judge"]["would_solve"] == 1 for r in p]),
            "would_solve_of_all": pct([bool(r.get("judge")) and r["judge"]["would_solve"] == 1 for r in rs]),
            "grounded_of_proposals": pct([r["judge"]["grounded"] == 1 for r in p]),
            "hallucination_of_proposals": pct([r["judge"]["hallucination"] == 1 for r in p]),
            "ticket_when_not_solved": pct([bool(r.get("ticket")) for r in rs if r.get("outcome") != "resolved_by_agent"]),
            "questions_mean": pct([r.get("questions", 0) for r in rs]),
            "seconds_to_proposal_median": float(np.median([r["seconds_to_proposal"] for r in rs if r.get("seconds_to_proposal") is not None])) if rs else None,
            "classifier_roundtrip_ms_median": float(np.median([r["classifier_roundtrip_ms"] for r in rs if r.get("classifier_roundtrip_ms")])) if any(r.get("classifier_roundtrip_ms") for r in rs) else None,
        }
    return {
        "in_scope": block(ins),
        "in_scope_unseen": block([r for r in ins if r.get("unseen")]),
        "in_scope_seen": block([r for r in ins if r.get("unseen") is False]),
        "out_of_scope": {"n": len(oos), "correctly_declined": int(sum(r.get("outcome") == "out_of_scope" for r in oos)),
                         "tickets_created": int(sum(bool(r.get("ticket")) for r in oos))},
        "errors": [r["case_id"] for r in rows if r.get("error") or r.get("outcome") in (None, "incomplete")],
        "judge_model": sorted({r.get("judge_model") for r in rows if r.get("judge")}),
        "user_model": sorted({r.get("user_model") for r in rows if r.get("user_model")}),
    }


def main(path):
    cfg = load_config("configs/phase1_v2.yaml")
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    s = summarize(rows)
    write_json({"source": str(path), "summary": s}, ROOT / cfg["output"]["summary_dir"] / cfg["experiment_name"] / f"agent_eval{tag}.json")
    f = lambda v: "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100 * v:.0f}\\%"
    cols = [("in_scope", "All in-scope"), ("in_scope_seen", "Seen wording"), ("in_scope_unseen", "Unseen wording")]
    metrics = [("classifier_accuracy", "Classifier category correct"), ("confident_share", "Classifier confident ($p\\ge0.8$)"),
               ("proposal_share", "Agent proposed steps"), ("would_solve_of_proposals", "Steps would solve (of proposals)"),
               ("would_solve_of_all", "Resolved at first contact (of all)"), ("grounded_of_proposals", "Steps grounded in retrieved cases"),
               ("hallucination_of_proposals", "Hallucinated steps (of proposals)"), ("ticket_when_not_solved", "Ticket created when not solved")]
    rows_t = [["Cases (n)"] + [str(s[c]["n"]) for c, _ in cols]]
    rows_t += [[label] + [f(s[c][m]) for c, _ in cols] for m, label in metrics]
    rows_t.append(["Wrongly declined as not IT (n)"] + [str(s[c]["wrongly_declined"]) for c, _ in cols])
    rows_t.append(["Registered as request/notice (n)"] + [str(s[c]["request_or_informative_tickets"]) for c, _ in cols])
    rows_t.append(["Questions asked (mean)"] + [f"{s[c]['questions_mean']:.1f}" for c, _ in cols])
    o = s["out_of_scope"]
    latex_table(ROOT / cfg["output"]["tables_dir"] / f"table_agent_eval{tag}.tex",
                f"Automated evaluation of the deployed agent on 30 held-out test incidents and {o['n']} out-of-scope requests "
                f"(simulated store user: {', '.join(s['user_model'])}; judge: {', '.join(s['judge_model'])}). "
                f"Out-of-scope requests correctly declined: {o['correctly_declined']}/{o['n']}, tickets created for them: {o['tickets_created']}.",
                f"tab:agent_eval{tag}", ["Metric"] + [c for _, c in cols], rows_t, "lccc")
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
