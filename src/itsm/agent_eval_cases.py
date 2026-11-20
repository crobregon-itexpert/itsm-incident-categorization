"""Cases for the automated 35-case agent evaluation (Section VI), anonymized.

    python -m itsm.agent_eval_cases --config configs/phase1_v2.yaml

Reads data/audit/agent_eval_key.csv (30 seed-0 test incidents, 3 per category, + 5 out-of-scope requests,
fixed on 2026-11-02) and writes data/agent/agent_eval_cases.json (local only; uploaded to the private
bucket). Titles are the shared anonymized titles (title_bert); resolution notes are anonymized with the
same rules as the knowledge base. Out-of-scope texts are taken verbatim from the evaluation workbook.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from .anonymize import doc_freq, mask_raw, mask_rare_raw
from .prepare import load_raw
from .runlog import ROOT, load_config
from .text import _RN_GLUED, name_stopwords, strip_solution_template


def _fix(s):
    """Repair UTF-8 read as cp1252 in the export ("â€“" -> "–")."""
    if isinstance(s, str) and "â€" in s:
        try:
            return s.encode("cp1252").decode("utf-8")
        except UnicodeError:
            return s.replace("â€“", "–").replace("â€”", "—")
    return s


def build(cfg):
    key = pd.read_csv(ROOT / "data/audit/agent_eval_key.csv")
    wb = pd.read_excel(ROOT / "data/audit/evaluacion_agente_35casos.xlsx", sheet_name="Casos")
    sh = pd.read_csv(ROOT / cfg["data"]["shared_csv"], keep_default_na=False)
    idmap = pd.read_csv(ROOT / "data/share/private_id_map.csv", dtype={"ticket_id": str})
    t = pd.read_pickle(ROOT / cfg["data"]["interim_path"])
    d = cfg["data"]
    raw = load_raw(cfg)
    raw = raw[raw[d["type_col"]].isin(d["ticket_types"])]
    names = name_stopwords(raw, cfg["text"].get("name_cols"), cfg["text"].get("name_token_keep"))
    tcfg = cfg["text"].get("solution_template")
    sol_all = t["solution_raw"].map(lambda v: mask_raw(_RN_GLUED.sub(" ", strip_solution_template(v, tcfg)), names))
    dfs = doc_freq(sol_all)
    t = t.assign(sol_anon=sol_all.map(lambda v: mask_rare_raw(v, dfs, cfg["text"]["anonymize"]["min_ticket_freq"])))
    cases = []
    for r in key.itertuples():
        if r.category_id < 0:
            txt = wb.loc[wb["Caso"] == r.case_id, "Reporte del usuario (escribir al agente)"].iloc[0]
            cases.append({"case_id": r.case_id, "oos": True, "title": txt, "resolution": None,
                          "category_id": None, "unseen": None})
            continue
        s = sh[sh.tid == int(r.tid)].iloc[0]
        assert s["split_seed0"] == "test", r.case_id          # never in the knowledge base
        tk = idmap[idmap.tid == int(r.tid)].ticket_id.iloc[0]
        cases.append({"case_id": r.case_id, "oos": False, "title": _fix(s["title_bert"]),
                      "resolution": t.loc[t.ticket_id == tk, "sol_anon"].iloc[0],
                      "category_id": int(r.category_id), "unseen": bool(r.unseen)})
    out = ROOT / "data/agent/agent_eval_cases.json"
    out.write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out, len(cases))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1_v2.yaml")
    build(load_config(ap.parse_args().config))
