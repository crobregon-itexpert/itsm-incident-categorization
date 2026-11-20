"""Inputs for the external Gemini baseline and the LLM label audit (anonymized; local only).

    python -m itsm.gemini_inputs

Writes data/agent/gemini_inputs.json:
  categories      ids, Spanish names and definitions (configs/categories.yaml)
  fewshot         3 training titles per category (seed-0 train split, random_state=0)
  baseline        1,000 seed-0 test tickets, stratified by category (random_state=0): tid, title, y_true
  audit           the 300 audit tickets fixed on 2026-11-02 (audit_key_k10.csv): anonymized title and
                  resolution note, cluster label
Requires the private data; the input file itself is not released.
"""
from __future__ import annotations

import json

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from .anonymize import doc_freq, mask_raw, mask_rare_raw
from .prepare import load_raw
from .runlog import ROOT, load_config
from .text import _RN_GLUED, name_stopwords, strip_solution_template


def main(cfg):
    sh = pd.read_csv(ROOT / cfg["data"]["shared_csv"], keep_default_na=False)
    cats = yaml.safe_load(open(ROOT / "configs/categories.yaml", encoding="utf-8"))["categories"]
    tr = sh[sh.split_seed0 == "train"]
    te = sh[sh.split_seed0 == "test"]
    few = tr.groupby("category_id").sample(3, random_state=0)
    base, _ = train_test_split(te, train_size=1000, stratify=te.category_id, random_state=0)

    d = cfg["data"]
    raw = load_raw(cfg)
    raw = raw[raw[d["type_col"]].isin(d["ticket_types"])]
    names = name_stopwords(raw, cfg["text"].get("name_cols"), cfg["text"].get("name_token_keep"))
    t = pd.read_pickle(ROOT / cfg["data"]["interim_path"])
    tcfg = cfg["text"].get("solution_template")
    clean = lambda v: mask_raw(_RN_GLUED.sub(" ", strip_solution_template(v, tcfg)), names)
    dfs = doc_freq(t["solution_raw"].map(clean))
    key = pd.read_csv(ROOT / "data/audit/audit_key_k10.csv")
    audit = [{"audit_id": r.audit_id, "title": r.title_bert,
              "resolution": mask_rare_raw(clean(r.solution_raw), dfs, cfg["text"]["anonymize"]["min_ticket_freq"]),
              "cluster_label": int(r.cluster_label)} for r in key.itertuples()]
    out = {
        "categories": [{"id": i, "name": cats[i]["name_es"], "definition": cats[i].get("definition_es", "")} for i in range(10)],
        "fewshot": [{"title": r.title_bert, "category_id": int(r.category_id)} for r in few.itertuples()],
        "baseline": [{"tid": int(r.tid), "title": r.title_bert, "y_true": int(r.category_id)} for r in base.itertuples()],
        "audit": audit,
    }
    p = ROOT / "data/agent/gemini_inputs.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    print(p, {k: len(v) for k, v in out.items()}, round(p.stat().st_size / 1024), "KB")


if __name__ == "__main__":
    main(load_config("configs/phase1_v2.yaml"))
