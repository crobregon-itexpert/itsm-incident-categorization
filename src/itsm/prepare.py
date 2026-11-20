"""Step 1: load the raw export, filter, clean text, cache. Logs row counts at every step.

    python -m itsm.prepare --config configs/phase1.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .runlog import ROOT, file_sha256, load_config, run_context, write_json
from .anonymize import doc_freq, mask_raw, mask_rare_clean, mask_rare_raw
from .text import TextNormalizer, apply_map, name_stopwords, plural_map, strip_solution_template


def load_raw(cfg: dict) -> pd.DataFrame:
    p = ROOT / cfg["data"]["raw_path"]
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p, sheet_name=cfg["data"].get("sheet", 0))
    return pd.read_csv(p, sep=None, engine="python", encoding="utf-8-sig")


def prepare(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    raw_path = ROOT / d["raw_path"]
    df = load_raw(cfg)
    counts = {"raw_rows": len(df)}

    df = df[df[d["type_col"]].isin(d["ticket_types"])].copy()
    counts["after_type_filter"] = len(df)

    names = name_stopwords(df, cfg["text"].get("name_cols"), cfg["text"].get("name_token_keep"))
    counts["name_tokens_removed_vocab"] = len(names)
    norm = TextNormalizer(cfg["text"], extra_stopwords=names)
    missing = {str(v).strip().lower() for v in d["solution_missing_values"]}

    out = pd.DataFrame({
        "ticket_id": df[d["id_col"]].astype(str).values,
        "ticket_type": df[d["type_col"]].values,
        "opened": pd.to_datetime(df.get("Fecha de apertura"), errors="coerce").values,
        "title_raw": df[d["title_col"]].fillna("").astype(str).values,
        "solution_raw": df[d["solution_col"]].fillna("").astype(str).values,
    })
    for c in d.get("external_label_cols", []):
        out["ext_" + c.lower().replace(" ", "_")] = df[c].astype("string").str.strip().values

    out["solution_missing"] = out["solution_raw"].str.strip().str.lower().isin(missing)
    out.loc[out["solution_missing"], "solution_raw"] = ""
    out["title"] = out["title_raw"].map(norm)
    tcfg = cfg["text"].get("solution_template")
    out["solution"] = out["solution_raw"].map(lambda v: norm(strip_solution_template(v, tcfg)))
    if cfg["text"].get("fold_plurals", True):
        pm = plural_map(pd.concat([out["title"], out["solution"]]))
        counts["plural_forms_folded"] = len(pm)
        out["title"] = out["title"].map(lambda v: apply_map(v, pm))
        out["solution"] = out["solution"].map(lambda v: apply_map(v, pm))
    extra = set(cfg["text"].get("domain_stopwords") or [])
    if extra:
        out["title"] = out["title"].map(lambda v: " ".join(t for t in v.split() if t not in extra))
        out["solution"] = out["solution"].map(lambda v: " ".join(t for t in v.split() if t not in extra))

    out = out[out["title"].str.len() > 0].copy()
    counts["after_empty_title_filter"] = len(out)
    out = out.drop_duplicates("ticket_id")
    counts["after_id_dedup"] = len(out)

    an = cfg["text"].get("anonymize")
    if an:
        # Same ticket population as before; only the text is masked (see itsm.anonymize).
        k = an["min_ticket_freq"]
        out["title_bert"] = out["title_raw"].map(lambda v: mask_raw(v, names))
        dfr = doc_freq(out["title_bert"])
        out["title_bert"] = out["title_bert"].map(lambda v: mask_rare_raw(v, dfr, k))
        # Classical input is derived from the anonymized title, so it can never contain
        # anything the shared title_bert does not. Same cleaning steps as above.
        norm_anon = TextNormalizer(cfg["text"], extra_stopwords=names | {"xxx"})
        t = out["title_bert"].map(norm_anon)
        if cfg["text"].get("fold_plurals", True):
            t = t.map(lambda v: apply_map(v, pm))
        if extra:
            t = t.map(lambda v: " ".join(w for w in v.split() if w not in extra))
        dfc = doc_freq(t)
        out["title"] = t.map(lambda v: mask_rare_clean(v, dfc, k))
        counts["anonymize_min_ticket_freq"] = k
        counts["titles_empty_after_masking"] = int((out["title"].str.len() == 0).sum())
        counts["title_bert_with_mask_pct"] = round(100 * out["title_bert"].str.contains("XXX").mean(), 2)

    # Group key: identical cleaned titles must never be split across train/test.
    out["group"] = out["title"]
    counts["unique_clean_titles"] = int(out["group"].nunique())
    counts["solution_missing"] = int(out["solution_missing"].sum())
    counts["solution_missing_pct"] = round(100 * out["solution_missing"].mean(), 2)
    counts["title_tokens_median"] = float(out["title"].str.split().str.len().median())

    path = ROOT / d["interim_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.reset_index(drop=True).to_pickle(path)

    write_json({
        **run_context(cfg),
        "raw_file": str(d["raw_path"]),
        "raw_sha256": file_sha256(raw_path),
        "ticket_types": d["ticket_types"],
        "counts": counts,
        "ticket_type_counts": out["ticket_type"].value_counts().to_dict(),
        "date_range": [str(out["opened"].min()), str(out["opened"].max())],
    }, ROOT / cfg["output"]["summary_dir"] / cfg["experiment_name"] / "data_statement.json")
    print(counts)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    prepare(load_config(ap.parse_args().config))
