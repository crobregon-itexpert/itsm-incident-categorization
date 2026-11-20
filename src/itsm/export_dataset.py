"""Export the anonymized, minimal dataset that is shared with the code.

    python -m itsm.export_dataset --config configs/phase1_v2.yaml

Output: data/share/itsm_incidents_2024_anon.csv (UTF-8), one row per incident:
  tid              pseudonymous id (1..N, in order of the original ticket id)
  title_bert       anonymized title in its original form (input for BERT)
  title_clean      cleaned, anonymized title tokens (input for TF-IDF models; also the title group)
  category_id      static category 0-9 (configs/categories.yaml)
  category_en      category name (paper)
  category_es      category name (interface)
  resolver_group   human resolver group (Grupo Asignado), for the routing experiment
  split_seed0..4   train / val / test for each of the 5 seeds

No other field leaves the organization. The map tid -> original ticket id is written to
data/share/private_id_map.csv, which must never be shared.
"""
from __future__ import annotations

import argparse

import pandas as pd
import yaml

from .runlog import ROOT, file_sha256, load_config, run_context, write_json
from .splits import labels_path, splits_dir

SHARE_NAME = "itsm_incidents_2024_anon.csv"


def export(cfg):
    if "title_bert" not in pd.read_pickle(ROOT / cfg["data"]["interim_path"]).columns:
        raise SystemExit("run itsm.prepare with text.anonymize enabled first")
    df = pd.read_pickle(ROOT / cfg["data"]["interim_path"])
    k = cfg["clustering"]["selected_k"]
    lab = pd.read_csv(labels_path(cfg, k), dtype={"ticket_id": str})
    df = df.merge(lab, on="ticket_id", validate="one_to_one")
    cats = yaml.safe_load(open(ROOT / "configs" / "categories.yaml", encoding="utf-8"))["categories"]
    df = df.sort_values("ticket_id", key=lambda s: s.astype(int)).reset_index(drop=True)
    df["tid"] = range(1, len(df) + 1)
    out = pd.DataFrame({
        "tid": df["tid"],
        "title_bert": df["title_bert"],
        "title_clean": df["title"],
        "category_id": df["cluster"].astype(int),
        "category_en": df["cluster"].map(lambda c: cats[int(c)]["name_en"]),
        "category_es": df["cluster"].map(lambda c: cats[int(c)]["name_es"]),
        "resolver_group": df["ext_grupo_asignado"].fillna(""),
    })
    for seed in cfg["split"]["seeds"]:
        sp = pd.read_csv(splits_dir(cfg) / f"seed{seed}.csv.gz", dtype={"ticket_id": str})
        out[f"split_seed{seed}"] = df[["ticket_id"]].merge(sp, on="ticket_id", how="left")["split"].values
    assert out.filter(like="split_seed").notna().all().all()

    d = ROOT / "data" / "share"
    d.mkdir(parents=True, exist_ok=True)
    path = d / SHARE_NAME
    out.to_csv(path, index=False, encoding="utf-8")
    df[["tid", "ticket_id"]].to_csv(d / "private_id_map.csv", index=False)
    meta = {**run_context(cfg), "file": SHARE_NAME, "sha256": file_sha256(path), "rows": len(out),
            "columns": list(out.columns), "category_counts": out["category_id"].value_counts().sort_index().to_dict(),
            "anonymization": cfg["text"].get("anonymize"), "categories_status": "see configs/categories.yaml"}
    write_json(meta, ROOT / cfg["output"]["summary_dir"] / cfg["experiment_name"] / "shared_dataset.json")
    print(path, len(out), meta["sha256"][:12])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1_v2.yaml")
    export(load_config(ap.parse_args().config))
