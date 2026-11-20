"""Step 3: fixed train/val/test splits, one per seed, saved to disk.

    python -m itsm.splits --config configs/phase1.yaml

Tickets with the same cleaned title share a group and always fall in the same
split, so the test set never contains a title seen in training. Splits are
stratified on the selected-k cluster labels and reused for every k, model and
target, so all results in the paper are evaluated on identical test sets.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .runlog import ROOT, load_config, run_context, write_json


def labels_path(cfg, k):
    # Labels can come from an earlier experiment (fixed taxonomy): clustering.labels_experiment
    exp = cfg["clustering"].get("labels_experiment") or cfg["experiment_name"]
    return ROOT / "results" / "labels" / exp / f"labels_k{k}.csv.gz"


def splits_dir(cfg):
    return ROOT / "results" / "splits" / cfg["experiment_name"]


def group_split(df, test_size, val_size, seed) -> np.ndarray:
    """Assign whole title groups to train/val/test.

    Groups are stratified by their majority label and shuffled with `seed`. Within each
    label, groups are visited in shuffled order and each group goes to the split that is
    furthest below its ticket-count target for that label.
    This keeps every split close to its share per label even with very large groups, and
    gives a genuinely different partition for every seed.
    """
    rng = np.random.RandomState(seed)
    targets = {"test": test_size, "val": val_size, "train": 1 - test_size - val_size}
    grp = df.groupby("group").agg(size=("cluster", "size"),
                                  label=("cluster", lambda v: v.value_counts().index[0])).reset_index()
    assign = {}
    for _, gl in grp.groupby("label"):
        gl = gl.sample(frac=1.0, random_state=rng)
        total = gl["size"].sum()
        filled = {k: 0 for k in targets}
        for g, n in zip(gl["group"], gl["size"]):
            # deficit relative to target after adding this group
            best = max(targets, key=lambda k: targets[k] * total - filled[k] - n / 2)
            assign[g] = best
            filled[best] += n
    return df["group"].map(assign).values


def random_split(y, test_size, val_size, seed) -> np.ndarray:
    """Stratified random split at ticket level (identical titles may appear in train and test,
    as they do in production; generalization is measured separately on unseen titles)."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    rest, test = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    train, val = train_test_split(rest, test_size=val_size / (1 - test_size), stratify=y[rest],
                                  random_state=seed + 1000)
    split = np.full(len(y), "train", dtype=object)
    split[test], split[val] = "test", "val"
    return split


def make_splits(cfg):
    s = cfg["split"]
    df = pd.read_pickle(ROOT / cfg["data"]["interim_path"])[["ticket_id", "group"]]
    lab = pd.read_csv(labels_path(cfg, cfg["clustering"]["selected_k"]), dtype={"ticket_id": str})
    df = df.merge(lab, on="ticket_id", validate="one_to_one")
    out_dir = splits_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    method = s.get("method", "group")
    for seed in s["seeds"]:
        if method == "stratified_random":
            split = random_split(df["cluster"].values, s["test_size"], s["val_size"], seed)
        else:
            split = group_split(df, s["test_size"], s["val_size"], seed)
        out = pd.DataFrame({"ticket_id": df["ticket_id"], "split": split})
        out.to_csv(out_dir / f"seed{seed}.csv.gz", index=False)
        summary[seed] = out["split"].value_counts().to_dict()
        if method == "group":
            overlap = df.assign(split=split).groupby("group")["split"].nunique().gt(1).sum()
            assert overlap == 0, f"{overlap} title groups leak across splits"
        else:
            tr = set(df["group"][split == "train"])
            te = df["group"][split == "test"]
            summary[f"{seed}_test_unseen_title_tickets"] = int((~te.isin(tr)).sum())
            summary[f"{seed}_test_unseen_titles"] = int(te[~te.isin(tr)].nunique())
        print(seed, summary[seed])
    write_json({**run_context(cfg), "split_config": s, "method": method, "counts": summary},
               out_dir / "splits.json")


def load_split(cfg, seed) -> pd.DataFrame:
    return pd.read_csv(splits_dir(cfg) / f"seed{seed}.csv.gz", dtype={"ticket_id": str})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    make_splits(load_config(ap.parse_args().config))
