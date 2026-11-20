"""Shared metrics and run writer used by both classical and BERT training."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)

from .runlog import ROOT, run_context, write_json


def metrics(y_true, y_pred, labels) -> dict:
    # Classes absent from both truth and prediction in this split carry no information;
    # counting them as F1 = 0 would penalize a split for not containing a rare class.
    present = set(np.unique(y_true)) | set(np.unique(y_pred))
    labels = [l for l in labels if l in present]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "n": int(len(y_true)),
    }


def run_dir(cfg, setting: str, model: str, seed: int) -> Path:
    return ROOT / cfg["output"]["runs_dir"] / cfg["experiment_name"] / setting / model / f"seed{seed}"


def setting_name(target: str, inputs: list[str], k: int | None) -> str:
    s = f"{target}__{'+'.join(inputs)}"
    return s + (f"__k{k}" if target == "cluster" else "")


def unique_title_metrics(groups, y_true, y_pred, labels) -> dict:
    """Each distinct (cleaned title, true label) counts once; prediction = majority within it.

    Removes the weight of very frequent templated titles (e.g. monitoring alerts), so the
    score reflects how well the model handles the variety of ticket wording.
    """
    d = pd.DataFrame({"g": groups, "t": y_true, "p": y_pred})
    u = d.groupby(["g", "t"])["p"].agg(lambda v: v.value_counts().index[0]).reset_index()
    out = metrics(u["t"].values, u["p"].values, labels)
    return out


def write_run(cfg, *, setting, model, seed, labels, label_names, ids_test, y_test, p_test,
              y_val, p_val, extra: dict, groups_test=None, groups_train=None) -> dict:
    d = run_dir(cfg, setting, model, seed)
    d.mkdir(parents=True, exist_ok=True)
    m_test = metrics(y_test, p_test, labels)
    m_val = metrics(y_val, p_val, labels)
    rep = classification_report(y_test, p_test, labels=labels, target_names=label_names,
                                output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, p_test, labels=labels)
    pd.DataFrame(cm, index=label_names, columns=label_names).to_csv(d / "confusion_test.csv")
    pd.DataFrame({"ticket_id": ids_test, "y_true": y_test, "y_pred": p_test}).to_csv(
        d / "predictions_test.csv.gz", index=False)
    if groups_test is not None:
        extra = {**extra, "test_unique_titles": unique_title_metrics(groups_test, y_test, p_test, labels)}
        if groups_train is not None:
            # Generalization: test tickets whose cleaned title never occurs in training, one per title
            seen = set(groups_train)
            m = np.array([g not in seen for g in groups_test])
            if m.any():
                gt, yt, pt = np.asarray(groups_test)[m], np.asarray(y_test)[m], np.asarray(p_test)[m]
                extra["test_unseen_titles"] = unique_title_metrics(gt, yt, pt, labels)
                extra["test_unseen_titles"]["n_tickets"] = int(m.sum())
    rec = {**run_context(cfg), "setting": setting, "model": model, "seed": seed,
           "labels": list(map(str, label_names)), "test": m_test, "val": m_val,
           "per_class_test": rep, **extra}
    write_json(rec, d / "metrics.json")
    return rec
