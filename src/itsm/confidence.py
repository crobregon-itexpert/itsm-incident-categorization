"""Confidence threshold analysis for the deployed classifier ("unclassified" option of the API).

    python -m itsm.confidence --config configs/phase1_v2.yaml

For every run whose predictions include class probabilities (p0..p9), and for each threshold t,
tickets with max probability >= t are classified automatically and the rest are left
"unclassified" for the analyst. Reports coverage (share classified automatically), accuracy and
macro-F1 on the classified tickets, and the expected calibration error (ECE, 15 bins).
Outputs: results/summary/<exp>/confidence.csv, paper/tables/table_confidence.tex,
paper/figures/confidence_coverage.pdf|png
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .report import MODEL_LABELS, latex_table, _style, INK2
from .runlog import ROOT, load_config, run_context, write_json

THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
SERIES = ["#2a78d6", "#eb6834"]          # categorical slots 1-2 (validated palette)


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e


def analyse(cfg):
    base = ROOT / cfg["output"]["runs_dir"] / cfg["experiment_name"]
    rows, curves = [], []
    for f in sorted(base.glob("*/*/seed*/predictions_test.csv.gz")):
        p = pd.read_csv(f)
        pc = [c for c in p.columns if c.startswith("p") and c[1:].isdigit()]
        if not pc:
            continue
        setting, model, seed = f.parts[-4], f.parts[-3], int(f.parts[-2][4:])
        P = p[pc].values
        conf, pred, y = P.max(1), P.argmax(1), p["y_true"].values
        correct = pred == y
        base_row = {"setting": setting, "model": model, "seed": seed, "ece": ece(conf, correct)}
        for t in THRESHOLDS:
            m = conf >= t
            rows.append({**base_row, "threshold": t, "coverage": m.mean(),
                         "accuracy_classified": correct[m].mean() if m.any() else np.nan,
                         "macro_f1_classified": f1_score(y[m], pred[m], average="macro") if m.any() else np.nan})
        order = np.argsort(-conf)
        cum = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
        cov = np.arange(1, len(order) + 1) / len(order)
        grid = np.linspace(0.3, 1.0, 71)
        curves.append(pd.DataFrame({"model": model, "seed": seed, "coverage": grid,
                                    "accuracy": np.interp(grid, cov, cum)}))
    df = pd.DataFrame(rows)
    out = ROOT / cfg["output"]["summary_dir"] / cfg["experiment_name"]
    df.to_csv(out / "confidence.csv", index=False)
    return df, pd.concat(curves) if curves else None


def outputs(cfg, df, curves):
    summ = df.groupby(["model", "threshold"]).agg(
        coverage=("coverage", "mean"), coverage_sd=("coverage", "std"),
        acc=("accuracy_classified", "mean"), acc_sd=("accuracy_classified", "std"),
        f1=("macro_f1_classified", "mean"), f1_sd=("macro_f1_classified", "std")).reset_index()
    eces = df.groupby("model")["ece"].agg(["mean", "std"])
    rows = []
    for _, r in summ.iterrows():
        rows.append([MODEL_LABELS.get(r.model, r.model), "none" if r.threshold == 0 else f"{r.threshold:.2f}",
                     f"{100*r.coverage:.1f} $\\pm$ {100*r.coverage_sd:.1f}", f"{100*r.acc:.1f} $\\pm$ {100*r.acc_sd:.1f}",
                     f"{r.f1:.3f} $\\pm$ {r.f1_sd:.3f}"])
    latex_table(ROOT / cfg["output"]["tables_dir"] / "table_confidence.tex",
                "Confidence threshold for the ``unclassified'' option: tickets whose top-class probability is "
                "below the threshold are left to the analyst. Coverage = share classified automatically; "
                "accuracy and macro-F1 on the classified tickets. Mean $\\pm$ std over 5 splits. ECE: "
                + ", ".join(f"{MODEL_LABELS.get(m, m)} {v['mean']:.3f}" for m, v in eces.iterrows()) + ".",
                "tab:confidence", ["Model", "Threshold", "Coverage (\\%)", "Accuracy (\\%)", "Macro-F1"], rows, "llccc", wide=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
    for color, (model, g) in zip(SERIES, curves.groupby("model")):
        m = g.groupby("coverage")["accuracy"].agg(["mean", "std"]).reset_index()
        ax.plot(m.coverage * 100, m["mean"] * 100, color=color, linewidth=2, label=MODEL_LABELS.get(model, model))
        ax.fill_between(m.coverage * 100, (m["mean"] - m["std"]) * 100, (m["mean"] + m["std"]) * 100,
                        color=color, alpha=0.15, linewidth=0)
    ax.set_xlabel("Coverage: tickets classified automatically (%)")
    ax.set_ylabel("Accuracy on classified (%)")
    ax.set_title("Accuracy vs coverage (test, 5 splits)", loc="left", fontsize=8, fontweight="bold")
    ax.legend(frameon=False, fontsize=7, labelcolor=INK2)
    _style(ax)
    fdir = ROOT / cfg["output"]["figures_dir"]
    fig.savefig(fdir / "confidence_coverage.pdf")
    fig.savefig(fdir / "confidence_coverage.png", dpi=200)
    plt.close(fig)
    write_json({**run_context(cfg), "thresholds": THRESHOLDS,
                "summary": summ.to_dict(orient="records"), "ece": eces.to_dict(orient="index")},
               ROOT / cfg["output"]["summary_dir"] / cfg["experiment_name"] / "confidence.json")
    return summ, eces


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1_v2.yaml")
    cfg = load_config(ap.parse_args().config)
    df, curves = analyse(cfg)
    summ, eces = outputs(cfg, df, curves)
    pd.set_option("display.width", 160)
    print(summ.round(4).to_string(index=False)); print(eces.round(4))
