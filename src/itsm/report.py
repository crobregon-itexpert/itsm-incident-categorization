"""Step 6: build every table and figure in the paper from the run logs. No number is typed by hand.

    python -m itsm.report --config configs/phase1.yaml

Outputs
  results/summary/<exp>/all_runs.csv         one row per (setting, model, seed)
  results/summary/<exp>/summary.csv          mean / std per (setting, model)
  results/summary/<exp>/significance.csv     McNemar tests, best model vs each other model
  paper/tables/*.tex                         LaTeX tables (IEEEtran-ready)
  paper/figures/*.pdf|png                    cluster sweep, confusion matrix, training curves
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .runlog import ROOT, load_config, write_json

MODEL_LABELS = {
    "logreg": "Logistic Regression", "linear_svm": "Linear SVM", "svm_word_char": "SVM (word+char n-grams)",
    "complement_nb": "Complement NB", "random_forest": "Random Forest", "mlp": "MLP (TF-IDF)",
    "bert_beto": "BERT (BETO, fine-tuned)", "bert_beto_clean": "BERT (BETO, cleaned title)",
    "bert_beto_tuned": "BERT (BETO, tuned)", "roberta_bne_tuned": "RoBERTa-BNE (tuned)", "xlmr_tuned": "XLM-RoBERTa (tuned)", "xlmr": "XLM-RoBERTa",
}
BLUE = "#2a78d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def _rel(p):
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def collect(cfg) -> pd.DataFrame:
    base = ROOT / cfg["output"]["runs_dir"] / cfg["experiment_name"]
    rows = []
    for f in sorted(base.glob("*/*/seed*/metrics.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        rows.append({
            "setting": r["setting"], "model": r["model"], "seed": r["seed"],
            **{f"test_{k}": v for k, v in r["test"].items() if k != "n"},
            "val_macro_f1": r["val"]["macro_f1"], "n_test": r["test"]["n"],
            "test_unique_accuracy": r.get("test_unique_titles", {}).get("accuracy"),
            "test_unique_macro_f1": r.get("test_unique_titles", {}).get("macro_f1"),
            "n_test_unique": r.get("test_unique_titles", {}).get("n"),
            "fit_seconds": r.get("fit_seconds"), "predict_ms_per_ticket": r.get("predict_ms_per_ticket"),
            "best_params": json.dumps(r.get("best_params") or r.get("bert_config", {}).get("learning_rate")),
            "test_unseen_macro_f1": r.get("test_unseen_titles", {}).get("macro_f1"),
            "test_unseen_accuracy": r.get("test_unseen_titles", {}).get("accuracy"),
            "git_commit": r["git_commit"], "path": _rel(f.parent),
        })
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = ["test_accuracy", "test_macro_f1", "test_weighted_f1", "test_balanced_accuracy",
               "test_unique_accuracy", "test_unique_macro_f1", "test_unseen_accuracy", "test_unseen_macro_f1",
               "val_macro_f1", "fit_seconds", "predict_ms_per_ticket"]
    g = runs.groupby(["setting", "model"])
    out = g[metrics].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_seeds"] = g.size()
    return out.reset_index()


def pm(mean, std, pct=False, nd=3):
    if pd.isna(mean):
        return "--"
    if pct:
        return f"{100*mean:.1f} $\\pm$ {100*(0 if pd.isna(std) else std):.1f}"
    return f"{mean:.{nd}f} $\\pm$ {(0 if pd.isna(std) else std):.{nd}f}"


def latex_table(path, caption, label, header, rows, colspec, wide=False):
    env = "table*" if wide else "table"
    width = "\\textwidth" if wide else "\\columnwidth"
    lines = [f"\\begin{{{env}}}[!t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\footnotesize", f"\\begin{{adjustbox}}{{max width={width}}}",
             f"\\begin{{tabular}}{{{colspec}}}", "\\hline",
             " & ".join(header) + " \\\\", "\\hline"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\hline", "\\end{tabular}", "\\end{adjustbox}", f"\\end{{{env}}}", ""]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def mcnemar(cfg, setting, model_a, model_b, seed):
    from scipy.stats import binomtest
    base = ROOT / cfg["output"]["runs_dir"] / cfg["experiment_name"] / setting
    a = pd.read_csv(base / model_a / f"seed{seed}" / "predictions_test.csv.gz", dtype={"ticket_id": str})
    b = pd.read_csv(base / model_b / f"seed{seed}" / "predictions_test.csv.gz", dtype={"ticket_id": str})
    m = a.merge(b, on="ticket_id", suffixes=("_a", "_b"))
    ca, cb = m.y_pred_a == m.y_true_a, m.y_pred_b == m.y_true_b
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, p


def main_setting(cfg):
    from .evaluate import setting_name
    c = cfg["classification"]
    return setting_name(c["target"], c["input_fields"], cfg["clustering"]["selected_k"])


def build_tables(cfg, runs, summ):
    tdir = ROOT / cfg["output"]["tables_dir"]
    sdir = ROOT / cfg["output"]["summary_dir"] / cfg["experiment_name"]
    main = main_setting(cfg)
    k = cfg["clustering"]["selected_k"]

    # Table I: main comparison
    s = summ[summ.setting == main].sort_values("test_macro_f1_mean", ascending=False)
    # v2 (random split): generalization = unseen titles; v1 (group split): unique titles
    gen = "unseen" if s["test_unseen_macro_f1_mean"].notna().any() else "unique"
    rows = []
    for _, r in s.iterrows():
        rows.append([MODEL_LABELS.get(r.model, r.model), pm(r.test_accuracy_mean, r.test_accuracy_std, pct=True),
                     pm(r.test_macro_f1_mean, r.test_macro_f1_std), pm(r.test_weighted_f1_mean, r.test_weighted_f1_std),
                     pm(r[f"test_{gen}_macro_f1_mean"], r[f"test_{gen}_macro_f1_std"])])
    n_test = int(runs[runs.setting == main].n_test.iloc[0]) if len(s) else 0
    latex_table(tdir / "table1_main.tex",
                f"Incident categorization on the held-out test set (k={k} categories, ticket title only). "
                f"Mean $\\pm$ standard deviation over {int(s.n_seeds.max()) if len(s) else 0} independent "
                f"train/validation/test splits; about {n_test:,} test tickets per split. "
                + ("Unseen titles: test tickets whose cleaned title never occurs in training, each distinct title counted once."
                   if gen == "unseen" else
                   "No title shared between splits. Unique titles: each distinct cleaned title counts once."),
                "tab:main", ["Model", "Accuracy (\\%)", "Macro-F1", "Weighted-F1", f"Macro-F1 ({gen} titles)"], rows, "lcccc", wide=True)

    # Significance: best model vs others, per seed (McNemar exact test)
    sig = []
    if len(s) > 1:
        best = s.iloc[0].model
        for other in s.model.iloc[1:]:
            for seed in sorted(runs[(runs.setting == main) & (runs.model == other)].seed):
                if seed in set(runs[(runs.setting == main) & (runs.model == best)].seed):
                    n01, n10, p = mcnemar(cfg, main, best, other, seed)
                    sig.append({"best": best, "other": other, "seed": seed, "best_right_other_wrong": n01,
                                "best_wrong_other_right": n10, "p_value": p})
    pd.DataFrame(sig).to_csv(sdir / "significance.csv", index=False)

    # k sensitivity
    ks = [x for x in sorted(runs.setting.unique()) if x.startswith("cluster__title__k")]
    sweep = pd.read_csv(sdir / "cluster_sweep.csv") if (sdir / "cluster_sweep.csv").exists() else None
    models = [m for m in ["logreg", "linear_svm", "bert_beto"] if m in set(runs.model)]
    rows = []
    for st in sorted(ks, key=lambda x: int(x.split("__k")[1])):
        kk = int(st.split("__k")[1])
        row = [str(kk)]
        if sweep is not None:
            sw = sweep[sweep.k == kk].iloc[0]
            row += [f"{sw.silhouette:.3f}", f"{sw.stability_ari_mean:.2f}", f"{sw.nmi_grupo_asignado:.2f}"]
        for m in models:
            r = summ[(summ.setting == st) & (summ.model == m)]
            row.append(pm(r.test_macro_f1_mean.iloc[0], r.test_macro_f1_std.iloc[0]) if len(r) else "--")
        rows.append(row)
    latex_table(tdir / "table_k_sensitivity.tex",
                "Sensitivity to the number of clusters $k$: cluster quality (silhouette, seed stability as mean "
                "pairwise ARI over 5 K-means seeds, NMI with the human resolver group) and test Macro-F1 "
                "(mean $\\pm$ std over 5 splits).",
                "tab:ksens", ["$k$", "Silh.", "Stab.", "NMI$_{grp}$"] + [MODEL_LABELS[m] for m in models], rows,
                "c" * (4 + len(models)))

    # Input ablation and external target
    rows = []
    for st, desc in [(main, "Title (deployable)"), (main.replace("__title__", "__title+solution__"),
                                                    "Title + solution (upper bound)"),
                     ("grupo__title", "Title $\\rightarrow$ resolver group")]:
        for m in ["logreg", "linear_svm", "svm_word_char", "complement_nb", "bert_beto"]:
            r = summ[(summ.setting == st) & (summ.model == m)]
            if len(r):
                r = r.iloc[0]
                rows.append([desc, MODEL_LABELS[m], pm(r.test_accuracy_mean, r.test_accuracy_std, pct=True),
                             pm(r.test_macro_f1_mean, r.test_macro_f1_std)])
    latex_table(tdir / "table_ablation.tex",
                "Input and target ablation. The solution field is written after the ticket is resolved, "
                "so title + solution is an upper bound and not a deployable setting. The resolver-group "
                "target uses the human-assigned field Grupo Asignado instead of cluster labels.",
                "tab:ablation", ["Setting", "Model", "Accuracy (\\%)", "Macro-F1"], rows, "llcc")

    # Transformer search (stage A, validation only): every configuration tried
    sf = ROOT / "results" / "search" / cfg["experiment_name"] / "stageA_seed0.json"
    if sf.exists():
        st = json.loads(sf.read_text(encoding="utf-8"))
        sel = [json.dumps(c, sort_keys=True) for c in st.get("selected", [])[:1]]
        name = {"bert_beto": "BETO", "xlmr": "XLM-RoBERTa base"}
        rows = []
        for r in sorted(st["runs"], key=lambda r: -r["best_val_macro_f1"]):
            c = r["config"]
            mark = " (selected)" if json.dumps(c, sort_keys=True) in sel else ""
            rows.append([name.get(c["model"], c["model"]) + mark, f"{c['learning_rate']:.0e}",
                         "yes" if c["class_weighted_loss"] else "no", f"{r['best_val_macro_f1']:.3f}",
                         str(r["best_epoch"]), f"{r['fit_seconds']/60:.0f}"])
        fx = st["protocol"]["fixed"]
        latex_table(tdir / "table_transformer_search.tex",
                    "Transformer hyperparameter search (declared before running): validation macro-F1 on seed 0; "
                    "the test split was not scored for any configuration in this stage. Fixed: AdamW, weight decay "
                    f"{fx['weight_decay']}, linear warm-up {int(100*fx['warmup_ratio'])}\\%, batch {fx['batch_size']}, "
                    f"max length {fx['max_len']}, up to {fx['max_epochs']} epochs, early stopping patience "
                    f"{fx['early_stopping_patience']}, mixed precision, full fine-tuning, NVIDIA T4.",
                    "tab:search", ["Model", "LR", "Class weights", "Val. Macro-F1", "Best epoch", "Min"], rows, "lccccc")

    # Latency / size (from itsm.benchmark)
    lat_f = sdir / "latency_cpu.csv"
    if lat_f.exists():
        lat = pd.read_csv(lat_f)
        rows = [[MODEL_LABELS.get(r.model, r.model), f"{r.latency_ms_median:.2f}", f"{r.latency_ms_p95:.2f}",
                 f"{r.model_size_mb:.1f}"] for r in lat.itertuples()]
        latex_table(tdir / "table_latency.tex",
                    "Inference cost of a single new ticket on CPU (raw title $\\rightarrow$ cleaning "
                    "$\\rightarrow$ TF-IDF $\\rightarrow$ prediction), 1,000 test tickets, one process.",
                    "tab:latency", ["Model", "Median (ms)", "p95 (ms)", "Size (MB)"], rows, "lccc")

    # Cluster sweep table
    if sweep is not None:
        rows = [[str(r.k), f"{r.silhouette:.3f}", f"{r.davies_bouldin:.2f}", f"{r.stability_ari_mean:.2f}",
                 f"{r.largest_cluster_pct:.1f}", f"{r.nmi_nivel_2:.2f}", f"{r.nmi_grupo_asignado:.2f}"]
                for r in sweep.itertuples()]
        latex_table(tdir / "table_cluster_sweep.tex",
                    "K-means on LSA vectors of title + solution: internal quality, stability and agreement "
                    "with human fields for $k=5\\ldots20$.", "tab:sweep",
                    ["$k$", "Silh.", "DB", "Stab.", "Largest (\\%)", "NMI$_{N2}$", "NMI$_{grp}$"], rows, "ccccccc")


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def build_figures(cfg, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fdir = ROOT / cfg["output"]["figures_dir"]
    fdir.mkdir(parents=True, exist_ok=True)
    sdir = ROOT / cfg["output"]["summary_dir"] / cfg["experiment_name"]
    k_sel = cfg["clustering"]["selected_k"]
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.titleweight": "bold",
                         "axes.labelcolor": INK2, "text.color": INK})

    # Cluster sweep: one metric per panel (no dual axes)
    if (sdir / "cluster_sweep.csv").exists():
        sw = pd.read_csv(sdir / "cluster_sweep.csv")
        panels = [("inertia", "Inertia (elbow)"), ("silhouette", "Silhouette (cosine)"),
                  ("davies_bouldin", "Davies-Bouldin (lower is better)"),
                  ("stability_ari_mean", "Seed stability (mean ARI)"),
                  ("nmi_nivel_2", "NMI with Nivel 2"), ("nmi_grupo_asignado", "NMI with resolver group")]
        fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.8), constrained_layout=True)
        for ax, (col, title) in zip(axes.ravel(), panels):
            ax.plot(sw.k, sw[col], color=BLUE, linewidth=2, marker="o", markersize=4)
            if col == "stability_ari_mean":
                ax.fill_between(sw.k, sw[col] - sw.stability_ari_std, sw[col] + sw.stability_ari_std,
                                color=BLUE, alpha=0.15, linewidth=0)
            ax.axvline(k_sel, color=INK2, linewidth=0.8, linestyle="--")
            ax.set_title(title, loc="left")
            ax.set_xlabel("k")
            _style(ax)
        fig.savefig(fdir / "cluster_sweep.pdf")
        fig.savefig(fdir / "cluster_sweep.png", dpi=200)
        plt.close(fig)

    # Confusion matrix (row-normalized) of the best model on the main setting, seed 0
    main = main_setting(cfg)
    sub = runs[(runs.setting == main) & (runs.seed == 0)]
    if len(sub):
        best = sub.sort_values("test_macro_f1", ascending=False).iloc[0]
        d = ROOT / best.path
        cm = pd.read_csv(d / "confusion_test.csv", index_col=0)
        names_file = ROOT / "configs" / f"cluster_names_k{k_sel}.yaml"
        if names_file.exists() and main.startswith("cluster__"):
            import yaml
            names = yaml.safe_load(open(names_file, encoding="utf-8"))
            lab = [str(names.get(i, i)) for i in range(len(cm))]   # rows/cols are in cluster-id order
            cm.index, cm.columns = lab, lab
        cmn = cm.div(cm.sum(axis=1), axis=0)
        fig, ax = plt.subplots(figsize=(7.16, 5.2), constrained_layout=True)
        im = ax.imshow(cmn.values, cmap="Blues", vmin=0, vmax=1)
        n = len(cm)
        ax.set_xticks(range(n), [str(c) for c in cm.columns], rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(n), [str(c) for c in cm.index], fontsize=7)
        for i in range(n):
            for j in range(n):
                v = cmn.values[i, j]
                if v >= 0.01:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if v > 0.6 else INK)
        ax.set_xlabel("Predicted category")
        ax.set_ylabel("True category")
        ax.set_title(f"{MODEL_LABELS.get(best.model, best.model)}, test split (seed 0), row-normalized",
                     loc="left")
        fig.colorbar(im, ax=ax, shrink=0.7)
        fig.savefig(fdir / "confusion_best.pdf")
        fig.savefig(fdir / "confusion_best.png", dpi=200)
        plt.close(fig)

    # BERT training curves if present (one panel per measure)
    hist = []
    for f in (ROOT / cfg["output"]["runs_dir"] / cfg["experiment_name"] / main / "bert_beto").glob("seed*/metrics.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        for h in r.get("history", []):
            hist.append({**h, "seed": r["seed"]})
    if hist:
        h = pd.DataFrame(hist)
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.4), constrained_layout=True)
        for ax, col, title in [(axes[0], "train_loss", "Training loss"),
                               (axes[1], "val_macro_f1", "Validation Macro-F1")]:
            for seed, g in h.groupby("seed"):
                ax.plot(g.epoch, g[col], color=BLUE, alpha=0.35, linewidth=1)
            m = h.groupby("epoch")[col].mean()
            ax.plot(m.index, m.values, color=BLUE, linewidth=2, marker="o", markersize=4, label="mean of seeds")
            ax.set_title(title, loc="left")
            ax.set_xlabel("Epoch")
            _style(ax)
        fig.savefig(fdir / "bert_training.pdf")
        fig.savefig(fdir / "bert_training.png", dpi=200)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1_v2.yaml")
    cfg = load_config(ap.parse_args().config)
    sdir = ROOT / cfg["output"]["summary_dir"] / cfg["experiment_name"]
    runs = collect(cfg)
    runs.to_csv(sdir / "all_runs.csv", index=False)
    summ = summarize(runs)
    summ.to_csv(sdir / "summary.csv", index=False)
    build_tables(cfg, runs, summ)
    build_figures(cfg, runs)
    cols = ["setting", "model", "n_seeds", "test_accuracy_mean", "test_accuracy_std", "test_macro_f1_mean",
            "test_macro_f1_std"]
    print(summ[cols].sort_values(["setting", "test_macro_f1_mean"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
