"""Generate notebooks/colab_transformer_search_phase1.ipynb.

Pre-declared hyperparameter search for transformer classifiers, selected on VALIDATION only:
  models: BETO (dccuchile/bert-base-spanish-wwm-cased), XLM-RoBERTa base (RoBERTa-BNE unavailable)
  learning rate: 2e-5, 3e-5;  class-weighted loss: on, off;  up to 8 epochs, early stopping (patience 2)
Stage A runs every configuration on seed 0 and records validation macro-F1 only (the test split is
never scored in Stage A). Stage B runs the selected configuration on seeds 0-4 and scores the test
split once per seed. All configurations tried are logged and reported.

    python scripts/make_colab_search_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA = "42a85a3d6bbcb39d867fb77a2f15ee569e96129c9f37fc57854c5febd505cfc9"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


cells = [
md("""
# Phase 1: transformer hyperparameter search (validation only), Colab

Experiment `phase1_v2`, same anonymized file and splits as the rest of the paper.

**Protocol (declared before running):**
- Grid: model in {BETO, XLM-RoBERTa base} x learning rate in {2e-5, 3e-5} x class-weighted loss in {on, off};
  up to 8 epochs, early stopping on validation macro-F1 (patience 2). 8 configurations.
- **Stage A** (selection): every configuration on seed 0; only the validation split is scored.
- **Stage B** (evaluation): the configuration with the best validation macro-F1 on seeds 0-4;
  the test split is scored once per seed. Optionally also the best configuration of the other model.
- Every configuration tried is saved in `results/search/phase1_v2/stageA_seed0.json` and reported.

Runtime: **T4 GPU**. Dynamic padding (titles are short) keeps each epoch to about a minute or two.
If Colab disconnects, run all again: finished configurations and seeds are skipped (keep Drive on).
"""),
code("""
# 1. GPU check
import torch, transformers, sklearn, pandas, numpy, platform
print("torch", torch.__version__, "| transformers", transformers.__version__)
assert torch.cuda.is_available(), "No GPU: Runtime > Change runtime type > T4 GPU"
print("GPU:", torch.cuda.get_device_name(0))
"""),
code("""
# 2. Google Drive (strongly recommended: resumes after a disconnect)
SAVE_TO_DRIVE = True
import os
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_DIR = "/content/drive/MyDrive/itsm_paper_search"
    os.makedirs(DRIVE_DIR, exist_ok=True)
    if os.path.exists(f"{DRIVE_DIR}/results") and not os.path.exists("results"):
        os.system(f'cp -r "{DRIVE_DIR}/results" .')      # resume
        print("resumed from Drive")

def backup():
    if SAVE_TO_DRIVE:
        os.system(f'cp -r results "{DRIVE_DIR}/"')
"""),
code(f"""
# 3. Dataset (itsm_incidents_2024_public.csv), verified by hash
import hashlib
from google.colab import files
CSV = "itsm_incidents_2024_public.csv"
if not os.path.exists(CSV) and SAVE_TO_DRIVE and os.path.exists(f"{{DRIVE_DIR}}/{{CSV}}"):
    os.system(f'cp "{{DRIVE_DIR}}/{{CSV}}" .')
if not os.path.exists(CSV):
    up = files.upload(); name = list(up.keys())[0]
    if name != CSV: os.rename(name, CSV)
    if SAVE_TO_DRIVE: os.system(f'cp {{CSV}} "{{DRIVE_DIR}}/"')
EXPECTED_SHA = "{EXPECTED_SHA}"
assert hashlib.sha256(open(CSV, "rb").read()).hexdigest() == EXPECTED_SHA, "wrong dataset version"
df = pandas.read_csv(CSV, keep_default_na=False)
print(df.shape)
"""),
code("""
# 4. Search space and fixed settings
MODELS = {
    "bert_beto":   ["dccuchile/bert-base-spanish-wwm-cased"],
    # RoBERTa-BNE was declared first but neither official repository loads (weights withdrawn);
    # replaced by XLM-RoBERTa base before any RoBERTa result existed.
    "xlmr": ["FacebookAI/xlm-roberta-base", "xlm-roberta-base"],
}
GRID = [dict(model=m, learning_rate=lr, class_weighted_loss=cw)
        for m in MODELS for lr in (2e-5, 3e-5) for cw in (True, False)]
FIXED = dict(text_field="title_bert", max_len=64, batch_size=32, max_epochs=8, early_stopping_patience=2,
             weight_decay=0.01, warmup_ratio=0.1)
SELECTION_SEED, EVAL_SEEDS = 0, [0, 1, 2, 3, 4]
ALSO_RUN_BEST_OF_OTHER_MODEL = False     # True adds ~1.5 h: 5 seeds of the other model's best config
EXP, SETTING = "phase1_v2", "cluster__title__k10"
LABEL_NAMES = df.drop_duplicates("category_id").set_index("category_id")["category_en"].sort_index().tolist()
for g in GRID: print(g)
"""),
code("""
# 5. Data, metrics, writer (same definitions as src/itsm/evaluate.py)
import json, time, copy, random, datetime
import numpy as np, pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

device = torch.device("cuda")
text = df[FIXED["text_field"]].astype(str).str.strip().values
y = df["category_id"].astype(int).values
groups = df["title_clean"].values
ids = df["tid"].astype(str).values
LABELS = list(range(len(LABEL_NAMES)))

def metrics(y_true, y_pred, labels=LABELS):
    present = set(np.unique(y_true)) | set(np.unique(y_pred)); labels = [l for l in labels if l in present]
    return dict(accuracy=accuracy_score(y_true, y_pred), balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
                macro_f1=f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
                weighted_f1=f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
                macro_precision=precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
                macro_recall=recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
                n=int(len(y_true)))

def unique_title_metrics(g, t, p):
    d = pd.DataFrame({"g": g, "t": t, "p": p})
    u = d.groupby(["g", "t"])["p"].agg(lambda v: v.value_counts().index[0]).reset_index()
    return metrics(u["t"].values, u["p"].values)

def to_py(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

_tok_cache = {}
def load_backbone(key, n_labels):
    last = None
    for name in MODELS[key]:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=n_labels)
            return name, tok, model
        except Exception as e:
            last = e; print("could not load", name, "->", type(e).__name__)
    raise RuntimeError(f"no checkpoint for {key}: {last}")

def make_loader(tok, idx, shuffle, seed):
    enc = tok(list(text[idx]), truncation=True, max_length=FIXED["max_len"])
    items = [(enc["input_ids"][i], y[j]) for i, j in enumerate(idx)]
    pad = tok.pad_token_id
    def collate(batch):
        L = max(len(a) for a, _ in batch)
        ids_ = torch.full((len(batch), L), pad, dtype=torch.long); att = torch.zeros((len(batch), L), dtype=torch.long)
        for r, (a, _) in enumerate(batch):
            ids_[r, :len(a)] = torch.tensor(a); att[r, :len(a)] = 1
        return ids_, att, torch.tensor([b for _, b in batch])
    return DataLoader(items, batch_size=FIXED["batch_size"], shuffle=shuffle, collate_fn=collate,
                      generator=torch.Generator().manual_seed(seed))

def predict(model, loader):
    model.eval(); preds, probs = [], []
    with torch.no_grad():
        for a, m, _ in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(input_ids=a.to(device), attention_mask=m.to(device)).logits.float()
            pr = torch.softmax(logits, -1).cpu().numpy(); probs.append(pr); preds.append(pr.argmax(-1))
    return np.concatenate(preds), np.concatenate(probs)

def fit(cfg, seed, score_test):
    \"\"\"Train with early stopping on validation. Test is scored only if score_test=True.\"\"\"
    set_seed(seed)
    part = df[f"split_seed{seed}"].values
    idx = {s: np.where(part == s)[0] for s in ("train", "val", "test")}
    ckpt, tok, model = load_backbone(cfg["model"], len(LABELS)); model.to(device)
    tr = make_loader(tok, idx["train"], True, seed); va = make_loader(tok, idx["val"], False, seed)
    w = None
    if cfg["class_weighted_loss"]:
        w = torch.tensor(compute_class_weight("balanced", classes=np.array(LABELS), y=y[idx["train"]]),
                         dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=FIXED["weight_decay"])
    steps = len(tr) * FIXED["max_epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(FIXED["warmup_ratio"] * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    hist, best, bad, t0 = [], None, 0, time.perf_counter()
    for ep in range(1, FIXED["max_epochs"] + 1):
        model.train(); losses = []
        for a, m, yy in tr:
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(input_ids=a.to(device), attention_mask=m.to(device)).logits
            loss = loss_fn(logits.float(), yy.to(device))
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); losses.append(loss.item())
        p_va, _ = predict(model, va)
        f1 = f1_score(y[idx["val"]], p_va, labels=LABELS, average="macro", zero_division=0)
        h = dict(epoch=ep, train_loss=float(np.mean(losses)), val_accuracy=float((p_va == y[idx["val"]]).mean()),
                 val_macro_f1=float(f1), elapsed_s=round(time.perf_counter() - t0, 1))
        hist.append(h); print(f"  {cfg['model']} lr={cfg['learning_rate']} cw={cfg['class_weighted_loss']} seed={seed} | {h}", flush=True)
        if best is None or f1 > best[0]:
            best, bad = (f1, ep, copy.deepcopy(model.state_dict()), p_va), 0
        else:
            bad += 1
            if bad >= FIXED["early_stopping_patience"]: break
    out = dict(checkpoint=ckpt, best_val_macro_f1=float(best[0]), best_epoch=best[1], history=hist,
               fit_seconds=time.perf_counter() - t0, n_train=len(idx["train"]), n_val=len(idx["val"]))
    if score_test:
        model.load_state_dict(best[2])
        te = make_loader(tok, idx["test"], False, seed)
        t1 = time.perf_counter(); p_te, pr_te = predict(model, te)
        out.update(idx=idx, p_val=best[3], p_te=p_te, pr_te=pr_te,
                   predict_ms_per_ticket=1000 * (time.perf_counter() - t1) / len(idx["test"]),
                   trainable_parameters=int(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    del model; torch.cuda.empty_cache()
    return out
"""),
code("""
# 6. Stage A: selection on seed 0, validation only
os.makedirs(f"results/search/{EXP}", exist_ok=True)
SEARCH = f"results/search/{EXP}/stageA_seed{SELECTION_SEED}.json"
log = json.load(open(SEARCH)) if os.path.exists(SEARCH) else {"protocol": dict(grid=GRID, fixed=FIXED,
        selection="max validation macro-F1 on seed 0; test never scored in stage A",
        dataset_sha256=EXPECTED_SHA), "runs": []}
done = {json.dumps(r["config"], sort_keys=True) for r in log["runs"]}
for cfg in GRID:
    key = json.dumps(cfg, sort_keys=True)
    if key in done: print("skip", cfg); continue
    r = fit(cfg, SELECTION_SEED, score_test=False)
    log["runs"].append(dict(config=cfg, **r))
    json.dump(log, open(SEARCH, "w"), indent=2, default=to_py); backup()
table = pd.DataFrame([dict(**r["config"], checkpoint=r["checkpoint"], val_macro_f1=r["best_val_macro_f1"],
                           best_epoch=r["best_epoch"], minutes=round(r["fit_seconds"] / 60, 1)) for r in log["runs"]])
print(table.sort_values("val_macro_f1", ascending=False).to_string(index=False))
"""),
code("""
# 7. Stage B: selected configuration(s) on seeds 0-4, test scored once per seed
runs = sorted(log["runs"], key=lambda r: (-r["best_val_macro_f1"], r["best_epoch"]))
selected = [runs[0]["config"]]
if ALSO_RUN_BEST_OF_OTHER_MODEL:
    other = [r for r in runs if r["config"]["model"] != selected[0]["model"]]
    if other: selected.append(other[0]["config"])
log["selected"] = selected; json.dump(log, open(SEARCH, "w"), indent=2, default=to_py)
print("selected:", selected)

for cfg in selected:
    tag = f"{cfg['model']}_tuned"
    for seed in EVAL_SEEDS:
        d = f"results/runs/{EXP}/{SETTING}/{tag}/seed{seed}"
        if os.path.exists(f"{d}/metrics.json"): print("skip", tag, seed); continue
        r = fit(cfg, seed, score_test=True); idx = r["idx"]
        yt, pt = y[idx["test"]], r["p_te"]
        seen = set(groups[idx["train"]]); m = np.array([g not in seen for g in groups[idx["test"]]])
        unseen = unique_title_metrics(groups[idx["test"]][m], yt[m], pt[m]); unseen["n_tickets"] = int(m.sum())
        rec = dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                   git_commit="colab-search-notebook:" + EXPECTED_SHA[:12],
                   config_path="notebooks/colab_transformer_search_phase1.ipynb",
                   environment=dict(python=platform.python_version(), torch=torch.__version__,
                                    transformers=transformers.__version__, sklearn=sklearn.__version__),
                   setting=SETTING, model=tag, seed=seed, labels=LABEL_NAMES,
                   test=metrics(yt, pt), val=metrics(y[idx["val"]], r["p_val"]),
                   per_class_test=classification_report(yt, pt, labels=LABELS, target_names=LABEL_NAMES,
                                                        output_dict=True, zero_division=0),
                   test_unique_titles=unique_title_metrics(groups[idx["test"]], yt, pt), test_unseen_titles=unseen,
                   target="cluster", inputs=["title"], text_field=FIXED["text_field"], k=10,
                   model_name=r["checkpoint"], bert_config=dict(**cfg, **FIXED), selected_by="stage A, seed 0 validation macro-F1",
                   best_epoch=r["best_epoch"], history=r["history"], fit_seconds=r["fit_seconds"],
                   predict_ms_per_ticket=r["predict_ms_per_ticket"], trainable_parameters=r["trainable_parameters"],
                   device="cuda", gpu=torch.cuda.get_device_name(0), n_train=len(idx["train"]),
                   n_val=len(idx["val"]), n_test=len(idx["test"]), dataset_sha256=EXPECTED_SHA)
        os.makedirs(d, exist_ok=True)
        pd.DataFrame(confusion_matrix(yt, pt, labels=LABELS), index=LABEL_NAMES, columns=LABEL_NAMES).to_csv(f"{d}/confusion_test.csv")
        pred = pd.DataFrame({"ticket_id": ids[idx["test"]], "y_true": yt, "y_pred": pt})
        for c in LABELS: pred[f"p{c}"] = np.round(r["pr_te"][:, c], 5)       # probabilities: confidence threshold analysis
        pred.to_csv(f"{d}/predictions_test.csv.gz", index=False)
        json.dump(rec, open(f"{d}/metrics.json", "w"), indent=2, ensure_ascii=False, default=to_py)
        print(f"TEST {tag} seed {seed}: acc={rec['test']['accuracy']:.4f} macroF1={rec['test']['macro_f1']:.4f} unseen={unseen['macro_f1']:.4f}")
        backup()
"""),
code("""
# 8. Summary and download
for cfg in log["selected"]:
    tag = f"{cfg['model']}_tuned"; rows = []
    for seed in EVAL_SEEDS:
        r = json.load(open(f"results/runs/{EXP}/{SETTING}/{tag}/seed{seed}/metrics.json"))
        rows.append(dict(seed=seed, accuracy=r["test"]["accuracy"], macro_f1=r["test"]["macro_f1"],
                         unseen_macro_f1=r["test_unseen_titles"]["macro_f1"], best_epoch=r["best_epoch"]))
    s = pd.DataFrame(rows); print(tag); print(s.to_string(index=False))
    print("mean", s[["accuracy", "macro_f1", "unseen_macro_f1"]].mean().round(4).to_dict())
    print("std ", s[["accuracy", "macro_f1", "unseen_macro_f1"]].std().round(4).to_dict())
os.system("zip -qr transformer_search_phase1_v2.zip results")
files.download("transformer_search_phase1_v2.zip")
"""),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                    "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                    "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = ROOT / "notebooks" / "colab_transformer_search_phase1.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(out)
