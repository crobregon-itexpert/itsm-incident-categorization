"""Generate notebooks/colab_bert_phase1.ipynb (self-contained BERT run for Google Colab).

    python scripts/make_colab_notebook.py

The notebook needs only the shared CSV (itsm_incidents_2024_public.csv) and writes run logs in
the same layout and format as src/itsm (results/runs/<exp>/<setting>/<model>/seed<n>/), so the
returned zip can be unpacked into the repository and picked up by `python -m itsm.report`.
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
# Phase 1: BERT (BETO) incident categorization, Colab run

Paper: *IT Service Desk Automation Using ML and LLM* (revision, AIEng-26010199).
Repository: `itsm-incident-categorization`, branch `main`, experiment `phase1_v2`.

**Before you start:** Runtime > Change runtime type > **T4 GPU**.

**Input:** only `itsm_incidents_2024_public.csv` (anonymized, 59,860 incidents; same file and
same splits as the classical models in the repository).

**What it does:** fine-tunes `dccuchile/bert-base-spanish-wwm-cased` on the 10 static categories
for each of the 5 seeds, selects the epoch on the validation split (macro-F1), scores the test
split once, and writes one folder per seed with `metrics.json`, `predictions_test.csv.gz` and
`confusion_test.csv`. About 10-15 minutes per seed on a T4.

**Output:** `bert_runs_phase1_v2.zip` (downloads at the end). Unzip it at the root of the
repository and run `python -m itsm.report --config configs/phase1_v2.yaml`.
"""),
code("""
# 1. GPU check
import torch, transformers, sklearn, pandas, numpy, platform
print("torch", torch.__version__, "| transformers", transformers.__version__, "| sklearn", sklearn.__version__)
assert torch.cuda.is_available(), "No GPU: Runtime > Change runtime type > T4 GPU"
print("GPU:", torch.cuda.get_device_name(0))
"""),
code("""
# 2. Optional: save results to Google Drive after every seed (protects against disconnects)
SAVE_TO_DRIVE = True
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_DIR = "/content/drive/MyDrive/itsm_paper_bert_runs"
"""),
code(f"""
# 3. Upload the dataset (itsm_incidents_2024_public.csv) and verify it
import hashlib, os
from google.colab import files
CSV = "itsm_incidents_2024_public.csv"
if not os.path.exists(CSV):
    up = files.upload()
    name = list(up.keys())[0]
    if name != CSV:
        os.rename(name, CSV)
sha = hashlib.sha256(open(CSV, "rb").read()).hexdigest()
EXPECTED_SHA = "{EXPECTED_SHA}"
print("sha256", sha)
assert sha == EXPECTED_SHA, "This is not the dataset version used in the paper (phase1_v2)."
df = pandas.read_csv(CSV, keep_default_na=False)
print(df.shape); df.head()
"""),
code("""
# 4. Configuration (same values as configs/phase1_v2.yaml, section `bert`)
CFG = dict(
    experiment="phase1_v2",
    setting="cluster__title__k10",
    model_tag="bert_beto",             # use "bert_beto_clean" with TEXT_FIELD="title_clean" for the ablation
    model_name="dccuchile/bert-base-spanish-wwm-cased",
    text_field="title_bert",           # anonymized title in its original form
    max_len=64, batch_size=32, epochs=4, learning_rate=2e-5, weight_decay=0.01,
    warmup_ratio=0.1, freeze_layers=0, class_weighted_loss=True, early_stopping_patience=2,
)
SEEDS = [0, 1, 2, 3, 4]
OUT = f"results/runs/{CFG['experiment']}/{CFG['setting']}/{CFG['model_tag']}"
LABEL_NAMES = df.drop_duplicates("category_id").set_index("category_id")["category_en"].sort_index().tolist()
print(len(LABEL_NAMES), "categories:", LABEL_NAMES)
"""),
code("""
# 5. Metrics and run writer (same definitions as src/itsm/evaluate.py)
import json, time, copy, random, datetime
import numpy as np, pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)

def metrics(y_true, y_pred, labels):
    present = set(np.unique(y_true)) | set(np.unique(y_pred))
    labels = [l for l in labels if l in present]
    return dict(accuracy=accuracy_score(y_true, y_pred), balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
                macro_f1=f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
                weighted_f1=f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
                macro_precision=precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
                macro_recall=recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
                n=int(len(y_true)))

def unique_title_metrics(groups, y_true, y_pred, labels):
    d = pd.DataFrame({"g": groups, "t": y_true, "p": y_pred})
    u = d.groupby(["g", "t"])["p"].agg(lambda v: v.value_counts().index[0]).reset_index()
    return metrics(u["t"].values, u["p"].values, labels)

def to_py(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)

def write_run(seed, ids_test, y_test, p_test, y_val, p_val, groups_test, groups_train, extra):
    d = f"{OUT}/seed{seed}"; os.makedirs(d, exist_ok=True)
    labels = list(range(len(LABEL_NAMES)))
    rec = dict(timestamp_utc=datetime.datetime.utcnow().isoformat(timespec="seconds"),
               git_commit="colab-notebook:" + EXPECTED_SHA[:12], config_path="notebooks/colab_bert_phase1.ipynb",
               environment=dict(python=platform.python_version(), torch=torch.__version__,
                                transformers=transformers.__version__, sklearn=sklearn.__version__,
                                pandas=pandas.__version__, numpy=numpy.__version__, platform=platform.platform()),
               setting=CFG["setting"], model=CFG["model_tag"], seed=seed, labels=LABEL_NAMES,
               test=metrics(y_test, p_test, labels), val=metrics(y_val, p_val, labels),
               per_class_test=classification_report(y_test, p_test, labels=labels, target_names=LABEL_NAMES,
                                                    output_dict=True, zero_division=0),
               test_unique_titles=unique_title_metrics(groups_test, y_test, p_test, labels), **extra)
    seen = set(groups_train); m = np.array([g not in seen for g in groups_test])
    rec["test_unseen_titles"] = unique_title_metrics(np.asarray(groups_test)[m], np.asarray(y_test)[m],
                                                     np.asarray(p_test)[m], labels)
    rec["test_unseen_titles"]["n_tickets"] = int(m.sum())
    pd.DataFrame(confusion_matrix(y_test, p_test, labels=labels), index=LABEL_NAMES, columns=LABEL_NAMES).to_csv(f"{d}/confusion_test.csv")
    pd.DataFrame({"ticket_id": ids_test, "y_true": y_test, "y_pred": p_test}).to_csv(f"{d}/predictions_test.csv.gz", index=False)
    with open(f"{d}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False, default=to_py)
    return rec
"""),
code("""
# 6. Training (same procedure as src/itsm/train_bert.py)
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight

device = torch.device("cuda")
tok = AutoTokenizer.from_pretrained(CFG["model_name"])
text = df[CFG["text_field"]].astype(str).str.strip().values
y = df["category_id"].astype(int).values
groups = df["title_clean"].values
ids = df["tid"].astype(str).values
labels = list(range(len(LABEL_NAMES)))

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def predict(model, loader):
    model.eval(); out = []
    with torch.no_grad():
        for a, m, _ in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out.append(model(input_ids=a.to(device), attention_mask=m.to(device)).logits.argmax(-1).cpu().numpy())
    return np.concatenate(out)

def train_one(seed):
    set_seed(seed)
    part = df[f"split_seed{seed}"].values
    idx = {s: np.where(part == s)[0] for s in ("train", "val", "test")}
    def loader(s, shuffle):
        enc = tok(list(text[idx[s]]), truncation=True, max_length=CFG["max_len"], padding="max_length", return_tensors="pt")
        ds = TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(y[idx[s]]))
        return DataLoader(ds, batch_size=CFG["batch_size"], shuffle=shuffle, generator=torch.Generator().manual_seed(seed))
    tr, va, te = loader("train", True), loader("val", False), loader("test", False)
    model = AutoModelForSequenceClassification.from_pretrained(CFG["model_name"], num_labels=len(labels)).to(device)
    if CFG["freeze_layers"]:
        base = getattr(model, model.base_model_prefix)
        for p in base.embeddings.parameters(): p.requires_grad = False
        for layer in base.encoder.layer[:CFG["freeze_layers"]]:
            for p in layer.parameters(): p.requires_grad = False
    weight = None
    if CFG["class_weighted_loss"]:
        weight = torch.tensor(compute_class_weight("balanced", classes=np.array(labels), y=y[idx["train"]]),
                              dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=CFG["learning_rate"],
                            weight_decay=CFG["weight_decay"])
    steps = len(tr) * CFG["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(CFG["warmup_ratio"] * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    history, best, bad, t0 = [], None, 0, time.perf_counter()
    for epoch in range(1, CFG["epochs"] + 1):
        model.train(); losses = []
        for a, m, yy in tr:
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(input_ids=a.to(device), attention_mask=m.to(device)).logits
            loss = loss_fn(logits.float(), yy.to(device))
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); losses.append(loss.item())
        p_va = predict(model, va)
        f1 = f1_score(y[idx["val"]], p_va, labels=labels, average="macro", zero_division=0)
        h = dict(epoch=epoch, train_loss=float(np.mean(losses)), val_accuracy=float((p_va == y[idx["val"]]).mean()),
                 val_macro_f1=float(f1), elapsed_s=round(time.perf_counter() - t0, 1))
        history.append(h); print(f"seed {seed} | {h}", flush=True)
        if best is None or f1 > best[0]:
            best, bad = (f1, epoch, copy.deepcopy(model.state_dict()), p_va), 0
        else:
            bad += 1
            if bad >= CFG["early_stopping_patience"]: break
    fit_s = time.perf_counter() - t0
    model.load_state_dict(best[2])
    t1 = time.perf_counter(); p_te = predict(model, te)
    pred_ms = 1000 * (time.perf_counter() - t1) / len(idx["test"])
    extra = dict(target="cluster", inputs=["title"], text_field=CFG["text_field"], k=10, model_name=CFG["model_name"],
                 bert_config=CFG, best_epoch=best[1], history=history,
                 trainable_parameters=int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                 fit_seconds=fit_s, predict_ms_per_ticket=pred_ms, device="cuda", gpu=torch.cuda.get_device_name(0),
                 n_train=len(idx["train"]), n_val=len(idx["val"]), n_test=len(idx["test"]), dataset_sha256=EXPECTED_SHA)
    rec = write_run(seed, ids[idx["test"]], y[idx["test"]], p_te, y[idx["val"]], best[3],
                    groups[idx["test"]], groups[idx["train"]], extra)
    print(f"TEST seed {seed}: acc={rec['test']['accuracy']:.4f} macroF1={rec['test']['macro_f1']:.4f} "
          f"unseen macroF1={rec['test_unseen_titles']['macro_f1']:.4f}")
    del model; torch.cuda.empty_cache()

for seed in SEEDS:
    if os.path.exists(f"{OUT}/seed{seed}/metrics.json"):
        print("skip seed", seed); continue
    train_one(seed)
    if SAVE_TO_DRIVE:
        os.system(f'mkdir -p "{DRIVE_DIR}" && cp -r results "{DRIVE_DIR}/"')
"""),
code("""
# 7. Summary and download
rows = []
for seed in SEEDS:
    r = json.load(open(f"{OUT}/seed{seed}/metrics.json"))
    rows.append(dict(seed=seed, accuracy=r["test"]["accuracy"], macro_f1=r["test"]["macro_f1"],
                     unseen_macro_f1=r["test_unseen_titles"]["macro_f1"], best_epoch=r["best_epoch"],
                     minutes=round(r["fit_seconds"] / 60, 1)))
s = pd.DataFrame(rows); print(s.to_string(index=False))
print("mean", s[["accuracy", "macro_f1", "unseen_macro_f1"]].mean().round(4).to_dict())
print("std ", s[["accuracy", "macro_f1", "unseen_macro_f1"]].std().round(4).to_dict())
os.system("zip -qr bert_runs_phase1_v2.zip results")
files.download("bert_runs_phase1_v2.zip")
"""),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                    "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                    "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = ROOT / "notebooks" / "colab_bert_phase1.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(out)
