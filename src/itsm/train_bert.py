"""Step 5: fine-tune a Spanish BERT (BETO) on the same labels, splits and logging as the classical models.

Runs on a GPU runtime (Colab, Vertex AI Workbench / Custom Job). Needs the repo, the
interim table (data/interim/tickets.pkl), results/labels and results/splits.

    pip install torch transformers
    python -m itsm.train_bert --config configs/phase1.yaml                 # selected k, 5 seeds
    python -m itsm.train_bert --k 6 12 --seeds 0 1 2                       # sensitivity
    python -m itsm.train_bert --model-name xlm-roberta-base --tag xlmr     # alternative encoder

Differences from the legacy code (legacy/src): the whole encoder is fine-tuned
(freeze_layers=0) with lr 2e-5 and linear warmup, the best epoch is chosen on the
validation split by macro-F1, and the test split is scored once with that epoch.
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np

from .evaluate import run_dir, setting_name, write_run
from .runlog import load_config
from .train_classical import load_task, split_for


def set_seed(seed):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode(tok, texts, max_len):
    return tok(list(texts), truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")


def predict(model, loader, device):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for ids, mask, _ in loader:
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            out.append(logits.argmax(-1).cpu().numpy())
    return np.concatenate(out)


def train_one(cfg, df, label_names, setting, seed, model_name, model_tag):
    import torch
    from sklearn.metrics import f1_score
    from sklearn.utils.class_weight import compute_class_weight
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    b = cfg["bert"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = list(range(len(label_names)))
    inputs = cfg["classification"]["input_fields"]
    field = b.get("text_field")
    text = (df[field] if field else df[inputs].fillna("").agg(" ".join, axis=1)).fillna("").str.strip().values
    y = df["y"].values
    part = split_for(cfg, df, seed)
    idx = {s: np.where(part == s)[0] for s in ("train", "val", "test")}

    tok = AutoTokenizer.from_pretrained(model_name)

    def loader(split, shuffle):
        enc = encode(tok, text[idx[split]], b["max_len"])
        ds = TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(y[idx[split]]))
        g = torch.Generator().manual_seed(seed)
        return DataLoader(ds, batch_size=b["batch_size"], shuffle=shuffle, generator=g)

    tr, va, te = loader("train", True), loader("val", False), loader("test", False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels)).to(device)

    n_frozen = b.get("freeze_layers", 0)
    if n_frozen:
        base = getattr(model, model.base_model_prefix)
        for p in base.embeddings.parameters():
            p.requires_grad = False
        for layer in base.encoder.layer[:n_frozen]:
            for p in layer.parameters():
                p.requires_grad = False
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    weight = None
    if b.get("class_weighted_loss", True):
        cw = compute_class_weight("balanced", classes=np.array(labels), y=y[idx["train"]])
        weight = torch.tensor(cw, dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=b["learning_rate"], weight_decay=b["weight_decay"])
    steps = len(tr) * b["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(b["warmup_ratio"] * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    history, best, bad = [], None, 0
    t_start = time.perf_counter()
    for epoch in range(1, b["epochs"] + 1):
        model.train()
        losses = []
        for ids, mask, yy in tr:
            opt.zero_grad()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
                loss = loss_fn(logits.float(), yy.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        p_va = predict(model, va, device)
        f1 = f1_score(y[idx["val"]], p_va, labels=labels, average="macro", zero_division=0)
        acc = float((p_va == y[idx["val"]]).mean())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_accuracy": acc, "val_macro_f1": f1})
        print(f"{setting} {model_tag} seed{seed} epoch {epoch}: loss={np.mean(losses):.4f} "
              f"val_acc={acc:.4f} val_macroF1={f1:.4f}", flush=True)
        if best is None or f1 > best[0]:
            best, bad = (f1, epoch, copy.deepcopy(model.state_dict()), p_va), 0
        else:
            bad += 1
            if bad >= b["early_stopping_patience"]:
                break
    train_s = time.perf_counter() - t_start

    model.load_state_dict(best[2])
    t0 = time.perf_counter()
    p_te = predict(model, te, device)
    pred_ms = 1000 * (time.perf_counter() - t0) / len(idx["test"])
    extra = {"target": cfg["classification"]["target"], "inputs": inputs, "k": df.attrs.get("k"),
             "model_name": model_name, "bert_config": b, "text_field": field or "+".join(inputs), "best_epoch": best[1], "history": history,
             "trainable_parameters": int(n_trainable), "fit_seconds": train_s,
             "predict_ms_per_ticket": pred_ms, "device": str(device),
             "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
             "n_train": len(idx["train"]), "n_val": len(idx["val"]), "n_test": len(idx["test"])}
    rec = write_run(cfg, setting=setting, model=model_tag, seed=seed, labels=labels, label_names=label_names,
                    ids_test=df["ticket_id"].values[idx["test"]], y_test=y[idx["test"]], p_test=p_te,
                    y_val=y[idx["val"]], p_val=best[3], extra=extra,
                    groups_test=df["group"].values[idx["test"]],
                                groups_train=df["group"].values[idx["train"]])
    print(f"TEST {setting} {model_tag} seed{seed}: acc={rec['test']['accuracy']:.4f} "
          f"macroF1={rec['test']['macro_f1']:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--k", nargs="+", type=int, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--tag", default="bert_beto")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    target = cfg["classification"]["target"]
    model_name = a.model_name or cfg["bert"]["model_name"]
    for k in a.k or [cfg["clustering"]["selected_k"]]:
        df, label_names = load_task(cfg, target, k)
        df.attrs["k"] = k if target == "cluster" else None
        setting = setting_name(target, cfg["classification"]["input_fields"], k)
        for seed in a.seeds or cfg["split"]["seeds"]:
            if not a.force and (run_dir(cfg, setting, a.tag, seed) / "metrics.json").exists():
                print(f"skip {setting} {a.tag} seed{seed}")
                continue
            train_one(cfg, df, label_names, setting, seed, model_name, a.tag)


if __name__ == "__main__":
    main()
