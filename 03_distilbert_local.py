"""DistilBERT fine-tuning on the shared 50K SNLI subset. 1 epoch, lr 5e-5, seed 42, single run.
Run AFTER 02_classical_local.py (which creates train_50k_seed42.csv)."""
"""Local Mac/CPU/MPS-compatible DistilBERT fine-tuning script."""
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)

SEED = 42
N_TRAIN = 50_000
LABELS = ["entailment", "neutral", "contradiction"]   # 0, 1, 2
MODEL = "distilbert-base-uncased"
LR = 5e-5
EPOCHS = 1
MAX_LEN = 128
PER_DEVICE_BATCH = 8
GRAD_ACCUM = 8                                        # effective batch 64; smaller per-device batch is safer on Mac
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.1

PROJECT_DIR = Path(__file__).resolve().parent
CLEAN_DIR = PROJECT_DIR / "results" / "cleaned"
TRAIN_50K = CLEAN_DIR / "train_50k_seed42.csv"
MODEL_DIR = PROJECT_DIR / "models" / "distilbert"
CKPT_DIR = PROJECT_DIR / "checkpoints" / "distilbert"
OUT_DIR = PROJECT_DIR / "results"


def read_split(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run 02_classical_local.py first.")
    df = pd.read_csv(path, usecols=["premise", "hypothesis", "label"])
    df["premise"] = df["premise"].fillna("").astype(str)
    df["hypothesis"] = df["hypothesis"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    if not df["label"].isin([0, 1, 2]).all():
        raise ValueError(f"{path} has labels outside {{0,1,2}}")
    return df


def tokenize(df, tokenizer):
    ds = Dataset.from_dict({"premise": df["premise"].tolist(), "hypothesis": df["hypothesis"].tolist(),
                            "label": df["label"].tolist()})
    return ds.map(lambda b: tokenizer(b["premise"], b["hypothesis"], truncation=True, max_length=MAX_LEN),
                  batched=True, remove_columns=["premise", "hypothesis"])


def evaluate(y, pred):
    rep = classification_report(y, pred, target_names=LABELS, output_dict=True, digits=4)
    return {"accuracy": float(accuracy_score(y, pred)),
            "f1_macro": float(f1_score(y, pred, average="macro")),
            "f1_weighted": float(f1_score(y, pred, average="weighted")),
            "per_class": {l: {"precision": rep[l]["precision"], "recall": rep[l]["recall"],
                              "f1": rep[l]["f1-score"]} for l in LABELS}}


def main():
    for d in (MODEL_DIR, CKPT_DIR, OUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    train_df = read_split(TRAIN_50K)
    assert len(train_df) == N_TRAIN, f"{TRAIN_50K} has {len(train_df)} rows"
    val_df = read_split(CLEAN_DIR / "validation.csv")
    test_df = read_split(CLEAN_DIR / "test.csv")
    test_df.insert(0, "id", np.arange(len(test_df)))
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    train_ds, val_ds, test_ds = (tokenize(d, tokenizer) for d in (train_df, val_df, test_df))

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=3, id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    total_steps = math.ceil(len(train_ds) / (PER_DEVICE_BATCH * GRAD_ACCUM)) * EPOCHS
    args = TrainingArguments(
        output_dir=str(CKPT_DIR), num_train_epochs=EPOCHS, learning_rate=LR, weight_decay=WEIGHT_DECAY,
        per_device_train_batch_size=PER_DEVICE_BATCH, gradient_accumulation_steps=GRAD_ACCUM,
        per_device_eval_batch_size=64, lr_scheduler_type="linear", warmup_steps=int(WARMUP_FRAC * total_steps),
        bf16=use_bf16, fp16=torch.cuda.is_available() and not use_bf16,
        save_strategy="no", report_to="none", logging_steps=50, seed=SEED, data_seed=SEED)
    kw = dict(model=model, args=args, train_dataset=train_ds, data_collator=DataCollatorWithPadding(tokenizer))
    try:
        trainer = Trainer(processing_class=tokenizer, **kw)
    except TypeError:                                   # older transformers
        trainer = Trainer(tokenizer=tokenizer, **kw)

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    val_m = evaluate(np.asarray(val_ds["label"]), trainer.predict(val_ds).predictions.argmax(-1))
    t1 = time.time()
    pred = trainer.predict(test_ds).predictions.argmax(-1)
    infer_time = time.time() - t1
    test_m = evaluate(np.asarray(test_ds["label"]), pred)
    print(classification_report(test_df["label"], pred, target_names=LABELS, digits=4))

    out = test_df[["id", "premise", "hypothesis", "label"]].copy()
    out["pred"] = pred
    out.to_csv(OUT_DIR / "distilbert_test_predictions.csv", index=False)

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    results = {
        "model": MODEL, "train_file": str(TRAIN_50K), "n_train": len(train_df), "seed": SEED,
        "hyperparameters": {"lr": LR, "epochs": EPOCHS, "max_len": MAX_LEN, "per_device_batch": PER_DEVICE_BATCH,
                            "grad_accum": GRAD_ACCUM, "effective_batch": PER_DEVICE_BATCH * GRAD_ACCUM,
                            "weight_decay": WEIGHT_DECAY, "warmup_frac": WARMUP_FRAC, "scheduler": "linear",
                            "precision": "bf16" if use_bf16 else ("fp16" if torch.cuda.is_available() else "fp32")},
        "train_time_s": train_time, "test_inference_s": infer_time,
        "validation": val_m, "test": test_m,
    }
    with open(OUT_DIR / "distilbert_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Val acc {val_m['accuracy']:.4f} | Test acc {test_m['accuracy']:.4f} | "
          f"macro-F1 {test_m['f1_macro']:.4f} | train {train_time / 60:.1f} min")


if __name__ == "__main__":
    main()
