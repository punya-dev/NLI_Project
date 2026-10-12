"""
demo/results.py
---------------
Loads saved experiment JSON files and prediction CSVs.
Generates metrics tables, confusion matrices, and error examples.
Never retrains; reads only from saved artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
LABELS      = ["entailment", "neutral", "contradiction"]


# ── JSON loaders ──────────────────────────────────────────────────────────────
def load_classical_results() -> Optional[dict]:
    path = RESULTS_DIR / "classical_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_distilbert_results() -> Optional[dict]:
    path = RESULTS_DIR / "distilbert_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ── CSV loaders ───────────────────────────────────────────────────────────────
def load_predictions(model_key: str) -> Optional[pd.DataFrame]:
    """model_key: 'classical' or 'distilbert'"""
    path = RESULTS_DIR / f"{model_key}_test_predictions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # ensure integer columns
    for col in ("label", "pred"):
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df


# ── confusion matrix ──────────────────────────────────────────────────────────
def confusion_matrix_counts(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return 3×3 confusion matrix (rows=true, cols=pred)."""
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_true, y_pred, labels=[0, 1, 2])


def format_confusion_matrix(cm: np.ndarray) -> str:
    """Render a 3×3 confusion matrix as ASCII."""
    col_w = 14
    hdr = " " * 18 + "".join(f"PRED:{l[:3].upper():^{col_w}}" for l in LABELS)
    lines = [hdr, "-" * (18 + col_w * 3)]
    for i, label in enumerate(LABELS):
        row = f"TRUE:{label[:3].upper():>12}  "
        row += "".join(f"{cm[i, j]:^{col_w}}" for j in range(3))
        lines.append(row)
    return "\n".join(lines)


# ── per-class metrics table ───────────────────────────────────────────────────
def format_per_class(per_class: dict) -> str:
    lines = [
        f"  {'Class':<14} {'Precision':>10} {'Recall':>10} {'F1':>10}",
        "  " + "-" * 46,
    ]
    for label in LABELS:
        m = per_class[label]
        lines.append(
            f"  {label:<14} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}"
        )
    return "\n".join(lines)


# ── error examples ────────────────────────────────────────────────────────────
def get_errors(
    df: pd.DataFrame,
    gold_filter: Optional[int] = None,
    pred_filter: Optional[int] = None,
    n: int = 10,
    shuffle: bool = True,
) -> pd.DataFrame:
    """
    Return rows where label != pred, optionally filtered by gold or predicted class.
    """
    errors = df[df["label"] != df["pred"]].copy()
    if gold_filter is not None:
        errors = errors[errors["label"] == gold_filter]
    if pred_filter is not None:
        errors = errors[errors["pred"] == pred_filter]
    if shuffle:
        errors = errors.sample(frac=1, random_state=0)
    return errors.head(n)


def get_disagreements(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    n: int = 10,
    shuffle: bool = True,
) -> pd.DataFrame:
    """
    Return test examples where model_a and model_b disagree on prediction.
    Both dataframes must cover the same rows (same 'id' column).
    """
    merged = df_a[["id", "premise", "hypothesis", "label", "pred"]].merge(
        df_b[["id", "pred"]].rename(columns={"pred": "pred_b"}),
        on="id",
    )
    merged = merged[merged["pred"] != merged["pred_b"]].rename(
        columns={"pred": "pred_a"}
    )
    if shuffle:
        merged = merged.sample(frac=1, random_state=0)
    return merged.head(n)


# ── summary helpers ───────────────────────────────────────────────────────────
def error_rate_by_class(df: pd.DataFrame) -> Dict[str, float]:
    rates = {}
    for i, label in enumerate(LABELS):
        sub = df[df["label"] == i]
        if len(sub) == 0:
            rates[label] = 0.0
        else:
            rates[label] = (sub["pred"] != sub["label"]).mean()
    return rates


def dataset_stats() -> dict:
    """Basic stats about the cleaned SNLI splits."""
    stats = {}
    clean_dir = RESULTS_DIR / "cleaned"
    for split in ("train_50k_seed42", "validation", "test"):
        path = clean_dir / f"{split}.csv"
        if path.exists():
            df = pd.read_csv(path, usecols=["label"])
            counts = df["label"].value_counts().sort_index().to_dict()
            stats[split] = {
                "total": len(df),
                "entailment":    counts.get(0, 0),
                "neutral":       counts.get(1, 0),
                "contradiction": counts.get(2, 0),
            }
    return stats
