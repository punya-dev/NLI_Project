"""
demo/display.py
---------------
All terminal rendering: colours, boxes, tables, banners.
No model logic here — pure presentation.
"""

from __future__ import annotations

import os
import shutil
import textwrap
from typing import Dict, List, Optional

from demo.inference import (
    LABEL_DESCRIPTIONS,
    LABEL_EMOJI,
    LABELS,
    PredictionResult,
)

# ── ANSI colour helpers ───────────────────────────────────────────────────────
_USE_COLOR = os.isatty(1)   # disable when piped / redirected


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def bold(t: str)   -> str: return _c("1", t)
def dim(t: str)    -> str: return _c("2", t)
def green(t: str)  -> str: return _c("32", t)
def cyan(t: str)   -> str: return _c("36", t)
def yellow(t: str) -> str: return _c("33", t)
def red(t: str)    -> str: return _c("31", t)
def blue(t: str)   -> str: return _c("34", t)
def magenta(t: str)-> str: return _c("35", t)


def label_color(label: str) -> str:
    return {
        "entailment":    green,
        "neutral":       blue,
        "contradiction": red,
    }.get(label, cyan)(label.upper())


# ── layout helpers ────────────────────────────────────────────────────────────
def terminal_width() -> int:
    return min(shutil.get_terminal_size((80, 24)).columns, 100)


def rule(char: str = "─") -> str:
    return dim(char * terminal_width())


def section(title: str) -> str:
    w = terminal_width()
    pad = max(0, (w - len(title) - 4) // 2)
    return f"\n{dim('─' * pad)}  {bold(title)}  {dim('─' * pad)}"


def banner() -> str:
    lines = [
        "",
        bold(cyan("  ╔══════════════════════════════════════════╗")),
        bold(cyan("  ║           NLI  MODEL  LAB                ║")),
        bold(cyan("  ║  SNLI · Natural Language Inference        ║")),
        bold(cyan("  ╚══════════════════════════════════════════╝")),
        dim("  Compare classical ML with a fine-tuned Transformer"),
        "",
    ]
    return "\n".join(lines)


def model_status_table(status: Dict[str, bool]) -> str:
    rows = [
        ("1", "Logistic Regression", "logreg"),
        ("2", "LinearSVC",           "linearsvc"),
        ("3", "LightGBM",            "lightgbm"),
        ("4", "DistilBERT",          "distilbert"),
    ]
    lines = [bold("  Available Models"), ""]
    for num, display, key in rows:
        ok_key = key if key != "logreg" else "logreg"
        ready = status.get(ok_key, False) and status.get("feature_components", False) \
            if key != "distilbert" else status.get("distilbert", False)
        dot = green("●") if ready else red("○")
        lines.append(f"  [{num}] {dot}  {display}")
    lines += [
        "",
        dim("  ● = artifact found   ○ = not yet trained"),
    ]
    return "\n".join(lines)


def main_menu() -> str:
    items = [
        ("1", "Predict with one model"),
        ("2", "Compare all models on the same input"),
        ("3", "View evaluation results"),
        ("4", "Explore misclassified examples"),
        ("5", "Dataset and preprocessing information"),
        ("6", "Training information"),
        ("7", "Check artifact status"),
        ("8", "Exit"),
    ]
    lines = [section("MAIN MENU"), ""]
    for key, desc in items:
        lines.append(f"  {bold(key)}.  {desc}")
    lines.append("")
    return "\n".join(lines)


# ── prediction card ───────────────────────────────────────────────────────────
def prediction_card(
    premise: str,
    hypothesis: str,
    result: PredictionResult,
) -> str:
    w = terminal_width()
    lines = [section("PREDICTION RESULT"), ""]

    if not result.ok:
        lines.append(f"  {red('Error:')} {result.error}")
        return "\n".join(lines)

    lbl = result.predicted_label
    lines += [
        f"  {bold('Model:')}      {result.model_name}",
        f"  {bold('Premise:')}    {textwrap.shorten(premise,   w - 16)}",
        f"  {bold('Hypothesis:')} {textwrap.shorten(hypothesis, w - 16)}",
        "",
        f"  Prediction: {LABEL_EMOJI[lbl]}  {label_color(lbl)}",
        f"  {dim(LABEL_DESCRIPTIONS[lbl])}",
        "",
    ]

    # scores
    score_header = {
        "probability":    "Class Probabilities",
        "decision_score": "Decision Scores  (not probabilities)",
        "softmax":        "Softmax Probabilities",
    }.get(result.score_type, "Scores")
    lines.append(f"  {bold(score_header)}")
    for label in LABELS:
        val  = result.scores.get(label, 0.0)
        mark = " ◀" if label == lbl else ""
        bar  = _prob_bar(val) if result.score_type != "decision_score" else ""
        lines.append(f"    {label:<14} {val:+.4f}  {bar}{dim(mark)}")

    lines += [
        "",
        dim(f"  Inference: {result.infer_time_s * 1000:.1f} ms"
            + (f"  (load: {result.load_time_s * 1000:.0f} ms)" if result.load_time_s > 0.01 else "")),
        "",
    ]
    return "\n".join(lines)


def _prob_bar(p: float, width: int = 20) -> str:
    filled = max(0, min(width, round(p * width)))
    return dim("[") + green("█" * filled) + dim("░" * (width - filled)) + dim("]")


# ── comparison table ──────────────────────────────────────────────────────────
def comparison_table(
    premise: str,
    hypothesis: str,
    results: List[PredictionResult],
) -> str:
    w = terminal_width()
    lines = [section("MODEL COMPARISON"), ""]
    lines += [
        f"  {bold('Premise:')}    {textwrap.shorten(premise,    w - 16)}",
        f"  {bold('Hypothesis:')} {textwrap.shorten(hypothesis, w - 16)}",
        "",
        bold(f"  {'Model':<22} {'Prediction':<16} {'Scores (Ent / Neu / Con)':>30}  {'ms':>6}"),
        "  " + "─" * (w - 4),
    ]
    for r in results:
        if not r.ok:
            lines.append(f"  {r.model_name:<22} {red('unavailable')}")
            continue
        lbl = r.predicted_label
        col = label_color(lbl)
        if r.score_type == "decision_score":
            score_str = "  ".join(f"{r.scores.get(l, 0):+.2f}" for l in LABELS)
            score_label = "decision"
        else:
            score_str = "  ".join(f"{r.scores.get(l, 0):.3f}" for l in LABELS)
            score_label = "prob"
        ms = f"{r.infer_time_s * 1000:.0f}"
        lines.append(
            f"  {r.model_name:<22} {col:<16}  {dim(score_str)}  {dim(ms):>6}"
        )

    # agreement summary
    valid = [r for r in results if r.ok]
    if len(valid) > 1:
        preds = [r.predicted_label for r in valid]
        if len(set(preds)) == 1:
            lines.append(f"\n  {green('✓ All models agree:')} {label_color(preds[0])}")
        else:
            lines.append(f"\n  {yellow('△ Models disagree')} — see individual scores above")

    lines.append("")
    return "\n".join(lines)


# ── evaluation summary ────────────────────────────────────────────────────────
def eval_summary(model_label: str, split: str, metrics: dict) -> str:
    pc = metrics.get("per_class", {})
    lines = [
        "",
        bold(f"  {model_label}  —  {split.upper()} SET"),
        "",
        f"  {'Accuracy:':<14} {metrics.get('accuracy', 0):.4f}",
        f"  {'Macro-F1:':<14} {metrics.get('f1_macro', 0):.4f}",
        f"  {'Weighted-F1:':<14} {metrics.get('f1_weighted', 0):.4f}",
        "",
        bold("  Per-class metrics"),
        f"  {'Class':<14} {'Precision':>10} {'Recall':>10} {'F1':>10}",
        "  " + "─" * 46,
    ]
    for label in LABELS:
        m = pc.get(label, {})
        lines.append(
            f"  {label:<14} {m.get('precision', 0):>10.4f}"
            f" {m.get('recall', 0):>10.4f} {m.get('f1', 0):>10.4f}"
        )
    return "\n".join(lines)


# ── error card ────────────────────────────────────────────────────────────────
def error_card(row: dict, idx: int, total: int) -> str:
    w = terminal_width()
    gold_lbl  = LABELS[row["label"]]
    pred_lbl  = LABELS[row["pred"]]
    lines = [
        "",
        bold(f"  Error Example {idx}/{total}"),
        "  " + "─" * min(w - 4, 60),
        f"  {bold('Premise:')}    {textwrap.fill(str(row['premise']),    w - 16, subsequent_indent=' ' * 16)}",
        f"  {bold('Hypothesis:')} {textwrap.fill(str(row['hypothesis']), w - 16, subsequent_indent=' ' * 16)}",
        "",
        f"  Gold label:  {LABEL_EMOJI[gold_lbl]}  {label_color(gold_lbl)}",
        f"  Predicted:   {LABEL_EMOJI[pred_lbl]}  {label_color(pred_lbl)}",
        "",
    ]
    return "\n".join(lines)


# ── dataset / preprocessing info ─────────────────────────────────────────────
PREPROCESSING_TEXT = """
  DATASET  —  Stanford Natural Language Inference (SNLI)
  ──────────────────────────────────────────────────────
  · 570 K manually annotated English sentence pairs.
  · Three labels: Entailment (0), Neutral (1), Contradiction (2).
  · Invalid '-' gold labels removed during EDA (01_eda_local.py).
  · A stratified 50 K subset (seed 42) is the shared training set for
    all models in this project.
  · Validation and test splits are the standard SNLI splits.

  CLASSICAL FEATURE ENGINEERING  (02_classical_local.py)
  ──────────────────────────────────────────────────────
  Two feature matrices are built per split:

  LINEAR matrix  (LogReg + LinearSVC)
    • TF-IDF unigrams+bigrams on premise / hypothesis (max 50 K features).
    • Element-wise difference  |TF-IDF_premise − TF-IDF_hypothesis|.
    • Cosine similarity between premise and hypothesis TF-IDF vectors.
    • Novel-word TF-IDF: tokens in hypothesis absent from premise.
    • Cross-pair hash features: hashed dropped×novel token combinations.
    • 14 dense numeric features (scaled): token counts, length ratio,
      Jaccard / Dice overlap, bigram overlap, negation indicators,
      hypothesis-subsumes-premise flag.

  TREE matrix  (LightGBM)
    • Dense numeric features (unscaled).
    • Cosine similarity.
    • Binary BOW for hypothesis tokens (top 3 K).
    • Binary BOW for novel tokens (top 3 K).

  DISTILBERT TOKENISATION  (03_distilbert_local.py)
  ──────────────────────────────────────────────────
  · AutoTokenizer from 'distilbert-base-uncased'.
  · Premise and hypothesis concatenated with [SEP] separator.
  · Truncated to max_length=128 tokens.
  · Fine-tuned for 1 epoch; no hand-crafted features needed.
"""

TRAINING_CONTEXT = """
  TRAINING OVERVIEW
  ─────────────────
  Classical models        02_classical_local.py
    Shared training set:  50,000 pairs (stratified, seed 42)
    LogReg   C=1.0,  solver=lbfgs, max_iter=1000
    LinearSVC C=0.1, squared hinge loss, dual=False
    LightGBM  n_estimators=1500, lr=0.08, num_leaves=63
               early stopping = 50 rounds (on validation)
    Selection: best validation accuracy → Logistic Regression

  DistilBERT fine-tuning  03_distilbert_local.py
    Base model:           distilbert-base-uncased
    Epochs:               1
    Learning rate:        5 × 10⁻⁵  (linear schedule, 10 % warmup)
    Effective batch:      64 (per-device 8 × grad accum 8)
    Max length:           128 tokens
    Weight decay:         0.01
    Seed:                 42
"""
