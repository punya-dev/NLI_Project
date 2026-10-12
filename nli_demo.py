#!/usr/bin/env python3
"""
nli_demo.py  —  NLI Model Lab
==============================
Interactive CLI for the SNLI Natural Language Inference project.

Usage
-----
  python nli_demo.py            # interactive menu
  python nli_demo.py predict    # single-model prediction (prompts for input)
  python nli_demo.py compare    # compare all models (prompts for input)
  python nli_demo.py results    # show saved evaluation results
  python nli_demo.py errors     # explore misclassified examples
  python nli_demo.py info       # dataset + preprocessing information
  python nli_demo.py training   # show training configuration
  python nli_demo.py status     # check saved model artifacts
  python nli_demo.py train      # run the full training pipeline (time-consuming)

All inference is done from locally saved model artifacts.
No network access or model download occurs during prediction.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Optional

# ── ensure project root is on path ───────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import warnings
warnings.filterwarnings(
    "ignore",
    message="Trying to unpickle estimator",
    category=UserWarning,
)

from demo import display, results as res_mod
from demo.display import (
    banner, bold, cyan, dim, green, label_color, magenta,
    main_menu, model_status_table, red, rule, section, yellow,
)
from demo.inference import (
    LABEL_DESCRIPTIONS, LABEL_EMOJI, LABELS, MODEL_DISPLAY,
    MODEL_NAMES, PREDICT_FN, registry,
)

# ── prepared example pairs ────────────────────────────────────────────────────
EXAMPLES = [
    {
        "label": "entailment (expected)",
        "premise":    "A man is playing guitar on stage.",
        "hypothesis": "A person is performing music.",
    },
    {
        "label": "contradiction (expected)",
        "premise":    "The children are sleeping quietly in bed.",
        "hypothesis": "The kids are playing noisily outside.",
    },
    {
        "label": "neutral (expected)",
        "premise":    "A woman is reading a book in the park.",
        "hypothesis": "The woman enjoys fiction novels.",
    },
]


# ── input helpers ─────────────────────────────────────────────────────────────
def _prompt(prompt_text: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"  {prompt_text}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise
    return value if value else (default or "")


def _choose(prompt_text: str, options: list[str], default: str = "1") -> str:
    value = _prompt(prompt_text, default)
    while value not in options:
        print(f"  {red('Please enter one of:')} {', '.join(options)}")
        value = _prompt(prompt_text, default)
    return value


def _get_sentence_pair() -> tuple[str, str]:
    """Ask for premise + hypothesis, offering example shortcuts."""
    print()
    print(bold("  Sentence pair input"))
    print(dim("  Enter 'e' for prepared examples, or type your own text.\n"))

    choice = _prompt("Use prepared examples? (y/n)", "n").lower()
    if choice in ("y", "yes"):
        print()
        for i, ex in enumerate(EXAMPLES, 1):
            print(f"  [{i}] {dim(ex['label'])}")
            print(f"       P: {ex['premise']}")
            print(f"       H: {ex['hypothesis']}")
        n = _choose("\n  Select example", [str(i) for i in range(1, len(EXAMPLES) + 1)])
        ex = EXAMPLES[int(n) - 1]
        print(f"\n  {bold('Premise:')}    {ex['premise']}")
        print(f"  {bold('Hypothesis:')} {ex['hypothesis']}")
        return ex["premise"], ex["hypothesis"]

    premise = _prompt("Premise")
    while not premise:
        print(f"  {red('Premise cannot be empty.')}")
        premise = _prompt("Premise")
    hypothesis = _prompt("Hypothesis")
    while not hypothesis:
        print(f"  {red('Hypothesis cannot be empty.')}")
        hypothesis = _prompt("Hypothesis")
    return premise, hypothesis


def _get_model_key() -> str:
    print()
    print(bold("  Select model"))
    for k, v in MODEL_DISPLAY.items():
        print(f"  [{list(MODEL_NAMES.keys())[list(MODEL_NAMES.values()).index(k)]}] {v}")
    num = _choose("\n  Enter number", list(MODEL_NAMES.keys()))
    return MODEL_NAMES[num]


# ── menu actions ──────────────────────────────────────────────────────────────
def action_predict():
    premise, hypothesis = _get_sentence_pair()
    model_key = _get_model_key()
    print(f"\n  {dim('Running')} {MODEL_DISPLAY[model_key]}{dim('...')}")
    result = PREDICT_FN[model_key](premise, hypothesis)
    print(display.prediction_card(premise, hypothesis, result))


def action_compare():
    premise, hypothesis = _get_sentence_pair()
    print(f"\n  {dim('Running all four models...')}")
    results_list = []
    for key in ("logreg", "linearsvc", "lightgbm", "distilbert"):
        print(f"    {dim(MODEL_DISPLAY[key])}…", end=" ", flush=True)
        r = PREDICT_FN[key](premise, hypothesis)
        results_list.append(r)
        if r.ok:
            print(label_color(r.predicted_label))
        else:
            print(red("unavailable"))
    print(display.comparison_table(premise, hypothesis, results_list))


def action_results():
    print(section("EVALUATION RESULTS"))

    # ── Classical ─────────────────────────────────────────────────────────────
    cr = res_mod.load_classical_results()
    if cr is None:
        print(f"\n  {red('classical_results.json not found.')} Run 02_classical_local.py first.")
    else:
        selected = cr.get("selected", "logreg")
        n_train  = cr.get("n_train", "?")
        runtime  = cr.get("total_runtime_s", 0)

        print(f"\n  {bold('Classical models')} — trained on {n_train:,} examples  "
              f"| total runtime {runtime / 60:.1f} min")
        print(f"  Selected model (by val accuracy): {bold(selected.upper())}")

        # validation — all models
        print(bold("\n  Validation accuracy — all classifiers"))
        print(f"  {'Model':<14} {'Accuracy':>10} {'Macro-F1':>10} {'Train time':>12}")
        print("  " + "─" * 50)
        val_all   = cr.get("validation_all_models", {})
        train_t   = cr.get("train_time_s", {})
        for name in ("logreg", "linearsvc", "lightgbm"):
            m = val_all.get(name, {})
            t = train_t.get(name, 0)
            marker = "  ← selected" if name == selected else ""
            print(
                f"  {name:<14} {m.get('accuracy', 0):>10.4f}"
                f" {m.get('f1_macro', 0):>10.4f}"
                f" {t:>10.0f}s{dim(marker)}"
            )

        # test — selected model
        print(display.eval_summary(
            f"Classical ({selected.upper()}) — best model",
            "test",
            cr.get("test", {}),
        ))

        # confusion matrix from CSV
        df_cls = res_mod.load_predictions("classical")
        if df_cls is not None:
            cm = res_mod.confusion_matrix_counts(
                df_cls["label"].to_numpy(), df_cls["pred"].to_numpy()
            )
            print(f"\n  {bold('Confusion Matrix')} — Classical ({selected.upper()}) test set")
            print()
            print(res_mod.format_confusion_matrix(cm))

    # ── DistilBERT ────────────────────────────────────────────────────────────
    dr = res_mod.load_distilbert_results()
    if dr is None:
        print(f"\n  {red('distilbert_results.json not found.')} "
              "Run 03_distilbert_local.py first.")
    else:
        train_t = dr.get("train_time_s", 0)
        infer_t = dr.get("test_inference_s", 0)
        print(f"\n  {bold('DistilBERT')}  — fine-tuned | train {train_t / 60:.1f} min"
              f" | test inference {infer_t:.1f}s")
        print(display.eval_summary("DistilBERT", "validation", dr.get("validation", {})))
        print(display.eval_summary("DistilBERT", "test", dr.get("test", {})))

        df_db = res_mod.load_predictions("distilbert")
        if df_db is not None:
            cm = res_mod.confusion_matrix_counts(
                df_db["label"].to_numpy(), df_db["pred"].to_numpy()
            )
            print(f"\n  {bold('Confusion Matrix')} — DistilBERT test set")
            print()
            print(res_mod.format_confusion_matrix(cm))

    print()


def action_errors():
    print(section("MISCLASSIFIED EXAMPLES"))
    print()

    # choose which model's predictions to use
    avail = []
    df_cls = res_mod.load_predictions("classical")
    df_db  = res_mod.load_predictions("distilbert")
    if df_cls is not None:
        avail.append(("c", "Classical (selected model)", df_cls))
    if df_db is not None:
        avail.append(("d", "DistilBERT", df_db))
    if not avail:
        print(f"  {red('No prediction CSVs found.')} Run the training scripts first.")
        return

    for code, label, _ in avail:
        print(f"  [{code}] {label}")
    if df_cls is not None and df_db is not None:
        print(f"  [x] Disagreements between Classical and DistilBERT")

    code = _choose("\n  Select source", [a[0] for a in avail] + (["x"] if len(avail) == 2 else []))

    if code == "x":
        df = res_mod.get_disagreements(df_cls, df_db, n=20)
        if df.empty:
            print(f"  {green('Both models agree on every test example!')} ")
            return
        n = min(len(df), 10)
        print(f"\n  Showing {n} of {len(df)} disagreements\n")
        for i, (_, row) in enumerate(df.head(n).iterrows(), 1):
            pred_a = LABELS[row["pred_a"]]
            pred_b = LABELS[row["pred_b"]]
            gold   = LABELS[row["label"]]
            print(bold(f"  Disagreement {i}/{n}"))
            print(f"  Premise:    {textwrap.shorten(str(row['premise']),    80)}")
            print(f"  Hypothesis: {textwrap.shorten(str(row['hypothesis']), 80)}")
            print(f"  Gold:           {label_color(gold)}")
            print(f"  Classical pred: {label_color(pred_a)}")
            print(f"  DistilBERT pred:{label_color(pred_b)}")
            print()
        return

    df = avail[0][2]  # default; overwritten by the loop below
    for c, _lbl, d in avail:
        if c == code:
            df = d
            break

    # filter options
    print()
    print(bold("  Filter by gold class"))
    print("  [0] All errors")
    for i, l in enumerate(LABELS, 1):
        print(f"  [{i}] Gold = {l}")
    gold_choice = _choose("  Select filter", ["0", "1", "2", "3"], "0")
    gold_filter = int(gold_choice) - 1 if gold_choice != "0" else None

    errors = res_mod.get_errors(df, gold_filter=gold_filter, n=20)
    if errors.empty:
        print(f"  {green('No errors found for that filter.')}")
        return

    n_show = min(10, len(errors))
    print(f"\n  Showing {n_show} of {len(errors)} errors\n")

    # error-rate by class
    rates = res_mod.error_rate_by_class(df)
    print(bold("  Error rate by gold class:"))
    for l, r in rates.items():
        bar = display._prob_bar(r, width=15)
        print(f"    {l:<14} {r:.1%}  {bar}")
    print()

    for i, (_, row) in enumerate(errors.iterrows(), 1):
        print(display.error_card(row.to_dict(), i, n_show))
        if i < n_show:
            cont = _prompt("  Press Enter for next, or 'q' to stop", "").lower()
            if cont == "q":
                break


def action_info():
    print(section("DATASET AND PREPROCESSING"))
    print(display.PREPROCESSING_TEXT)

    # live stats if data is present
    stats = res_mod.dataset_stats()
    if stats:
        print(bold("  Split sizes used in this project"))
        print(f"  {'Split':<22} {'Total':>8} {'Entail':>8} {'Neutral':>8} {'Contra':>8}")
        print("  " + "─" * 58)
        split_labels = {
            "train_50k_seed42": "Train (50K subset)",
            "validation":       "Validation",
            "test":             "Test",
        }
        for key, lbl in split_labels.items():
            s = stats.get(key, {})
            if s:
                print(
                    f"  {lbl:<22} {s['total']:>8,}"
                    f" {s['entailment']:>8,} {s['neutral']:>8,}"
                    f" {s['contradiction']:>8,}"
                )
    print()


def action_training():
    print(section("TRAINING INFORMATION"))
    print(display.TRAINING_CONTEXT)

    cr = res_mod.load_classical_results()
    if cr:
        print(bold("  Classical results from classical_results.json"))
        hp = cr.get("hyperparameters", {})
        print(f"    LogReg  C={hp.get('logreg_C')}  LinearSVC C={hp.get('linearsvc_C')}")
        lgbm = hp.get("lightgbm", {})
        print(f"    LightGBM n_estimators={lgbm.get('n_estimators')} "
              f"lr={lgbm.get('learning_rate')} leaves={lgbm.get('num_leaves')}")

    dr = res_mod.load_distilbert_results()
    if dr:
        print(bold("\n  DistilBERT results from distilbert_results.json"))
        hp = dr.get("hyperparameters", {})
        print(f"    lr={hp.get('lr')}  epochs={hp.get('epochs')}  "
              f"max_len={hp.get('max_len')}  eff_batch={hp.get('effective_batch')}")
        print(f"    precision={hp.get('precision')}  weight_decay={hp.get('weight_decay')}")
        print(f"    train_time={dr.get('train_time_s', 0) / 60:.1f} min  "
              f"test_infer={dr.get('test_inference_s', 0):.1f}s")

    print()
    print(dim("  To re-run training from scratch:"))
    print(dim("    python 01_eda_local.py"))
    print(dim("    python 02_classical_local.py"))
    print(dim("    python 03_distilbert_local.py"))
    print()


def action_status():
    print(section("ARTIFACT STATUS"))
    status = registry.artifact_status()
    print()

    checks = [
        ("feature_components", "Classical feature_components.pkl"),
        ("logreg",             "logreg_model.pkl"),
        ("linearsvc",          "linearsvc_model.pkl"),
        ("lightgbm",           "lightgbm_model.pkl"),
        ("distilbert",         "DistilBERT model directory (non-empty)"),
    ]
    for key, label in checks:
        ok  = status.get(key, False)
        dot = green("●  FOUND") if ok else red("○  MISSING")
        print(f"  {dot}   {label}")

    # result files
    print()
    print(bold("  Result files"))
    for fname in (
        "classical_results.json",
        "distilbert_results.json",
        "classical_test_predictions.csv",
        "distilbert_test_predictions.csv",
    ):
        path = PROJECT_DIR / "results" / fname
        ok   = path.exists()
        dot  = green("●  FOUND") if ok else red("○  MISSING")
        size = f"({path.stat().st_size / 1024:.0f} KB)" if ok else ""
        print(f"  {dot}   {fname}  {dim(size)}")
    print()


def action_train():
    print(section("TRAINING PIPELINE"))
    print(f"""
  {yellow('⚠  This will run all training scripts in order.')}
  Expected time: several minutes to hours depending on hardware.

  Scripts:
    1. python 01_eda_local.py          (EDA + data cleaning)
    2. python 02_classical_local.py    (LogReg / LinearSVC / LightGBM)
    3. python 03_distilbert_local.py   (DistilBERT fine-tuning — slow on CPU)
""")
    confirm = _prompt("Start training pipeline? (yes/no)", "no").lower()
    if confirm not in ("yes", "y"):
        print(f"  {dim('Cancelled.')}")
        return

    scripts = [
        "01_eda_local.py",
        "02_classical_local.py",
        "03_distilbert_local.py",
    ]
    for script in scripts:
        path = PROJECT_DIR / script
        if not path.exists():
            print(f"  {red('Missing:')} {script}")
            continue
        print(f"\n  {bold('Running')} {script}  {dim('...')}")
        print(rule())
        try:
            subprocess.run(
                [sys.executable, str(path)],
                cwd=str(PROJECT_DIR),
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"\n  {red('Script failed:')} {e}")
            break
        print(rule())
    print(f"\n  {green('Training pipeline completed.')}\n")


# ── interactive loop ──────────────────────────────────────────────────────────
def interactive_loop():
    print(banner())
    status = registry.artifact_status()
    print(model_status_table(status))

    while True:
        print(main_menu())
        try:
            choice = _prompt("Enter option", "1")
        except KeyboardInterrupt:
            print(f"\n\n  {dim('Use option 8 to exit cleanly.')}\n")
            continue

        print()
        try:
            if choice == "1":
                action_predict()
            elif choice == "2":
                action_compare()
            elif choice == "3":
                action_results()
            elif choice == "4":
                action_errors()
            elif choice == "5":
                action_info()
            elif choice == "6":
                action_training()
            elif choice == "7":
                action_status()
            elif choice == "8":
                print(f"  {cyan('Goodbye!')}\n")
                break
            else:
                print(f"  {red('Invalid option.')} Please enter 1–8.")
        except KeyboardInterrupt:
            print(f"\n  {dim('Interrupted. Returning to menu...')}\n")
        except Exception as exc:
            print(f"\n  {red('Unexpected error:')} {exc}")
            import traceback
            print(dim(textwrap.indent(traceback.format_exc(), "  ")))

        input(f"\n  {dim('Press Enter to continue...')}")


# ── CLI dispatch ──────────────────────────────────────────────────────────────
_COMMANDS = {
    "predict":  action_predict,
    "compare":  action_compare,
    "results":  action_results,
    "errors":   action_errors,
    "info":     action_info,
    "training": action_training,
    "status":   action_status,
    "train":    action_train,
}


def main():
    args = sys.argv[1:]
    if not args:
        interactive_loop()
        return

    cmd = args[0].lower()
    if cmd in _COMMANDS:
        print(banner())
        _COMMANDS[cmd]()
    elif cmd in ("--help", "-h", "help"):
        print(__doc__)
    else:
        print(f"Unknown command: {cmd!r}")
        print(f"Available: {', '.join(_COMMANDS)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
