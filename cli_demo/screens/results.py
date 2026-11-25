"""Screen 2 — 📊 Benchmark Results: pre-computed accuracy/F1 tables."""
import questionary
from rich.table import Table
from rich import box

from ui.console import console
from ui.layout import print_header, pause
from data.loader import load_result, load_main_results, load_gap_closure

DATASETS = {
    "snli_test": "SNLI Test",
    "snli_val":  "SNLI Validation",
    "mnli_m":    "MNLI Matched",
    "mnli_mm":   "MNLI Mismatched",
}

MODEL_IDS = ["lr", "lr_hyponly", "distilbert", "distilbert_kd", "distilbert_kd_extra", "llama"]
MODEL_DISPLAY = {
    "lr":                   "LR",
    "lr_hyponly":           "LR (hyp-only baseline)",
    "distilbert":           "DistilBERT baseline",
    "distilbert_kd":        "DB + KD (generic pairs)",
    "distilbert_kd_extra":  "DB + KD + Extra SNLI",
    "llama":                "Llama-1B LoRA  ★ BEST",
}
BEST_MODEL = "llama"


def _show_main_table(dataset_key: str, metric: str):
    dataset_name = DATASETS[dataset_key]
    metric_label = "Accuracy" if metric == "accuracy" else "F1 Macro"

    print_header(f"📊  Benchmark Results — {dataset_name}")
    console.print(f"  Metric: [bold cyan]{metric_label}[/]\n")

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        padding=(0, 2),
        min_width=70,
    )
    table.add_column("Model",         min_width=32)
    table.add_column(metric_label,    min_width=10, justify="right")
    table.add_column("Entailment",    min_width=12, justify="right")
    table.add_column("Neutral",       min_width=10, justify="right")
    table.add_column("Contradiction", min_width=14, justify="right")

    best_overall = 0.0
    for mid in MODEL_IDS:
        try:
            data    = load_result(mid, dataset_key)
            metrics = data["metrics"]
        except Exception:
            continue

        overall = metrics["accuracy"] if metric == "accuracy" else metrics["f1_macro"]
        pc      = metrics.get("per_class", {})
        e_val   = pc.get("entailment",    {}).get("recall" if metric == "accuracy" else "f1", 0.0)
        n_val   = pc.get("neutral",        {}).get("recall" if metric == "accuracy" else "f1", 0.0)
        c_val   = pc.get("contradiction",  {}).get("recall" if metric == "accuracy" else "f1", 0.0)

        if overall > best_overall:
            best_overall = overall

        is_best = mid == BEST_MODEL
        name    = MODEL_DISPLAY[mid]
        name_str = f"[bold green]{name}[/]" if is_best else name
        overall_str = f"[bold green]{overall*100:.2f}%[/]" if is_best else f"{overall*100:.2f}%"

        table.add_row(
            name_str,
            overall_str,
            f"{e_val*100:.1f}%" if e_val else "—",
            f"{n_val*100:.1f}%" if n_val else "—",
            f"{c_val*100:.1f}%" if c_val else "—",
        )

    console.print(table)

    # Gap closure footnote
    if dataset_key in ("snli_test", "mnli_m", "mnli_mm"):
        try:
            gap_df = load_gap_closure()
            col = f"{dataset_key}_closed_%"
            if col in gap_df.columns:
                console.print("\n  [dim]Gap closure vs. LR→Llama baseline:[/]")
                parts = []
                for _, row in gap_df.iterrows():
                    parts.append(f"  [cyan]{row['model']}[/]: {row[col]:.1f}%")
                console.print("  " + "   │   ".join(p.strip() for p in parts[:3]))
        except Exception:
            pass

    console.print()


def _show_per_class_detail(dataset_key: str):
    """Show full precision/recall/F1 per class for a selected model."""
    choices = [questionary.Choice(MODEL_DISPLAY[m], value=m) for m in MODEL_IDS]
    choices.append(questionary.Choice("← Back", value="back"))
    mid = questionary.select("Select model for per-class breakdown:", choices=choices).ask()
    if mid is None or mid == "back":
        return

    try:
        data    = load_result(mid, dataset_key)
        metrics = data["metrics"]
        pc      = metrics.get("per_class", {})
    except Exception as e:
        console.print(f"[red]Error loading results: {e}[/]")
        pause()
        return

    print_header(f"📊  Per-Class Detail — {MODEL_DISPLAY[mid]}")
    console.print(f"  Dataset: [bold]{DATASETS[dataset_key]}[/]  |  Overall accuracy: [bold green]{metrics['accuracy']*100:.2f}%[/]\n")

    table = Table(box=box.ROUNDED, header_style="bold cyan", padding=(0, 2))
    table.add_column("Class",         min_width=16)
    table.add_column("Precision",     min_width=12, justify="right")
    table.add_column("Recall",        min_width=10, justify="right")
    table.add_column("F1",            min_width=10, justify="right")
    table.add_column("Support (n)",   min_width=12, justify="right")

    colors = {"entailment": "green", "neutral": "yellow", "contradiction": "red"}
    for cls in ["entailment", "neutral", "contradiction"]:
        d = pc.get(cls, {})
        c = colors[cls]
        table.add_row(
            f"[{c}]{cls}[/]",
            f"{d.get('precision',0)*100:.2f}%",
            f"{d.get('recall',   0)*100:.2f}%",
            f"{d.get('f1',       0)*100:.2f}%",
            str(d.get("support", "—")),
        )

    console.print(table)

    # Confusion matrix
    cm = metrics.get("confusion_matrix")
    if cm:
        console.print("\n  [bold cyan]Confusion Matrix[/]  (rows = Actual, cols = Predicted)\n")
        cm_table = Table(box=box.SIMPLE, show_header=True, header_style="bold", padding=(0, 3))
        cm_table.add_column("Actual \\ Pred", style="bold")
        cm_table.add_column("Entailment",   justify="right")
        cm_table.add_column("Neutral",      justify="right")
        cm_table.add_column("Contradiction",justify="right")
        row_labels = ["Entailment", "Neutral", "Contradiction"]
        row_colors = ["green", "yellow", "red"]
        for i, (rl, rc) in enumerate(zip(row_labels, row_colors)):
            vals = [str(cm[i][j]) for j in range(3)]
            # Highlight diagonal (correct predictions)
            vals[i] = f"[bold {rc}]{vals[i]}[/]"
            cm_table.add_row(f"[{rc}]{rl}[/]", *vals)
        console.print(cm_table)

    pause()


def show():
    dataset_key = "snli_test"
    metric      = "accuracy"

    while True:
        _show_main_table(dataset_key, metric)

        action = questionary.select(
            "Options:",
            choices=[
                questionary.Choice("🗄  Change dataset",              value="dataset"),
                questionary.Choice("📐  Change metric (acc / F1)",    value="metric"),
                questionary.Choice("🔍  View per-class breakdown",    value="perclass"),
                questionary.Choice("← Back to Main Menu",            value="back"),
            ],
        ).ask()

        if action is None or action == "back":
            return

        elif action == "dataset":
            ds = questionary.select(
                "Select dataset:",
                choices=[questionary.Choice(v, value=k) for k, v in DATASETS.items()]
                        + [questionary.Choice("← Back", value="back")],
            ).ask()
            if ds and ds != "back":
                dataset_key = ds

        elif action == "metric":
            m = questionary.select(
                "Select metric:",
                choices=[
                    questionary.Choice("Accuracy",  value="accuracy"),
                    questionary.Choice("F1 Macro",  value="f1"),
                    questionary.Choice("← Back",    value="back"),
                ],
            ).ask()
            if m and m != "back":
                metric = m

        elif action == "perclass":
            _show_per_class_detail(dataset_key)
