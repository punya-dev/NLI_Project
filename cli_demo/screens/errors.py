"""Screen 3 — 🔍 Error Analysis: hard/easy, error browser, confusion matrix."""
import questionary
from rich.table import Table
from rich.panel import Panel
from rich import box

from ui.console import console
from ui.layout import print_header, label_color, pause
from data.loader import load_hard_easy, load_errors_sample, load_result

MODEL_DISPLAY = {
    "lr":                  "LR",
    "distilbert":          "DistilBERT baseline",
    "distilbert_kd":       "DB + KD (generic)",
    "distilbert_kd_extra": "DB + KD + Extra SNLI",
    "llama":               "Llama-1B LoRA ★",
}


def _hard_easy():
    print_header("🔍  Hard vs Easy Examples — SNLI Test")
    console.print(
        "  [dim]Hard = annotators disagreed on the label\n"
        "  Easy = all annotators agreed[/]\n"
    )

    try:
        df = load_hard_easy()
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        pause()
        return

    table = Table(box=box.ROUNDED, header_style="bold cyan", padding=(0, 2))
    table.add_column("Model",         min_width=36)
    table.add_column("Overall Acc",   min_width=12, justify="right")
    table.add_column("Hard Acc",      min_width=10, justify="right")
    table.add_column("Easy Acc",      min_width=10, justify="right")

    for _, row in df.iterrows():
        model_name = str(row["model"])
        is_best    = "Llama" in model_name

        def _fmt(val):
            s = str(val)
            # strip "± x" for display
            return s.split("±")[0].strip() + "%" if "%" not in s else s

        overall = str(row.get("overall", "—"))
        hard    = str(row.get("hard",    "—"))
        easy    = str(row.get("easy",    "—"))

        # Add % if missing
        def _pct(v):
            v = v.split("±")[0].strip()
            try:
                return f"{float(v):.1f}%"
            except Exception:
                return v + "%"

        m_str = f"[bold green]{model_name}[/]" if is_best else model_name
        o_str = f"[bold green]{_pct(overall)}[/]" if is_best else _pct(overall)

        table.add_row(
            m_str,
            o_str,
            _pct(hard),
            _pct(easy),
        )

    console.print(table)
    console.print(
        "\n  [dim]Key insight: Llama shows the biggest improvement on [bold]Hard[/] examples (+9.5pp vs DistilBERT).[/]\n"
    )
    pause()


def _error_browser():
    print_header("🔍  Error Sample Browser")
    console.print("  [dim]Real examples where DistilBERT failed but Llama succeeded (and vice versa)[/]\n")

    try:
        df = load_errors_sample()
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        pause()
        return

    rows      = df.to_dict("records")
    total     = len(rows)
    idx       = 0

    while True:
        print_header(f"🔍  Error Browser  [{idx+1} / {total}]")
        row = rows[idx]

        # Category banner
        cat   = str(row.get("category", ""))
        color = "green" if "Llama right" in cat else "yellow"
        console.print(Panel(f"[bold {color}]{cat}[/]", border_style=color, padding=(0, 2)))
        console.print()

        # Premise / hypothesis / gold
        console.print(f"  [dim]Premise   :[/]  {row.get('premise', '')}")
        console.print(f"  [dim]Hypothesis:[/]  {row.get('hypothesis', '')}")
        gold  = str(row.get("gold", ""))
        gc    = label_color(gold)
        console.print(f"  [dim]Gold Label:[/]  [bold {gc}]{gold.upper()}[/]\n")

        # Model predictions
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold", padding=(0, 3))
        table.add_column("Model",      min_width=28)
        table.add_column("Prediction", min_width=16)

        col_map = {
            "baseline": "DistilBERT baseline",
            "kd":       "DB + KD",
            "llama":    "Llama-1B LoRA ★",
        }
        for col_key, model_name in col_map.items():
            pred  = str(row.get(col_key, "—")).strip()
            pc    = label_color(pred)
            icon  = "✅" if pred == gold else "❌"
            pred_str = f"[{pc}]{icon} {pred.upper() if pred != '—' else '—'}[/]"
            table.add_row(model_name, pred_str)

        console.print(table)
        console.print()

        action = questionary.select(
            "Navigate:",
            choices=[
                questionary.Choice("▶  Next example",  value="next"),
                questionary.Choice("◀  Prev example",  value="prev"),
                questionary.Choice("🔢  Jump to #",     value="jump"),
                questionary.Choice("← Back",           value="back"),
            ],
        ).ask()

        if action is None or action == "back":
            return
        elif action == "next":
            idx = (idx + 1) % total
        elif action == "prev":
            idx = (idx - 1) % total
        elif action == "jump":
            num = questionary.text(
                f"Jump to example (1–{total}):",
                validate=lambda t: t.isdigit() and 1 <= int(t) <= total,
            ).ask()
            if num:
                idx = int(num) - 1


def _confusion_matrix():
    choices = [questionary.Choice(v, value=k) for k, v in MODEL_DISPLAY.items()]
    choices.append(questionary.Choice("← Back", value="back"))
    mid = questionary.select("Select model for confusion matrix:", choices=choices).ask()
    if mid is None or mid == "back":
        return

    ds_choice = questionary.select(
        "Select dataset:",
        choices=[
            questionary.Choice("SNLI Test",       value="snli_test"),
            questionary.Choice("MNLI Matched",    value="mnli_m"),
            questionary.Choice("MNLI Mismatched", value="mnli_mm"),
            questionary.Choice("← Back",          value="back"),
        ],
    ).ask()
    if ds_choice is None or ds_choice == "back":
        return

    try:
        data = load_result(mid, ds_choice)
        cm   = data["metrics"]["confusion_matrix"]
        acc  = data["metrics"]["accuracy"]
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        pause()
        return

    print_header(f"🔍  Confusion Matrix — {MODEL_DISPLAY[mid]}")
    console.print(f"  Dataset: [bold]{ds_choice}[/]  |  Overall accuracy: [bold green]{acc*100:.2f}%[/]\n")
    console.print("  [dim]Rows = Actual label, Columns = Predicted label[/]\n")

    cls_labels = ["Entailment", "Neutral", "Contradiction"]
    cls_colors = ["green",      "yellow",  "red"]

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold", padding=(0, 3))
    table.add_column("Actual \\ Predicted", style="bold", min_width=20)
    for lbl, col in zip(cls_labels, cls_colors):
        table.add_column(f"[{col}]{lbl}[/]", justify="right", min_width=14)

    for i, (rl, rc) in enumerate(zip(cls_labels, cls_colors)):
        total_row = sum(cm[i])
        vals = []
        for j in range(3):
            count = cm[i][j]
            pct   = count / total_row * 100 if total_row else 0
            cell  = f"{count}  ({pct:.1f}%)"
            if i == j:
                cell = f"[bold {rc}]{cell}[/]"
            else:
                cell = f"[dim]{cell}[/]"
            vals.append(cell)
        table.add_row(f"[{rc}]{rl}[/]", *vals)

    console.print(table)

    # Most confused classes
    off_diag = [(cm[i][j], cls_labels[i], cls_labels[j]) for i in range(3) for j in range(3) if i != j]
    off_diag.sort(reverse=True)
    top = off_diag[0]
    console.print(f"\n  [dim]Most confused: [bold]{top[1]}[/] predicted as [bold]{top[2]}[/] ({top[0]} cases)[/]\n")

    pause()


def show():
    while True:
        print_header("🔍  Error Analysis")

        action = questionary.select(
            "Choose analysis:",
            choices=[
                questionary.Choice("📊  Hard vs Easy examples (SNLI test)",   value="hardeasy"),
                questionary.Choice("📋  Browse error samples (paginated)",     value="browser"),
                questionary.Choice("🔢  Confusion matrix for a model",         value="confmat"),
                questionary.Choice("← Back to Main Menu",                     value="back"),
            ],
        ).ask()

        if action is None or action == "back":
            return
        elif action == "hardeasy":
            _hard_easy()
        elif action == "browser":
            _error_browser()
        elif action == "confmat":
            _confusion_matrix()
