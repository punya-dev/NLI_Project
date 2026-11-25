"""Screen 4 — 📈 Analysis Deep-Dive: property breakdown, genre, significance."""
import questionary
from rich.table import Table
from rich import box

from ui.console import console
from ui.layout import print_header, pause
from data.loader import (
    load_accuracy_by_property,
    load_genre,
    load_significance,
)

DATASETS = {
    "snli_test": "SNLI Test",
    "mnli_m":    "MNLI Matched",
    "mnli_mm":   "MNLI Mismatched",
}

PROPERTIES = {
    "hyp_negation":    "Hypothesis Negation (does hypothesis contain negation?)",
    "overlap_bucket":  "Word Overlap Bucket (premise ↔ hypothesis overlap)",
    "hyp_len_bucket":  "Hypothesis Length Bucket",
    "prem_len_bucket": "Premise Length Bucket",
}

# Columns in accuracy_by_property.csv that map to model names
PROP_MODEL_COLS = [
    "LR",
    "DistilBERT baseline",
    "DB + distillation (kd)",
    "DB + distillation + generic",
    "DB + distillation + extra SNLI (control)",
    "Llama-1B LoRA",
]


def _accuracy_by_property():
    # Choose property
    prop_choices = [questionary.Choice(v, value=k) for k, v in PROPERTIES.items()]
    prop_choices.append(questionary.Choice("← Back", value="back"))
    prop = questionary.select("Select linguistic property:", choices=prop_choices).ask()
    if prop is None or prop == "back":
        return

    # Choose dataset
    ds_choices = [questionary.Choice(v, value=k) for k, v in DATASETS.items()]
    ds_choices.append(questionary.Choice("← Back", value="back"))
    dataset = questionary.select("Select dataset:", choices=ds_choices).ask()
    if dataset is None or dataset == "back":
        return

    try:
        df = load_accuracy_by_property()
        df = df[(df["eval_set"] == dataset) & (df["property"] == prop)]
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        pause()
        return

    if df.empty:
        console.print(f"[yellow]No data found for {prop} / {dataset}[/]")
        pause()
        return

    print_header(f"📈  Accuracy by Property: {PROPERTIES[prop]}")
    console.print(f"  Dataset: [bold]{DATASETS[dataset]}[/]\n")

    # Determine which model columns are available
    available_cols = [c for c in PROP_MODEL_COLS if c in df.columns]

    table = Table(box=box.ROUNDED, header_style="bold cyan", padding=(0, 1))
    table.add_column("Value / Bucket",    min_width=24)
    table.add_column("N",                 min_width=7, justify="right")
    for col in available_cols:
        short = col.replace("DB + distillation", "DB+KD").replace(" (control)", "").replace(" + generic", "+gen")
        table.add_column(short, min_width=9, justify="right")

    for _, row in df.iterrows():
        vals = [str(row.get(c, "—")) for c in available_cols]
        # Highlight the best (Llama) column
        try:
            llama_idx = available_cols.index("Llama-1B LoRA")
            vals[llama_idx] = f"[bold green]{vals[llama_idx]}%[/]"
            # others
            vals = [f"{v}%" if i != llama_idx and v != "—" else v for i, v in enumerate(vals)]
        except ValueError:
            vals = [f"{v}%" if v != "—" else v for v in vals]

        table.add_row(str(row["value"]), str(int(row["n"])), *vals)

    console.print(table)
    console.print()
    pause()


def _genre_breakdown():
    dataset = questionary.select(
        "Select MNLI dataset:",
        choices=[
            questionary.Choice("MNLI Matched",    value="mnli_m"),
            questionary.Choice("MNLI Mismatched", value="mnli_mm"),
            questionary.Choice("← Back",          value="back"),
        ],
    ).ask()
    if dataset is None or dataset == "back":
        return

    try:
        df = load_genre(dataset)
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        pause()
        return

    print_header(f"📈  Accuracy by Genre — {DATASETS[dataset]}")
    console.print("  [dim]MNLI has 5 genres; models trained on SNLI generalize differently to each[/]\n")

    # Determine available model columns
    model_cols = [c for c in df.columns if c != "genre"]
    if not model_cols:
        model_cols = list(df.columns)

    table = Table(box=box.ROUNDED, header_style="bold cyan", padding=(0, 2))
    table.add_column("Genre", min_width=16)
    for col in model_cols:
        short = str(col).replace("DB + distillation", "DB+KD").replace(" (kd)", "").replace(" + generic", "+gen").replace(" + extra SNLI (control)", "+extra")
        is_llama = "Llama" in str(col)
        table.add_column(f"[bold green]{short}[/]" if is_llama else short, min_width=10, justify="right")

    for genre, row in df.iterrows():
        vals = []
        for col in model_cols:
            v = row.get(col, "—")
            is_llama = "Llama" in str(col)
            try:
                cell = f"{float(v):.1f}%"
            except Exception:
                cell = str(v)
            vals.append(f"[bold green]{cell}[/]" if is_llama else cell)
        table.add_row(str(genre), *vals)

    console.print(table)
    console.print(
        "\n  [dim]Insight: Llama leads on every genre. DB+KD+generic shows biggest relative\n"
        "  improvement over baseline on out-of-domain MNLI genres.[/]\n"
    )
    pause()


def _significance_table():
    print_header("📈  Statistical Significance vs. DistilBERT Baseline")
    console.print(
        "  [dim]McNemar test, 3 seeds. All DB+KD variants show significant improvement on MNLI.[/]\n"
    )

    try:
        df = load_significance()
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        pause()
        return

    table = Table(box=box.ROUNDED, header_style="bold cyan", padding=(0, 2))
    table.add_column("Model",         min_width=36)
    table.add_column("Dataset",       min_width=12)
    table.add_column("Δ Acc (pts)",   min_width=12, justify="right")
    table.add_column("95% CI",        min_width=18)
    table.add_column("Sig. seeds",    min_width=12, justify="right")

    for _, row in df.iterrows():
        diff    = float(row["diff_pts"])
        color   = "green" if diff > 0 else "red"
        sig_str = str(row.get("seeds_sig(p<.05)", "—"))
        all_sig = sig_str.startswith("3/3")
        table.add_row(
            str(row["model"]),
            str(row["eval_set"]),
            f"[{color}]{diff:+.2f}[/]",
            str(row.get("ci95", "—")),
            f"[bold green]{sig_str}[/]" if all_sig else f"[yellow]{sig_str}[/]",
        )

    console.print(table)
    console.print()
    pause()


def show():
    while True:
        print_header("📈  Analysis Deep-Dive")

        action = questionary.select(
            "Choose analysis:",
            choices=[
                questionary.Choice("🔤  Accuracy by linguistic property",   value="prop"),
                questionary.Choice("🗂   Genre breakdown (MNLI)",             value="genre"),
                questionary.Choice("📐  Statistical significance table",     value="sig"),
                questionary.Choice("← Back to Main Menu",                   value="back"),
            ],
        ).ask()

        if action is None or action == "back":
            return
        elif action == "prop":
            _accuracy_by_property()
        elif action == "genre":
            _genre_breakdown()
        elif action == "sig":
            _significance_table()
