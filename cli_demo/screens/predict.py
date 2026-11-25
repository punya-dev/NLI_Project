"""Screen 1 — 🔮 Live Predict: load model, get predictions, compare models."""
import time
import questionary
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from ui.console import console
from ui.layout import print_header, confidence_bar, colored_label, label_color, pause
from models.registry import MODEL_REGISTRY, get_model, is_loaded

# ── Curated example pairs ─────────────────────────────────────────────────────
EXAMPLES = [
    {
        "label": "1 — Simple entailment",
        "premise":    "A man is playing guitar on a stage.",
        "hypothesis": "A person is performing music.",
        "expected":   "entailment",
    },
    {
        "label": "2 — Clear contradiction",
        "premise":    "Two dogs run through the snow.",
        "hypothesis": "The animals are inside a house.",
        "expected":   "contradiction",
    },
    {
        "label": "3 — Implicit location inference",
        "premise":    "A woman is reading a book in the park.",
        "hypothesis": "A woman is outdoors.",
        "expected":   "entailment",
    },
    {
        "label": "4 — Tricky near-entailment (neutral)",
        "premise":    "A child is playing near a pool.",
        "hypothesis": "The child is swimming.",
        "expected":   "neutral",
    },
    {
        "label": "5 — Negation handling (LR struggles here)",
        "premise":    "No animals were harmed in the making of this film.",
        "hypothesis": "Animals were harmed.",
        "expected":   "contradiction",
    },
    {
        "label": "6 — World knowledge: telescope ≠ binoculars",
        "premise":    "Two children look through a telescope on a city street.",
        "hypothesis": "Two children are looking through binoculars.",
        "expected":   "contradiction",
    },
    {
        "label": "7 — Subtle medical contradiction",
        "premise":    "A nurse is listening to a woman's heartbeat with a stethoscope.",
        "hypothesis": "A nurse is taking a woman's temperature.",
        "expected":   "contradiction",
    },
]

MODEL_CHOICES = [
    questionary.Choice(
        f"{cfg['display_name']:46s}  ({cfg['snli_acc']}% SNLI)  [load: {cfg['load_time']}]",
        value=mid,
    )
    for mid, cfg in MODEL_REGISTRY.items()
] + [
    questionary.Choice("── Compare ALL models side-by-side ──", value="all"),
    questionary.Choice("← Back to Main Menu", value="back"),
]


def _load_with_spinner(model_id: str):
    """Load a model showing a spinner. Returns the model or None on failure."""
    if is_loaded(model_id):
        return get_model(model_id)

    cfg = MODEL_REGISTRY[model_id]
    console.print(f"\n  [cyan]Loading [bold]{cfg['display_name']}[/bold]…[/]  [dim](expected: {cfg['load_time']})[/]")

    try:
        from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
        with Progress(
            SpinnerColumn(),
            TextColumn("[cyan]{task.description}"),
            TimeElapsedColumn(),
            transient=True,
        ) as prog:
            task = prog.add_task(f"Loading {cfg['short_name']}…", total=None)
            model = get_model(model_id)
        console.print(f"  [green]✓ {cfg['short_name']} loaded![/]\n")
        return model
    except Exception as e:
        console.print(f"\n  [red]✗ Failed to load {cfg['display_name']}[/]")
        console.print(f"  [dim red]{e}[/]\n")
        return None


def _show_single_result(model_id: str, result: dict, premise: str, hypothesis: str):
    cfg = MODEL_REGISTRY[model_id]
    label = result["label"]
    probs = result["probs"]

    console.print(
        Panel(
            f"[bold]{cfg['display_name']}[/]",
            border_style="cyan",
            padding=(0, 2),
        )
    )
    console.print(f"  [dim]Premise  :[/]  {premise}")
    console.print(f"  [dim]Hypothesis:[/] {hypothesis}\n")

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan", padding=(0, 2))
    table.add_column("Label", min_width=18)
    table.add_column("Confidence", min_width=32)

    ordered = sorted(probs.items(), key=lambda x: -x[1])
    for lbl, p in ordered:
        bar = confidence_bar(p)
        is_best = lbl == label
        icon = "✅" if lbl == "entailment" and is_best else ("❌" if lbl == "contradiction" and is_best else ("➡ " if lbl == "neutral" and is_best else "   "))
        color = label_color(lbl)
        lbl_str = f"[bold {color}]{icon} {lbl.upper()}[/]" if is_best else f"[dim]   {lbl}[/]"
        bar_str = f"[bold {color}]{bar}[/]" if is_best else f"[dim]{bar}[/]"
        table.add_row(lbl_str, bar_str)

    console.print(table)

    if "error" in result:
        console.print(f"  [dim red]Note: {result['error']}[/]\n")


def _show_compare_table(results: dict, premise: str, hypothesis: str):
    """Show all-models comparison table."""
    console.print(f"\n  [dim]Premise  :[/]  {premise}")
    console.print(f"  [dim]Hypothesis:[/] {hypothesis}\n")

    table = Table(
        title="All Models — Side-by-Side",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        padding=(0, 2),
    )
    table.add_column("Model", min_width=36)
    table.add_column("Prediction", min_width=16)
    table.add_column("Confidence", min_width=10)
    table.add_column("Time", min_width=7)

    for mid, (result, elapsed) in results.items():
        cfg = MODEL_REGISTRY[mid]
        label = result.get("label", "error")
        conf  = result.get("confidence", 0.0)
        color = label_color(label)
        icon  = "✅" if label == "entailment" else ("❌" if label == "contradiction" else ("➡ " if label == "neutral" else "⚠ "))
        table.add_row(
            cfg["display_name"],
            f"[{color}]{icon} {label}[/]",
            f"[bold]{conf*100:.1f}%[/]",
            f"{elapsed:.2f}s",
        )

    console.print(table)


def show():
    while True:
        print_header("🔮  Live Predict")

        model_id = questionary.select(
            "Select a model (or compare all):",
            choices=MODEL_CHOICES,
            style=questionary.Style([
                ("selected", "fg:cyan bold"),
                ("pointer",  "fg:cyan bold"),
            ]),
        ).ask()

        if model_id is None or model_id == "back":
            return

        # ── Input ─────────────────────────────────────────────────────────────
        input_choice = questionary.select(
            "Input method:",
            choices=[
                questionary.Choice("Type my own premise & hypothesis", value="type"),
                questionary.Choice("Pick from example pairs",          value="examples"),
                questionary.Choice("← Back",                          value="back"),
            ],
        ).ask()

        if input_choice is None or input_choice == "back":
            continue

        if input_choice == "examples":
            ex_choices = [questionary.Choice(e["label"], value=i) for i, e in enumerate(EXAMPLES)]
            ex_choices.append(questionary.Choice("← Back", value="back"))
            ex_idx = questionary.select("Choose an example:", choices=ex_choices).ask()
            if ex_idx is None or ex_idx == "back":
                continue
            ex = EXAMPLES[ex_idx]
            premise    = ex["premise"]
            hypothesis = ex["hypothesis"]
            console.print(f"\n  [dim]Premise  :[/]  {premise}")
            console.print(f"  [dim]Hypothesis:[/] {hypothesis}")
            console.print(f"  [dim]Expected  :[/]  [bold]{ex['expected']}[/]\n")
        else:
            premise = questionary.text(
                "Premise:",
                multiline=False,
                validate=lambda t: True if t.strip() else "Please enter a premise",
            ).ask()
            if premise is None:
                continue
            hypothesis = questionary.text(
                "Hypothesis:",
                multiline=False,
                validate=lambda t: True if t.strip() else "Please enter a hypothesis",
            ).ask()
            if hypothesis is None:
                continue

        # ── Inference ─────────────────────────────────────────────────────────
        print_header("🔮  Live Predict — Results")

        if model_id == "all":
            results = {}
            for mid in MODEL_REGISTRY:
                model = _load_with_spinner(mid)
                if model is None:
                    results[mid] = ({"label": "error", "confidence": 0.0, "probs": {}}, 0.0)
                    continue
                t0 = time.time()
                result = model.predict(premise, hypothesis)
                results[mid] = (result, time.time() - t0)
            _show_compare_table(results, premise, hypothesis)
        else:
            model = _load_with_spinner(model_id)
            if model is not None:
                t0 = time.time()
                result = model.predict(premise, hypothesis)
                elapsed = time.time() - t0
                _show_single_result(model_id, result, premise, hypothesis)
                console.print(f"  [dim]Inference time: {elapsed:.3f}s[/]\n")

        # ── Post-prediction menu ──────────────────────────────────────────────
        action = questionary.select(
            "What next?",
            choices=[
                questionary.Choice("🔄  Predict again (same model)",          value="again"),
                questionary.Choice("🔀  Compare with ALL models",             value="compare"),
                questionary.Choice("🎯  Try another example / input",         value="new"),
                questionary.Choice("← Back to Main Menu",                    value="back"),
            ],
        ).ask()

        if action is None or action == "back":
            return
        elif action == "again":
            # re-run with same model and same input
            print_header("🔮  Live Predict — Results")
            if model_id != "all":
                model = _load_with_spinner(model_id)
                if model is not None:
                    t0 = time.time()
                    result = model.predict(premise, hypothesis)
                    elapsed = time.time() - t0
                    _show_single_result(model_id, result, premise, hypothesis)
                    console.print(f"  [dim]Inference time: {elapsed:.3f}s[/]\n")
            pause()
        elif action == "compare":
            print_header("🔮  Compare All Models")
            results = {}
            for mid in MODEL_REGISTRY:
                model = _load_with_spinner(mid)
                if model is None:
                    results[mid] = ({"label": "error", "confidence": 0.0, "probs": {}}, 0.0)
                    continue
                t0 = time.time()
                res = model.predict(premise, hypothesis)
                results[mid] = (res, time.time() - t0)
            _show_compare_table(results, premise, hypothesis)
            pause()
        elif action == "new":
            continue   # loops back to model select
