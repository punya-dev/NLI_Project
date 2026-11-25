"""Screen 5 — ⚡ Compute & Efficiency."""
import questionary
from rich.table import Table
from rich import box

from ui.console import console
from ui.layout import print_header, pause
from data.loader import load_compute


def show():
    print_header("⚡  Compute & Efficiency")
    console.print(
        "  [dim]Trade-off between model accuracy, parameter count, training cost, and inference speed.[/]\n"
    )

    try:
        df = load_compute()
    except Exception as e:
        console.print(f"[red]Error loading compute data: {e}[/]")
        pause()
        return

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        padding=(0, 2),
        min_width=80,
    )
    table.add_column("Model",             min_width=36)
    table.add_column("Params",            min_width=10, justify="right")
    table.add_column("Train Time",        min_width=12, justify="right")
    table.add_column("Peak GPU",          min_width=10, justify="right")
    table.add_column("Inference (ex/s)",  min_width=16, justify="right")

    # Build a simple in-memory bar for inference speed
    max_speed = 15000.0

    for _, row in df.iterrows():
        model_name = str(row["model"])
        params     = row.get("params_M",          "—")
        train_min  = row.get("train_min",         "—")
        gpu        = row.get("peak_gpu_GB",       "—")
        speed      = row.get("infer_ex_per_s",    "—")

        is_llama   = "Llama" in model_name
        is_lr      = model_name == "LR"
        style      = "bold green" if is_llama else ("bold yellow" if is_lr else "")

        try:
            params_str = f"{float(params):,.1f} M"
        except Exception:
            params_str = str(params)

        try:
            train_str = f"{float(train_min):.1f} min"
        except Exception:
            train_str = str(train_min)

        try:
            gpu_str = f"{float(gpu):.1f} GB"
        except Exception:
            gpu_str = "CPU only" if str(gpu) == "nan" else str(gpu)

        try:
            sp    = float(speed)
            bar_w = 10
            filled = max(1, round(sp / max_speed * bar_w))
            bar   = "█" * filled + "░" * (bar_w - filled)
            speed_str = f"{sp:>8,.0f}  {bar}"
        except Exception:
            speed_str = str(speed)

        m_str = f"[{style}]{model_name}[/]" if style else model_name
        p_str = f"[{style}]{params_str}[/]" if style else params_str
        t_str = f"[{style}]{train_str}[/]" if style else train_str

        table.add_row(
            m_str,
            p_str,
            t_str,
            gpu_str,
            speed_str,
        )

    console.print(table)

    console.print(
        "\n  [bold cyan]Key Takeaways:[/]\n"
        "  • [yellow]LR[/]        : Fastest inference by far (14K ex/s), but lowest accuracy (74.6%).\n"
        "  • [cyan]DistilBERT[/]: Great accuracy/speed trade-off (~3K ex/s, 85–88% accuracy).\n"
        "  • [bold cyan]DB + KD + generic[/]: Best distilbert-class model — +1.5pp MNLI with same speed.\n"
        "  • [green]Llama[/]     : Best accuracy (91.2%) but 40× slower and needs 8GB GPU.\n"
    )

    pause()
