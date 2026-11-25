"""Header, banner, and layout helpers."""
import os
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from ui.console import console


APP_TITLE = "NLI DEMO  ·  NLP Phase 3"
APP_SUBTITLE = "Natural Language Inference — 5 Models Compared"


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def print_header(subtitle: str = ""):
    clear()
    title_text = Text()
    title_text.append(APP_TITLE, style="bold cyan")
    if subtitle:
        title_text.append(f"\n{subtitle}", style="dim white")
    else:
        title_text.append(f"\n{APP_SUBTITLE}", style="dim white")
    console.print(
        Panel(
            Align.center(title_text),
            border_style="cyan",
            padding=(0, 2),
        )
    )
    console.print()


def print_section(title: str):
    console.rule(f"[bold cyan]{title}[/]")
    console.print()


def confidence_bar(prob: float, width: int = 22) -> str:
    filled = round(prob * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar}  {prob * 100:.1f}%"


def label_color(label: str) -> str:
    return {
        "entailment":    "green",
        "neutral":       "yellow",
        "contradiction": "red",
    }.get(label.lower(), "white")


def colored_label(label: str) -> str:
    color = label_color(label)
    icon = {"entailment": "✅", "neutral": "➡ ", "contradiction": "❌"}.get(label.lower(), "  ")
    return f"[{color}]{icon} {label.upper()}[/]"


def pause(msg: str = "\n  [dim]Press Enter to continue...[/]"):
    console.print(msg)
    input()
