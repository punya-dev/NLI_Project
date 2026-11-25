"""Main menu screen."""
import questionary
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich import box

from ui.console import console
from ui.layout import clear
from models.registry import MODEL_REGISTRY, is_loaded

MENU_CHOICES = [
    questionary.Choice("🔮   Live Predict          run any model on custom input",   value="predict"),
    questionary.Choice("📊   Benchmark Results     pre-computed accuracy/F1 tables", value="results"),
    questionary.Choice("🔍   Error Analysis        hard/easy, error browser, confusion matrix", value="errors"),
    questionary.Choice("📈   Analysis Deep-Dive    property breakdown, genre, significance",   value="analysis"),
    questionary.Choice("⚡   Compute & Efficiency  speed, params, training cost",    value="compute"),
    questionary.Choice("✖    Exit",                                                   value="exit"),
]

BANNER = r"""
  ███╗   ██╗██╗     ██╗
  ████╗  ██║██║     ██║
  ██╔██╗ ██║██║     ██║
  ██║╚██╗██║██║     ██║
  ██║ ╚████║███████╗██║
  ╚═╝  ╚═══╝╚══════╝╚═╝  Demo
"""


def _status_line() -> str:
    loaded = [mid for mid in MODEL_REGISTRY if is_loaded(mid)]
    if not loaded:
        return "  [dim]No models loaded yet — will load on first use[/]"
    names = ", ".join(MODEL_REGISTRY[m]["short_name"] for m in loaded)
    return f"  [dim]Loaded in session: [cyan]{names}[/][/]"


def show() -> str:
    clear()
    console.print(
        Panel(
            Align.center(
                Text.from_markup(
                    "[bold cyan]NLI DEMO[/]  ·  NLP Phase 3\n"
                    "[dim]Natural Language Inference — 5 Models Compared[/]"
                )
            ),
            border_style="cyan",
            padding=(1, 4),
        )
    )
    console.print(_status_line())
    console.print()

    choice = questionary.select(
        "Main Menu — use ↑↓ to navigate, Enter to select:",
        choices=MENU_CHOICES,
        style=questionary.Style([
            ("selected",        "fg:cyan bold"),
            ("pointer",         "fg:cyan bold"),
            ("highlighted",     "fg:cyan"),
            ("answer",          "fg:cyan bold"),
            ("question",        "fg:white bold"),
        ]),
    ).ask()

    return choice if choice is not None else "exit"
