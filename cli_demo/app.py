"""
App controller — the main event loop.
Runs screens in a while-loop, routing from the main menu.
"""
from ui.console import console
from ui.layout import clear


def run():
    # Lazy imports so startup is instant
    from screens import main_menu, predict, results, errors, analysis, compute

    # Greeting
    clear()
    console.print(
        "\n  [bold cyan]Welcome to the NLI Demo![/]  "
        "[dim]Loading menus...[/]\n"
    )

    while True:
        choice = main_menu.show()

        if choice == "predict":
            predict.show()
        elif choice == "results":
            results.show()
        elif choice == "errors":
            errors.show()
        elif choice == "analysis":
            analysis.show()
        elif choice == "compute":
            compute.show()
        elif choice == "exit" or choice is None:
            clear()
            console.print("\n  [bold cyan]Thanks for using the NLI Demo! 👋[/]\n")
            break
