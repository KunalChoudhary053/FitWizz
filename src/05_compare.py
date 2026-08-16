"""
05_compare.py -- the honest final verdict. Reads whatever every prior
script actually produced (reports/metrics.json) and states plainly
which one won. No cherry-picking, no re-running until something looks
better -- this reports what happened.
"""
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

with open("../reports/metrics.json") as f:
    metrics = json.load(f)

console.print(Panel.fit("[bold green]FitWizz -- Final Comparison[/bold green]", border_style="green"))

table = Table(title="Test-set MAE, all models, same 69 held-out days", border_style="grey50", header_style="bold green")
table.add_column("Model")
table.add_column("MAE (ms)", justify="right")
table.add_column("What it is")

labels = {
    "mean_baseline": ("Mean baseline", "Always guesses the training average. Ignores steps/sleep entirely."),
    "linear_baseline": ("Linear regression", "A straight-line fit of steps + sleep -> HRV."),
    "nn_from_scratch": ("Neural net (from scratch)", "2 -> 4 (ReLU) -> 1, hand-written forward pass + backprop."),
}
for key, (name, desc) in labels.items():
    if key in metrics:
        table.add_row(name, str(metrics[key]["mae"]), desc)

console.print(table)

winner = min(metrics, key=lambda k: metrics[k]["mae"])
winner_name = labels.get(winner, (winner,))[0]

console.print(f"\n[bold green]Winner: {winner_name} ({metrics[winner]['mae']} ms MAE)[/bold green]")

if winner == "mean_baseline":
    console.print(
        "\n[bold]Honest read:[/bold] neither the linear model nor the neural net beat simply\n"
        "guessing the average. That means steps and sleep alone don't contain enough signal\n"
        "to predict this person's HRV -- a real, legitimate finding\n"
        "The likely next step isn't a bigger model -- it's more/better inputs (RHR, respiratory\n"
        "rate, or the WHOOP journal's behavioral flags) or more data."
    )
else:
    console.print(f"\n[bold]Honest read:[/bold] {winner_name} found real signal beyond a simple average.")