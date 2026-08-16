"""
01_load_explore.py -- first honest look at the real data before any model
touches it. No feature engineering, no modeling -- just: what do we
actually have, and how much of it is usable.
"""
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# --- WHOOP physiological cycles: our source for HRV and WHOOP's own sleep duration ---
whoop = pd.read_csv("../data/physiological_cycles.csv")
whoop["date"] = pd.to_datetime(whoop["Cycle start time"]).dt.date
whoop["hrv_ms"] = whoop["Heart rate variability (ms)"]
whoop["sleep_hours_whoop"] = whoop["Asleep duration (min)"] / 60

# --- Apple Health daily rollup: our source for steps ---
apple = pd.read_csv("../data/daily_health.csv", parse_dates=["date"])
apple["date"] = apple["date"].dt.date

# --- merge on date: this tells us the REAL usable sample size for the model ---
merged = whoop[["date", "hrv_ms", "sleep_hours_whoop"]].merge(
    apple[["date", "steps"]], on="date", how="inner"
)
complete = merged.dropna(subset=["hrv_ms", "sleep_hours_whoop", "steps"])
complete.to_csv("../data/model_ready_hrv.csv", index=False)

console.print(Panel.fit("[bold green]FitWizz -- Data Coverage Check[/bold green]", border_style="green"))

coverage = Table(title="Source Coverage", border_style="grey50", header_style="bold green")
coverage.add_column("Source")
coverage.add_column("Field")
coverage.add_column("Days present", justify="right")
coverage.add_column("Of total", justify="right")
coverage.add_row("WHOOP", "HRV", str(whoop["hrv_ms"].notna().sum()), str(len(whoop)))
coverage.add_row("WHOOP", "Sleep duration", str(whoop["sleep_hours_whoop"].notna().sum()), str(len(whoop)))
coverage.add_row("Apple Health", "Steps", str(apple["steps"].notna().sum()), str(len(apple)))
console.print(coverage)

console.print(f"\n[bold]Date range:[/bold] {whoop['date'].min()} to {whoop['date'].max()}")
console.print(f"[bold green]Days with ALL THREE (steps, sleep, HRV) present: {len(complete)}[/bold green]  <- real usable sample size\n")

stats = Table(title="Model-Ready Stats", border_style="grey50", header_style="bold green")
stats.add_column("Metric")
stats.add_column("HRV (ms)", justify="right")
stats.add_column("Sleep (hrs)", justify="right")
stats.add_column("Steps", justify="right")
desc = complete[["hrv_ms", "sleep_hours_whoop", "steps"]].describe().round(1)
for row_name in ["mean", "std", "min", "50%", "max"]:
    stats.add_row(row_name, str(desc.loc[row_name, "hrv_ms"]), str(desc.loc[row_name, "sleep_hours_whoop"]), str(desc.loc[row_name, "steps"]))
console.print(stats)

console.print("\n[dim]Saved -> data/model_ready_hrv.csv[/dim]")