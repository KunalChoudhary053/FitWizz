"""
02_baseline.py -- the "dumb" model. This exists purely to give every
later, fancier model an honest number it has to beat -- see the
resting-HR project earlier today, where the "smart" Random Forest
actually LOST to a simple rolling average. That's the entire reason
this file runs before any neural net does.

Two baselines, dumbest first:
1. Mean baseline -- always guess the average HRV from training data.
   Ignores steps and sleep completely. This is the true floor.
2. Linear regression -- a straight-line fit of HRV from steps + sleep.
   This is the actual "dumb linear model" the neural net has to beat.

TIME-ORDERED SPLIT, not random shuffle: same discipline as the
resting-HR project -- we hold out the most recent ~20% of days as a
true "future" test set, since random shuffling on daily physiological
data leaks information and produces a falsely optimistic score.
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

df = pd.read_csv("../data/model_ready_hrv.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)

split_idx = int(len(df) * 0.8)
train, test = df.iloc[:split_idx], df.iloc[split_idx:]

console.print(Panel.fit("[bold green]FitWizz -- 02: Baseline Models[/bold green]", border_style="green"))
console.print(f"Train: {len(train)} days ({train.date.min().date()} to {train.date.max().date()})")
console.print(f"Test:  {len(test)} days ({test.date.min().date()} to {test.date.max().date()})\n")

X_train, y_train = train[["steps", "sleep_hours_whoop"]], train["hrv_ms"]
X_test, y_test = test[["steps", "sleep_hours_whoop"]], test["hrv_ms"]

results = {}

def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    results[name] = {"mae": round(float(mae), 2), "rmse": round(float(rmse), 2)}

# --- baseline 1: mean guess, ignores steps/sleep entirely ---
mean_pred = np.full(len(y_test), y_train.mean())
evaluate(y_test, mean_pred, "mean_baseline")

# --- baseline 2: linear regression on steps + sleep ---
lin = LinearRegression()
lin.fit(X_train, y_train)
lin_pred = lin.predict(X_test)
evaluate(y_test, lin_pred, "linear_baseline")

table = Table(title="Baseline Results", border_style="grey50", header_style="bold green")
table.add_column("Model")
table.add_column("MAE (ms)", justify="right")
table.add_column("RMSE (ms)", justify="right")
for name, r in results.items():
    table.add_row(name, str(r["mae"]), str(r["rmse"]))
console.print(table)

console.print(f"\n[bold]Linear model coefficients:[/bold] steps={lin.coef_[0]:.5f}, sleep_hours={lin.coef_[1]:.3f}")
console.print("[dim]This is the number 03_nn_from_scratch.py and 04_nn_pytorch.py have to beat.[/dim]\n")

with open("../reports/metrics.json", "w") as f:
    json.dump(results, f, indent=2)
console.print("[dim]Saved -> reports/metrics.json[/dim]")