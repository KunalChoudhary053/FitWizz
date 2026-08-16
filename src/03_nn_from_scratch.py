"""
03_nn_from_scratch.py -- forward pass, loss, and backprop, all written
by hand. No PyTorch. This is the part almost nobody who *uses* ML has
actually built themselves -- worth being able to explain line by line.

Architecture: 2 inputs (steps, sleep) -> 4 hidden neurons (ReLU) -> 1 output (HRV).
Kept small on purpose -- 273 training rows can't support much more than
this without immediately overfitting.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

df = pd.read_csv("../data/model_ready_hrv.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
split_idx = int(len(df) * 0.8)
train, test = df.iloc[:split_idx], df.iloc[split_idx:]

# scale inputs -- steps (~10,000s) and sleep (~5-8) are wildly different
# magnitudes. Without this, steps would dominate the gradients just
# because its numbers are bigger, not because it matters more.
# Mean/std from TRAIN ONLY, applied to test -- same no-leakage rule as always.
X_train_raw = train[["steps", "sleep_hours_whoop"]].values
X_mean, X_std = X_train_raw.mean(axis=0), X_train_raw.std(axis=0)

def scale(X):
    return (X - X_mean) / X_std

X_train = scale(X_train_raw)
X_test = scale(test[["steps", "sleep_hours_whoop"]].values)
y_train = train["hrv_ms"].values.reshape(-1, 1)
y_test = test["hrv_ms"].values.reshape(-1, 1)

W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.5
b2 = np.zeros((1, 1))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    # ReLU's slope is 1 where it was "on" (z>0), 0 where it was "off".
    # This is the local derivative the chain rule needs at that point.
    return (z > 0).astype(float)

def forward(X):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2   # linear output -- no activation here, this IS the prediction
    return z1, a1, z2

lr = 0.01
epochs = 5000
n = X_train.shape[0]

for epoch in range(epochs):
    # --- forward pass ---
    z1, a1, z2 = forward(X_train)
    y_pred = z2
    loss = np.mean((y_pred - y_train) ** 2)   # MSE

    # --- backward pass: chain rule, working from the loss back to W1 ---
    # d(loss)/d(z2): derivative of MSE w.r.t. the prediction itself
    dL_dz2 = 2 * (y_pred - y_train) / n

    # how much each W2 weight contributed: chain rule through the second layer
    dL_dW2 = a1.T @ dL_dz2
    dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

    # push the error back through W2 to see how much each hidden neuron mattered
    dL_da1 = dL_dz2 @ W2.T
    # then through the ReLU: kill the gradient anywhere ReLU was "off"
    dL_dz1 = dL_da1 * relu_derivative(z1)

    # how much each W1 weight contributed: chain rule through the first layer
    dL_dW1 = X_train.T @ dL_dz1
    dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

    # --- gradient descent: nudge every weight downhill ---
    W2 -= lr * dL_dW2
    b2 -= lr * dL_db2
    W1 -= lr * dL_dW1
    b1 -= lr * dL_db1

    if epoch % 500 == 0 or epoch == epochs - 1:
        print(f"epoch {epoch:5d}  train MSE loss = {loss:.3f}")

# --- evaluate on held-out test days, same metric as the baseline (MAE) ---
_, _, test_pred = forward(X_test)
test_mae = np.mean(np.abs(test_pred - y_test))
print(f"\nFinal test MAE: {test_mae:.2f} ms")
print(f"Baseline (mean) MAE was: 6.77 ms")
print(f"Baseline (linear) MAE was: 6.94 ms")

import json
with open("../reports/metrics.json") as f:
    metrics = json.load(f)
metrics["nn_from_scratch"] = {"mae": round(float(test_mae), 2)}
with open("../reports/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved -> reports/metrics.json")