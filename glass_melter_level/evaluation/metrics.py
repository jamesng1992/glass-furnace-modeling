"""
Model evaluation metrics: RMSE, MAE, R².

Provides consistent metric computation and formatted printing
for comparing Neural ODE model variants.
"""

import numpy as np
import torch
from typing import Dict, Optional
from sklearn.metrics import mean_absolute_error, r2_score


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Compute RMSE, MAE, and R² metrics.

    Args:
        y_true: Ground truth values (numpy array or tensor)
        y_pred: Predicted values (numpy array or tensor)

    Returns:
        Dictionary with 'rmse', 'mae', 'r2' keys
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    return {"rmse": rmse, "mae": mae, "r2": r2}


def print_metrics_row(name: str, metrics: Dict[str, float], unit: str = "") -> None:
    """
    Print metrics in a formatted row.

    Args:
        name: Model or state name
        metrics: Dictionary with 'rmse', 'mae', 'r2'
        unit: Physical unit string (e.g., ' m', ' m/h')
    """
    print(
        f"   {name:>8s}: "
        f"RMSE={metrics['rmse']:.6f}{unit}, "
        f"MAE={metrics['mae']:.6f}{unit}, "
        f"R²={metrics['r2']:.4f}"
    )


def print_model_comparison(
    results: Dict[str, Dict[str, Dict[str, float]]],
    title: str = "MODEL COMPARISON",
) -> None:
    """
    Print a formatted comparison table for multiple model variants.

    Args:
        results: Nested dict: {model_name: {'open_loop': metrics, 'corrected': metrics}}
        title: Section title
    """
    print("=" * 110)
    print(f"  {title}")
    print("=" * 110)

    # Open-loop table
    print(f"\n{'Model':<20} {'Training Data':<25} "
          f"{'RMSE (m)':>12} {'MAE (m)':>12} {'R²':>10}")
    print("-" * 100)

    for name, data in results.items():
        ol = data.get("open_loop", {})
        desc = data.get("description", "")
        print(
            f"{name:<20} {desc:<25} "
            f"{ol.get('rmse', 0):>12.6f} "
            f"{ol.get('mae', 0):>12.6f} "
            f"{ol.get('r2', 0):>10.4f}"
        )

    # Corrected table
    print(f"\n{'Model':<20} {'Training Data':<25} "
          f"{'RMSE (m)':>12} {'MAE (m)':>12} {'R²':>10} {'RMSE Improv.':>14}")
    print("-" * 110)

    for name, data in results.items():
        ol = data.get("open_loop", {})
        corr = data.get("corrected", {})
        desc = data.get("description", "")
        improv = (1 - corr.get("rmse", 0) / ol.get("rmse", 1)) * 100 if ol.get("rmse", 0) > 0 else 0
        print(
            f"{name:<20} {desc:<25} "
            f"{corr.get('rmse', 0):>12.6f} "
            f"{corr.get('mae', 0):>12.6f} "
            f"{corr.get('r2', 0):>10.4f} "
            f"{improv:>13.1f}%"
        )

    # Find best
    best_name = min(results, key=lambda n: results[n].get("corrected", {}).get("rmse", float("inf")))
    best_rmse = results[best_name]["corrected"]["rmse"]
    print(f"\n  🏆 Best Model (corrected): {best_name} with RMSE = {best_rmse:.6f} m")
