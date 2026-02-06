"""
Evaluate and compare Neural ODE model variants on LQI data.

Usage:
    python scripts/evaluate.py --data-path data/LQI_data.xlsx \
        --v1 trained_models/neural_ode_v1.pt \
        --v2 trained_models/neural_ode_v2.pt \
        --v3 trained_models/neural_ode_v3.pt
"""

import argparse
import torch
import numpy as np

from glass_melter_level.config import (
    GlassMelterParams,
    DEFAULT_CORRECTION_GAIN_H,
    DEFAULT_CORRECTION_GAIN_V,
)
from glass_melter_level.data.loader import load_lqi_data
from glass_melter_level.training.trajectory import prepare_lqi_trajectory
from glass_melter_level.models.neural_ode import (
    NeuralODE_GlassMelter,
    load_model,
    integrate_neural_ode,
    integrate_neural_ode_with_correction,
)
from glass_melter_level.evaluation.metrics import compute_metrics, print_model_comparison
from glass_melter_level.visualization.style import setup_publication_style
from glass_melter_level.visualization.plots import (
    plot_state_comparison,
    plot_model_comparison_bars,
)


def evaluate_model(model, traj, correction_gain_h, correction_gain_v):
    """Evaluate a single model with open-loop and corrected predictions."""
    model.eval()
    with torch.no_grad():
        # Open-loop
        x_open = integrate_neural_ode(model, traj["x0"], traj["u"], traj["t"])

        # Corrected
        h_meas = traj["x"][:, 0].numpy()
        x_corr = integrate_neural_ode_with_correction(
            model, traj["x0"], traj["u"], traj["t"],
            h_measured=h_meas,
            correction_gain=correction_gain_h,
        )

    h_true = traj["x"][:, 0].numpy()
    v_true = traj["x"][:, 1].numpy()
    qm_true = traj["x"][:, 2].numpy()

    m_open = compute_metrics(h_true, x_open[:, 0].numpy())
    m_corr = compute_metrics(h_true, x_corr[:, 0].numpy())

    return {
        "t": traj["t"].numpy(),
        "h_true": h_true, "v_true": v_true, "qm_true": qm_true,
        "h_pred_orig": x_open[:, 0].numpy(),
        "v_pred_orig": x_open[:, 1].numpy(),
        "qm_pred_orig": x_open[:, 2].numpy(),
        "h_pred": x_corr[:, 0].numpy(),
        "v_pred": x_corr[:, 1].numpy(),
        "qm_pred": x_corr[:, 2].numpy(),
        "rmse_h_orig": m_open["rmse"],
        "mae_h_orig": m_open["mae"],
        "r2_h_orig": m_open["r2"],
        "rmse_h": m_corr["rmse"],
        "mae_h": m_corr["mae"],
        "r2_h": m_corr["r2"],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Neural ODE models")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--v1", type=str, default=None, help="Path to v1 model checkpoint")
    parser.add_argument("--v2", type=str, default=None, help="Path to v2 model checkpoint")
    parser.add_argument("--v3", type=str, default=None, help="Path to v3 model checkpoint")
    parser.add_argument("--save-dir", type=str, default="docs/figures")
    args = parser.parse_args()

    setup_publication_style()
    params = GlassMelterParams()

    # Load test data
    _, df_est = load_lqi_data(args.data_path)
    traj = prepare_lqi_trajectory(df_est, subsample=12)

    models = {}
    descriptions = {
        "Neural ODE v1": "Simulated (LQI-range)",
        "Neural ODE v2": "Simulated (ext. range)",
        "Neural ODE v3": "LQI (60/20/20 split)",
    }

    for label, path in [("Neural ODE v1", args.v1), ("Neural ODE v2", args.v2), ("Neural ODE v3", args.v3)]:
        if path is None:
            continue
        model = load_model(path, params=params)
        results = evaluate_model(
            model, traj,
            DEFAULT_CORRECTION_GAIN_H,
            DEFAULT_CORRECTION_GAIN_V,
        )
        models[label] = {
            "open_loop": {"rmse": results["rmse_h_orig"], "mae": results["mae_h_orig"], "r2": results["r2_h_orig"]},
            "corrected": {"rmse": results["rmse_h"], "mae": results["mae_h"], "r2": results["r2_h"]},
            "description": descriptions[label],
        }

        # Save individual comparison plot
        tag = label.split()[-1].lower()
        plot_state_comparison(results, save_path=f"{args.save_dir}/neural_ode_{tag}_lqi_test")
        print(f"✅ Saved {label} comparison figure")

    if models:
        print_model_comparison(models)

        # Bar chart comparison
        names = list(models.keys())
        plot_model_comparison_bars(
            [models[n]["open_loop"]["rmse"] for n in names],
            [models[n]["open_loop"]["mae"] for n in names],
            [models[n]["open_loop"]["r2"] for n in names],
            [models[n]["corrected"]["rmse"] for n in names],
            [models[n]["corrected"]["mae"] for n in names],
            [models[n]["corrected"]["r2"] for n in names],
            model_labels=[n.split()[-1] for n in names],
            save_path=f"{args.save_dir}/neural_ode_comparison_metrics",
        )
        print("✅ Saved comparison bar chart")


if __name__ == "__main__":
    main()
