"""
Publication-quality plot functions for Neural ODE evaluation.

Provides reusable plotting routines for time-series state comparison
and bar-chart metric comparison across model variants.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from glass_melter_level.visualization.style import get_pub_colors, add_subplot_label


def plot_state_comparison(
    results: dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 9),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot open-loop vs corrected predictions for h, v, q_m.

    Args:
        results: Dictionary with keys:
            t, h_true, h_pred, h_pred_orig, v_true, v_pred, v_pred_orig,
            qm_true, qm_pred, qm_pred_orig, rmse_h, rmse_h_orig
        save_path: If given, save figure (without extension — saves .pdf + .png)
        figsize: Figure size in inches
        title: Optional supertitle

    Returns:
        matplotlib Figure
    """
    colors = get_pub_colors()
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    r = results
    t = r["t"]

    # RMSE values
    rmse_h_orig = float(np.sqrt(np.mean((r["h_pred_orig"] - r["h_true"]) ** 2)))
    rmse_h_corr = float(r.get("rmse_h", np.sqrt(np.mean((r["h_pred"] - r["h_true"]) ** 2))))

    # (a) Level h
    ax = axes[0]
    ax.plot(t, r["h_true"], color=colors["true"], label="True (LQI)", lw=2.5)
    ax.plot(
        t,
        r["h_pred_orig"],
        color=colors["open_loop"],
        ls="--",
        label=f"Open-Loop (RMSE = {rmse_h_orig:.4f} m)",
        lw=2.5,
    )
    ax.plot(
        t,
        r["h_pred"],
        color=colors["corrected"],
        label=f"Corrected (RMSE = {rmse_h_corr:.4f} m)",
        lw=2.5,
    )
    ax.set_ylabel("Level $h$ [m]")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.35),
        ncol=3,
        frameon=True,
        fontsize=16,
    )
    add_subplot_label(ax, "(a)")

    # (b) Velocity v
    ax = axes[1]
    ax.plot(t, r["v_true"], color=colors["true"], label="True (LQI)", lw=2.5)
    if "v_pred_orig" in r:
        ax.plot(
            t,
            r["v_pred_orig"],
            color=colors["open_loop"],
            ls="--",
            label="Open-Loop",
            lw=2.5,
        )
    ax.plot(t, r["v_pred"], color=colors["corrected"], label="Corrected", lw=2.5)
    ax.set_ylabel("Velocity $v$ [m/h]")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,
        frameon=True,
        fontsize=16,
    )
    add_subplot_label(ax, "(b)")

    # (c) Melting rate q_m
    ax = axes[2]
    ax.plot(t, r["qm_true"], color=colors["true"], label="True (LQI)", lw=2.5)
    if "qm_pred_orig" in r:
        ax.plot(
            t,
            r["qm_pred_orig"],
            color=colors["open_loop"],
            ls="--",
            label="Open-Loop",
            lw=2.5,
        )
    ax.plot(t, r["qm_pred"], color=colors["corrected"], label="Corrected", lw=2.5)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Melting Rate $q_m$ [m³/h]")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,
        frameon=True,
        fontsize=16,
    )
    add_subplot_label(ax, "(c)")

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.45)

    if title:
        fig.suptitle(title, y=1.02)

    if save_path:
        plt.savefig(f"{save_path}.pdf", dpi=300, bbox_inches="tight", format="pdf")
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

    return fig


def plot_model_comparison_bars(
    rmse_open: List[float],
    mae_open: List[float],
    r2_open: List[float],
    rmse_corr: List[float],
    mae_corr: List[float],
    r2_corr: List[float],
    model_labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Create a 3-panel bar chart comparing open-loop vs corrected metrics.

    Args:
        rmse_open: RMSE values for open-loop models [m]
        mae_open: MAE values for open-loop models [m]
        r2_open: R² values for open-loop models
        rmse_corr: RMSE values for corrected models [m]
        mae_corr: MAE values for corrected models [m]
        r2_corr: R² values for corrected models
        model_labels: X-axis labels (default: ['v1', 'v2', 'v3'])
        save_path: Save path (without extension)
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    if model_labels is None:
        model_labels = ["v1", "v2", "v3"]

    colors = get_pub_colors()
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    x = np.arange(len(model_labels))
    width = 0.35

    color_open = colors["open_loop"]
    color_corr = colors["corrected"]

    # (a) RMSE
    ax = axes[0]
    ax.bar(x - width / 2, rmse_open, width, label="Open-loop",
           color=color_open, edgecolor="black", linewidth=1.5)
    ax.bar(x + width / 2, rmse_corr, width, label="Corrected",
           color=color_corr, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("RMSE [m]")
    ax.set_xlabel("N-ODE Model")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_yscale("log")
    ax.set_ylim([1e-5, 0.1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22),
              ncol=2, frameon=True, fontsize=16)
    add_subplot_label(ax, "(a)")

    # (b) MAE
    ax = axes[1]
    ax.bar(x - width / 2, mae_open, width, label="Open-loop",
           color=color_open, edgecolor="black", linewidth=1.5)
    ax.bar(x + width / 2, mae_corr, width, label="Corrected",
           color=color_corr, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("MAE [m]")
    ax.set_xlabel("N-ODE Model")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_yscale("log")
    ax.set_ylim([1e-5, 0.1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22),
              ncol=2, frameon=True, fontsize=16)
    add_subplot_label(ax, "(b)")

    # (c) R²
    ax = axes[2]
    ax.bar(x - width / 2, r2_open, width, label="Open-loop",
           color=color_open, edgecolor="black", linewidth=1.5)
    ax.bar(x + width / 2, r2_corr, width, label="Corrected",
           color=color_corr, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("$R^2$")
    ax.set_xlabel("N-ODE Model")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_ylim([0, 1.08])
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22),
              ncol=2, frameon=True, fontsize=16)
    add_subplot_label(ax, "(c)")

    plt.tight_layout()
    plt.subplots_adjust(top=0.82)

    if save_path:
        plt.savefig(f"{save_path}.pdf", dpi=300, bbox_inches="tight", format="pdf")
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

    return fig
