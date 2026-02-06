"""
Trajectory preparation utilities for Neural ODE training.

Converts simulation data and LQI DataFrames into tensor-based trajectory
dictionaries suitable for the training loop.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from glass_melter_level.config import STATE_COLS, INPUT_COLS


def prepare_trajectory(
    sim: dict,
    subsample: int = 10,
) -> dict:
    """
    Convert a simulation result dictionary to a trajectory for training.

    Args:
        sim: Simulation result with keys: t, h, v, q_m, z1-z4, u1, u2, w
        subsample: Subsample factor

    Returns:
        Dictionary with keys: t, x, u, x0 (all tensors)
    """
    n_pts = len(sim["t"])
    indices = list(range(0, n_pts, subsample))

    t_traj = torch.tensor(sim["t"][indices], dtype=torch.float32)

    x_traj = torch.stack([
        torch.tensor(sim["h"][indices], dtype=torch.float32),
        torch.tensor(sim["v"][indices], dtype=torch.float32),
        torch.tensor(sim["q_m"][indices], dtype=torch.float32),
        torch.tensor(sim["z1"][indices], dtype=torch.float32),
        torch.tensor(sim["z2"][indices], dtype=torch.float32),
        torch.tensor(sim["z3"][indices], dtype=torch.float32),
        torch.tensor(sim["z4"][indices], dtype=torch.float32),
    ], dim=1)

    u_traj = torch.stack([
        torch.tensor(sim["u1"][indices], dtype=torch.float32),
        torch.tensor(sim["u2"][indices], dtype=torch.float32),
        torch.tensor(sim["w"][indices], dtype=torch.float32),
    ], dim=1)

    return {"t": t_traj, "x": x_traj, "u": u_traj, "x0": x_traj[0]}


def prepare_lqi_trajectory(
    df: pd.DataFrame,
    subsample: int = 12,
) -> dict:
    """
    Convert a standardized LQI DataFrame to a trajectory for training.

    Args:
        df: Standardized DataFrame with columns: t, u1, u2, w, h, v, q_m, z1-z4
        subsample: Subsample factor (12 ≈ 1-minute resolution for ~5s data)

    Returns:
        Dictionary with keys: t, x, u, x0 (all tensors)
    """
    n_pts = len(df)
    indices = list(range(0, n_pts, subsample))

    t_traj = torch.tensor(df["t"].values[indices], dtype=torch.float32)

    x_traj = torch.stack([
        torch.tensor(df[col].values[indices], dtype=torch.float32)
        for col in STATE_COLS
    ], dim=1)

    u_traj = torch.stack([
        torch.tensor(df[col].values[indices], dtype=torch.float32)
        for col in INPUT_COLS
    ], dim=1)

    return {"t": t_traj, "x": x_traj, "u": u_traj, "x0": x_traj[0]}


def split_lqi_data(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split LQI data into train/val/test sets (temporal split).

    Args:
        df: Full LQI DataFrame
        train_frac: Fraction for training (default: 0.6)
        val_frac: Fraction for validation (default: 0.2)

    Returns:
        Tuple of (df_train, df_val, df_test)
    """
    n_total = len(df)
    n_train = int(train_frac * n_total)
    n_val = int(val_frac * n_total)

    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_val = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    df_test = df.iloc[n_train + n_val:].reset_index(drop=True)

    return df_train, df_val, df_test


def trajectories_to_device(
    trajectories: List[dict],
    device: torch.device,
) -> List[dict]:
    """
    Move all trajectory tensors to the specified device.

    Args:
        trajectories: List of trajectory dicts with tensor values
        device: Target device

    Returns:
        List of trajectory dicts on the device
    """
    return [
        {k: v.to(device) if isinstance(v, torch.Tensor) else v
         for k, v in traj.items()}
        for traj in trajectories
    ]
