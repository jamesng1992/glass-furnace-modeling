"""
Data preprocessing: normalization, subsampling, and trajectory preparation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def compute_norm_params(df: pd.DataFrame, eps: float = 1e-5) -> Dict[str, float]:
    """
    Compute normalization parameters (min/max) for all columns.

    Args:
        df: DataFrame with simulation or LQI data
        eps: Small margin to avoid division by zero

    Returns:
        Dictionary with '{col}_min' and '{col}_max' keys
    """
    norm_params = {}
    for col in df.columns:
        if col == "t":
            norm_params[f"{col}_min"] = df[col].min()
            norm_params[f"{col}_max"] = df[col].max()
        else:
            norm_params[f"{col}_min"] = df[col].min() - eps
            norm_params[f"{col}_max"] = df[col].max() + eps
    return norm_params


def normalize(data: np.ndarray, key: str, norm_params: Dict[str, float]) -> np.ndarray:
    """
    Min-max normalize data to [0, 1].

    Args:
        data: Array to normalize
        key: Column name (used to look up min/max)
        norm_params: Dictionary of normalization parameters

    Returns:
        Normalized array
    """
    vmin = norm_params[f"{key}_min"]
    vmax = norm_params[f"{key}_max"]
    if vmax - vmin < 1e-8:
        return np.zeros_like(data)
    return (data - vmin) / (vmax - vmin)


def denormalize(data: np.ndarray, key: str, norm_params: Dict[str, float]) -> np.ndarray:
    """
    Reverse min-max normalization.

    Args:
        data: Normalized array
        key: Column name
        norm_params: Dictionary of normalization parameters

    Returns:
        Denormalized array
    """
    vmin = norm_params[f"{key}_min"]
    vmax = norm_params[f"{key}_max"]
    return data * (vmax - vmin) + vmin


def subsample_dataframe(
    df: pd.DataFrame,
    factor: int = 12,
    columns: Optional[list] = None,
) -> pd.DataFrame:
    """
    Subsample a DataFrame by a given factor.

    Args:
        df: Input DataFrame
        factor: Subsample every `factor` rows
        columns: Columns to include (default: all)

    Returns:
        Subsampled DataFrame
    """
    if columns is not None:
        return df[columns].iloc[::factor].reset_index(drop=True)
    return df.iloc[::factor].reset_index(drop=True)
