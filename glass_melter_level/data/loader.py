"""
Data loading utilities for glass melter LQI control data.

Handles loading from Excel files with TrueStates and EstimatedStates sheets,
and creates standardized DataFrames with consistent column naming.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def load_lqi_data(
    filepath: str,
    sheet_true: str = "Signals_TrueStates",
    sheet_est: str = "Signals_EstimatedStates",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load LQI control data from Excel file.

    The Excel file contains two sheets:
    - Signals_TrueStates: Ground truth system states
    - Signals_EstimatedStates: Observer-estimated states

    Args:
        filepath: Path to the Excel file (e.g., 'glass_melter_LQI_outputs 1.xlsx')
        sheet_true: Name of the TrueStates sheet
        sheet_est: Name of the EstimatedStates sheet

    Returns:
        Tuple of (df_true, df_est) with standardized column names:
            t, u1, u2, w, h, v, q_m, z1, z2, z3, z4
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    xl = pd.ExcelFile(filepath)

    # Load TrueStates
    df_raw_true = pd.read_excel(xl, sheet_name=sheet_true)
    df_true = _standardize_lqi_dataframe(df_raw_true, source="true")

    # Load EstimatedStates
    df_raw_est = pd.read_excel(xl, sheet_name=sheet_est)
    df_est = _standardize_lqi_dataframe(df_raw_est, source="estimated")

    return df_true, df_est


def _standardize_lqi_dataframe(df: pd.DataFrame, source: str = "true") -> pd.DataFrame:
    """
    Create a standardized DataFrame with consistent column names.

    Column mappings:
        'u' → u1 (charging rate, t/h)
        'q_p' → u2 (pull rate, m³/h)
        'h_abs' → h (glass level, m)
        'x_v' / 'xhat_v' → v (level rate, m/h)
        'q_m' → q_m (molten inflow, m³/h)
        'x_z1'-'x_z4' / 'xhat_z1'-'xhat_z4' → z1-z4 (delay states)

    Args:
        df: Raw DataFrame from Excel
        source: 'true' for TrueStates, 'estimated' for EstimatedStates

    Returns:
        Standardized DataFrame
    """
    if source == "true":
        v_col = "x_v"
        z_prefix = "x_z"
    else:
        v_col = "xhat_v"
        z_prefix = "xhat_z"

    return pd.DataFrame({
        "t": df["t"],
        "u1": df["u"],
        "u2": df["q_p"],
        "w": 0.0,  # Water flow not available in LQI data
        "h": df["h_abs"],
        "v": df[v_col],
        "q_m": df["q_m"],
        "z1": df[f"{z_prefix}1"],
        "z2": df[f"{z_prefix}2"],
        "z3": df[f"{z_prefix}3"],
        "z4": df[f"{z_prefix}4"],
    })


def get_initial_conditions(df: pd.DataFrame) -> dict:
    """
    Extract initial conditions from a standardized DataFrame.

    Args:
        df: Standardized DataFrame with columns [h, v, q_m, z1, z2, z3, z4]

    Returns:
        Dictionary of initial conditions
    """
    return {
        "h0": df["h"].iloc[0],
        "v0": df["v"].iloc[0],
        "qm0": df["q_m"].iloc[0],
        "z1_0": df["z1"].iloc[0],
        "z2_0": df["z2"].iloc[0],
        "z3_0": df["z3"].iloc[0],
        "z4_0": df["z4"].iloc[0],
    }


def get_data_ranges(df: pd.DataFrame) -> dict:
    """
    Compute min/max ranges for all inputs and states.

    Args:
        df: Standardized DataFrame

    Returns:
        Dictionary with (min, max) tuples for each column
    """
    cols = ["u1", "u2", "h", "v", "q_m", "z1", "z2", "z3", "z4"]
    return {col: (df[col].min(), df[col].max()) for col in cols if col in df.columns}


def load_csv_data(filepath: str) -> pd.DataFrame:
    """Load a CSV data file."""
    return pd.read_csv(filepath)
