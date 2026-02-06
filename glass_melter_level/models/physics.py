"""
First-principles ODE model for the glass melting furnace.

This module implements the 7-state ODE system derived from mass/energy balances:
    States:  x = [h, v, q_m, z1, z2, z3, z4]
    Inputs:  u = [u1 (charging), u2 (pull rate), w (water)]

The model captures:
    - Transport delay chain (Erlang-4): batch → ready-to-melt
    - Melting lag: first-order dynamics for molten inflow
    - Level dynamics: mass balance with smoothing

Reference: Glass Melter Model Design (PDF equations).
"""

import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from typing import Tuple, Optional

from glass_melter_level.config import GlassMelterParams


class GlassMelterODE:
    """
    Glass melter ODE system (7-state model, no bias b_p).

    Usage:
        model = GlassMelterODE()
        t, sol = model.simulate(t_span=(0, 72), y0=[...], u1=u1_arr, u2=u2_arr, w=w_arr)
    """

    def __init__(self, params: Optional[GlassMelterParams] = None):
        self.params = params or GlassMelterParams()

    def derivatives(self, y: list, t: float, u1_func, u2_func, w_func) -> list:
        """
        Compute state derivatives.

        Args:
            y: State vector [h, v, q_m, z1, z2, z3, z4]
            t: Current time [h]
            u1_func: Interpolation function for charging rate u1(t) [t/h]
            u2_func: Interpolation function for pull rate u2(t) [m³/h]
            w_func: Interpolation function for water flow w(t) [m³/h]

        Returns:
            State derivatives [dh/dt, dv/dt, dq_m/dt, dz1/dt, ..., dz4/dt]
        """
        h, v, q_m, z1, z2, z3, z4 = y
        p = self.params

        u1 = u1_func(t)
        u2 = u2_func(t)
        w = w_func(t)

        # Pull rate (directly u2, no bias)
        q_p = u2

        # Transport delay chain (Erlang N=4)
        a = p.N / p.theta  # = 2 h⁻¹
        dz1_dt = a * (u1 - z1)
        dz2_dt = a * (z1 - z2)
        dz3_dt = a * (z2 - z3)
        dz4_dt = a * (z3 - z4)

        # Melting lag
        qm_ss = p.kc * z4 * (1 + p.kw * (w - p.w0))
        dqm_dt = (-q_m + qm_ss) / p.tau_m

        # Level dynamics
        dh_dt = v
        dv_dt = ((q_m - q_p) / p.A - v) / p.tau_l

        return [dh_dt, dv_dt, dqm_dt, dz1_dt, dz2_dt, dz3_dt, dz4_dt]

    def simulate(
        self,
        t_span: Tuple[float, float],
        y0: list,
        u1: np.ndarray,
        u2: np.ndarray,
        w: np.ndarray,
        n_points: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the glass melter ODE system.

        Args:
            t_span: (t_start, t_end) in hours
            y0: Initial state [h0, v0, qm0, z1_0, z2_0, z3_0, z4_0]
            u1: Charging rate profile [t/h]
            u2: Pull rate profile [m³/h]
            w: Water flow profile [m³/h]
            n_points: Number of evaluation points (default: len(u1))

        Returns:
            Tuple of (t_eval, solution) where solution has shape [n_points, 7]
        """
        if n_points is None:
            n_points = len(u1)

        t_eval = np.linspace(t_span[0], t_span[1], n_points)

        # Create interpolation functions
        t_input = np.linspace(t_span[0], t_span[1], len(u1))
        u1_func = interp1d(t_input, u1, kind="linear", fill_value="extrapolate")
        u2_func = interp1d(t_input, u2, kind="linear", fill_value="extrapolate")
        w_func = interp1d(t_input, w, kind="linear", fill_value="extrapolate")

        sol = odeint(self.derivatives, y0, t_eval, args=(u1_func, u2_func, w_func))

        return t_eval, sol


def simulate_scenario(
    scenario: dict,
    params: Optional[GlassMelterParams] = None,
) -> dict:
    """
    Simulate a single scenario and return results.

    Args:
        scenario: Dictionary with keys: y0, u1, u2, w, t, name
        params: Physical parameters (default: GlassMelterParams())

    Returns:
        Dictionary with simulation results including all states
    """
    model = GlassMelterODE(params)
    T_sim = scenario["t"][-1]

    t_eval, sol = model.simulate(
        t_span=(0, T_sim),
        y0=scenario["y0"],
        u1=scenario["u1"],
        u2=scenario["u2"],
        w=scenario["w"],
    )

    return {
        "name": scenario["name"],
        "t": t_eval,
        "u1": scenario["u1"],
        "u2": scenario["u2"],
        "w": scenario["w"],
        "h": sol[:, 0],
        "v": sol[:, 1],
        "q_m": sol[:, 2],
        "z1": sol[:, 3],
        "z2": sol[:, 4],
        "z3": sol[:, 5],
        "z4": sol[:, 6],
        "h0": scenario["y0"][0],
        "v0": scenario["y0"][1],
        "qm0": scenario["y0"][2],
        "z1_0": scenario["y0"][3],
        "z2_0": scenario["y0"][4],
        "z3_0": scenario["y0"][5],
        "z4_0": scenario["y0"][6],
    }
