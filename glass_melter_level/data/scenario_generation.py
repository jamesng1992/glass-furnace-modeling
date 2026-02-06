"""
Scenario generation for Neural ODE training data.

Generates diverse simulation scenarios with input profiles (u1, u2, w)
that match or extend the LQI control data operating ranges.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def get_lqi_initial_conditions(
    lqi_ic: dict, vary_slightly: bool = False
) -> list:
    """
    Get initial conditions matching LQI data, optionally with small variations.

    Args:
        lqi_ic: Dictionary with keys h0, v0, qm0, z1_0, ..., z4_0
        vary_slightly: If True, add small random perturbations

    Returns:
        List of 7 initial conditions [h0, v0, qm0, z1_0, z2_0, z3_0, z4_0]
    """
    ic = dict(lqi_ic)  # Copy

    if vary_slightly:
        ic["h0"] = ic["h0"] + np.random.uniform(-0.01, 0.01)
        ic["v0"] = ic["v0"] + np.random.uniform(-0.002, 0.002)
        ic["qm0"] = ic["qm0"] + np.random.uniform(-0.1, 0.1)
        for key in ["z1_0", "z2_0", "z3_0", "z4_0"]:
            ic[key] = ic[key] + np.random.uniform(-0.5, 0.5)

    return [ic["h0"], ic["v0"], ic["qm0"],
            ic["z1_0"], ic["z2_0"], ic["z3_0"], ic["z4_0"]]


def generate_u2_profile(
    t: np.ndarray,
    profile_type: str = "step",
    u2_range: Tuple[float, float] = (4.25, 6.75),
) -> np.ndarray:
    """
    Generate a pull rate (u2) profile.

    Args:
        t: Time array [hours]
        profile_type: One of 'constant', 'step', 'ramp', 'sinusoidal'
        u2_range: (min, max) pull rate range [m³/h]

    Returns:
        u2 profile array
    """
    u2_min, u2_max = u2_range
    T_sim = t[-1]

    if profile_type == "constant":
        return np.full_like(t, np.random.uniform(u2_min, u2_max))

    elif profile_type == "step":
        u2 = np.full_like(t, np.random.uniform(u2_min, (u2_min + u2_max) / 2))
        n_steps = np.random.randint(2, 6)
        step_times = np.sort(np.random.uniform(5, T_sim - 5, n_steps))
        for st in step_times:
            u2[t >= st] = np.random.uniform(u2_min, u2_max)
        return u2

    elif profile_type == "ramp":
        return np.linspace(
            np.random.uniform(u2_min, (u2_min + u2_max) / 2),
            np.random.uniform((u2_min + u2_max) / 2, u2_max),
            len(t),
        )

    elif profile_type == "sinusoidal":
        mean_u2 = (u2_min + u2_max) / 2
        amplitude = (u2_max - u2_min) / 4
        freq = np.random.uniform(0.5, 2) * 2 * np.pi / T_sim
        return mean_u2 + amplitude * np.sin(freq * t + np.random.uniform(0, 2 * np.pi))

    else:
        raise ValueError(f"Unknown profile type: {profile_type}")


def generate_u1_profile(
    t: np.ndarray,
    u2_profile: np.ndarray,
    u1_range: Tuple[float, float] = (0.0, 22.17),
    scenario_type: str = "balanced",
) -> np.ndarray:
    """
    Generate a charging rate (u1) profile.

    Args:
        t: Time array [hours]
        u2_profile: Corresponding pull rate profile
        u1_range: (min, max) charging rate range [t/h]
        scenario_type: One of 'low_charging', 'medium_charging', 'high_charging',
                        'step_response', 'ramp', 'balanced'

    Returns:
        u1 profile array
    """
    u1_min, u1_max = u1_range
    T_sim = t[-1]

    if scenario_type == "low_charging":
        base = np.random.uniform(0, 10)
        noise = np.random.normal(0, 1.5, len(t))
        return np.clip(base + noise, u1_min, 12)

    elif scenario_type == "medium_charging":
        base = np.random.uniform(8, 20)
        noise = np.random.normal(0, 2.5, len(t))
        return np.clip(base + noise, u1_min, u1_max)

    elif scenario_type == "high_charging":
        base = np.random.uniform(18, min(28, u1_max))
        noise = np.random.normal(0, 2.0, len(t))
        return np.clip(base + noise, 12, u1_max)

    elif scenario_type == "step_response":
        u1 = np.full_like(t, np.random.uniform(8, 20))
        n_steps = np.random.randint(2, 5)
        step_times = np.sort(np.random.uniform(5, T_sim - 5, n_steps))
        for st in step_times:
            u1[t >= st] = np.random.uniform(u1_min, u1_max)
        return np.clip(u1, u1_min, u1_max)

    elif scenario_type == "ramp":
        start = np.random.uniform(u1_min, 15)
        end = np.random.uniform(15, u1_max)
        if np.random.random() > 0.5:
            start, end = end, start
        return np.linspace(start, end, len(t))

    else:  # balanced — controller-like behavior
        gain = np.random.uniform(1.5, 3.0)
        offset = np.random.uniform(0, 5)
        u1 = gain * u2_profile + offset
        noise = np.random.normal(0, 1.0, len(t))

        # Add step changes
        n_adjustments = np.random.randint(3, 8)
        adjustment_times = np.sort(np.random.uniform(2, T_sim - 2, n_adjustments))
        for at in adjustment_times:
            idx = np.argmin(np.abs(t - at))
            adjustment = np.random.uniform(-5, 8)
            decay = np.exp(-(t[idx:] - at) / np.random.uniform(2, 8))
            u1[idx:] += adjustment * decay

        return np.clip(u1 + noise, u1_min, u1_max)


def generate_scenarios(
    n_scenarios: int = 100,
    t_sim: float = 72.0,
    dt: float = 1 / 60,
    lqi_ic: Optional[dict] = None,
    u1_range: Tuple[float, float] = (0.0, 22.17),
    u2_range: Tuple[float, float] = (4.25, 6.75),
    seed: int = 42,
) -> List[dict]:
    """
    Generate a set of simulation scenarios for Neural ODE training.

    Args:
        n_scenarios: Number of scenarios to generate
        t_sim: Simulation duration [hours]
        dt: Time step [hours]
        lqi_ic: LQI initial conditions dict (if None, uses defaults)
        u1_range: Charging rate range [t/h]
        u2_range: Pull rate range [m³/h]
        seed: Random seed

    Returns:
        List of scenario dictionaries with keys:
            name, y0, u1, u2, w, t, exact_lqi_ic
    """
    np.random.seed(seed)

    N_points = int(t_sim / dt) + 1
    t = np.linspace(0, t_sim, N_points)

    if lqi_ic is None:
        # Default initial conditions
        lqi_ic = {
            "h0": 0.9, "v0": 0.0, "qm0": 5.0,
            "z1_0": 10.0, "z2_0": 10.0, "z3_0": 10.0, "z4_0": 10.0,
        }

    profile_types = ["step", "ramp", "constant", "sinusoidal"]
    u1_types = ["low_charging", "medium_charging", "high_charging",
                "step_response", "ramp", "balanced"]

    scenarios = []
    for i in range(n_scenarios):
        use_exact_ic = np.random.random() < 0.8
        y0 = get_lqi_initial_conditions(lqi_ic, vary_slightly=not use_exact_ic)

        u2_type = np.random.choice(profile_types)
        u2_profile = generate_u2_profile(t, u2_type, u2_range)

        u1_type = np.random.choice(u1_types)
        u1_profile = generate_u1_profile(t, u2_profile, u1_range, u1_type)

        scenarios.append({
            "name": f"Scenario_{i}",
            "y0": y0,
            "u1": u1_profile,
            "u2": u2_profile,
            "w": np.zeros_like(t),
            "t": t,
            "exact_lqi_ic": use_exact_ic,
        })

    return scenarios
