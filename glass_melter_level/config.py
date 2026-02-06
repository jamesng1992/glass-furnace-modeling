"""
Physical parameters and constants for the glass melting furnace model.

These parameters are derived from the Glass Melter Model Design document
and match the 7-state ODE system (without bias term b_p).
"""

from dataclasses import dataclass


@dataclass
class GlassMelterParams:
    """
    Parameters for the glass melter ODE model (7-state, no b_p).

    Reference: Glass Melter Model Design (PDF equations).

    Attributes:
        A: Melt surface area [m²]
        rho: Molten glass density [kg/m³]
        eta: Batch-to-melt yield [-]
        kc: Batch-to-melt conversion gain [m³/h per t/h] = (eta × 1000) / rho
        tau_m: Melting first-order time constant [h]
        theta: Total transport delay [h]
        N: Number of delay chain cells [-]
        tau_l: Level-rate smoothing time constant [h]
        w0: Nominal water flow [m³/h]
        kw: Water sensitivity coefficient [1/(m³/h)]
        kp: Pull calibration factor [-]
    """

    # Geometry
    A: float = 60.0         # [m²] Melt surface area
    rho: float = 2400.0     # [kg/m³] Molten glass density

    # Batch-to-molten conversion
    eta: float = 0.95       # [-] Yield
    kc: float = 0.3958      # [m³/h per t/h] = (eta × 1000) / rho

    # Melting lag
    tau_m: float = 3.0      # [h] Melting first-order lag

    # Transport delay chain (Erlang N=4)
    theta: float = 2.0      # [h] Total transport delay
    N: int = 4              # [-] Number of delay cells (tau_d = theta/N = 0.5 h)

    # Level-rate smoothing
    tau_l: float = 0.10     # [h] Level dynamics lag (~6 min)

    # Water mixing factor
    w0: float = 0.20        # [m³/h] Nominal water flow
    kw: float = 0.02        # [1/(m³/h)] Water sensitivity

    # Production calibration
    kp: float = 1.0         # [-] Pull calibration factor


# Glass density for unit conversion (tonnes/m³)
GLASS_DENSITY = 2.4  # tonnes/m³

# State and input column names (consistent across the codebase)
STATE_COLS = ["h", "v", "q_m", "z1", "z2", "z3", "z4"]
INPUT_COLS = ["u1", "u2", "w"]

# Number of states and inputs
N_STATES = 7
N_INPUTS = 3

# Default correction gains for state feedback
DEFAULT_CORRECTION_GAIN_H = 0.3   # Level correction gain (L_h)
DEFAULT_CORRECTION_GAIN_V = 0.1   # Velocity correction gain (L_v)

# Physical bounds for state clamping during integration
STATE_BOUNDS = {
    "h": (-0.5, 2.0),       # Level [m]
    "v": (-0.2, 0.2),       # Velocity [m/h]
    "q_m": (0.0, 15.0),     # Melting rate [m³/h]
    "z": (0.0, 50.0),       # Delay states [t/h]
}

# Derivative clamping limits
DERIVATIVE_LIMITS = {
    "dh": 0.1,     # Max |dh/dt| [m/h]
    "dv": 0.05,    # Max |dv/dt| [m/h²]
    "dqm": 2.0,    # Max |dq_m/dt| [m³/h²]
    "dz": 10.0,    # Max |dz/dt| [t/h²]
}

# Default Neural ODE architecture
DEFAULT_HIDDEN_DIM = 64
DEFAULT_N_LAYERS = 3
DEFAULT_NEURAL_CORRECTION_SCALE = 0.1  # Scale factor for neural correction
