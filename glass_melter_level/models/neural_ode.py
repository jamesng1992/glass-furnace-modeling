"""
Hybrid Neural ODE model for glass melter level prediction.

Combines known physics (transport delay chain, melting lag, level dynamics)
with a learned neural network correction term:

    dx/dt = f_physics(x, u) + 0.1 * f_theta(x, u)

Also provides integration functions with optional state feedback correction
inspired by Luenberger observers.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Union

from glass_melter_level.config import (
    GlassMelterParams,
    STATE_BOUNDS,
    DERIVATIVE_LIMITS,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_N_LAYERS,
    DEFAULT_NEURAL_CORRECTION_SCALE,
    DEFAULT_CORRECTION_GAIN_H,
    DEFAULT_CORRECTION_GAIN_V,
)


class NeuralODE_GlassMelter(nn.Module):
    """
    Hybrid Neural ODE for glass melter state prediction.

    The network predicts dx/dt = f_physics(x, u) + scale * f_nn(x, u)

    States:  x = [h, v, q_m, z1, z2, z3, z4]  (7 states)
    Inputs:  u = [u1, u2, w]                     (3 inputs)

    Args:
        hidden_dim: Number of hidden units per layer
        n_layers: Number of hidden layers
        use_physics: If True, use hybrid physics + neural; if False, pure neural
        params: Physical parameters (default: GlassMelterParams())
        correction_scale: Scale factor for neural correction (default: 0.1)
    """

    def __init__(
        self,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        n_layers: int = DEFAULT_N_LAYERS,
        use_physics: bool = True,
        params: Optional[GlassMelterParams] = None,
        correction_scale: float = DEFAULT_NEURAL_CORRECTION_SCALE,
    ):
        super().__init__()

        self.use_physics = use_physics
        self.correction_scale = correction_scale
        self.state_dim = 7
        self.input_dim = 3

        # Physical parameters
        p = params or GlassMelterParams()
        self.A = p.A
        self.tau_m = p.tau_m
        self.theta = p.theta
        self.N = p.N
        self.tau_l = p.tau_l
        self.kc = p.kc
        self.w0 = p.w0
        self.kw = p.kw

        # Neural network for corrections (or full dynamics)
        total_input = self.state_dim + self.input_dim  # 10
        layers = []
        in_features = total_input
        for _ in range(n_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.Tanh())
            in_features = hidden_dim
        layers.append(nn.Linear(hidden_dim, self.state_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for corrections."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def physics_derivatives(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Compute derivatives using known physics.

        Args:
            x: States [batch, 7] = [h, v, q_m, z1, z2, z3, z4]
            u: Inputs [batch, 3] = [u1, u2, w]

        Returns:
            dx/dt [batch, 7]
        """
        h, v, q_m = x[:, 0], x[:, 1], x[:, 2]
        z1, z2, z3, z4 = x[:, 3], x[:, 4], x[:, 5], x[:, 6]
        u1, u2, w = u[:, 0], u[:, 1], u[:, 2]

        # Transport delay chain (Erlang-4)
        a = self.N / self.theta  # 2.0 h⁻¹
        dz1_dt = a * (u1 - z1)
        dz2_dt = a * (z1 - z2)
        dz3_dt = a * (z2 - z3)
        dz4_dt = a * (z3 - z4)

        # Melting lag
        qm_ss = self.kc * z4 * (1 + self.kw * (w - self.w0))
        dqm_dt = (-q_m + qm_ss) / self.tau_m

        # Level dynamics (q_p = u2 directly, no bias)
        q_p = u2
        dh_dt = v
        dv_dt = ((q_m - q_p) / self.A - v) / self.tau_l

        return torch.stack(
            [dh_dt, dv_dt, dqm_dt, dz1_dt, dz2_dt, dz3_dt, dz4_dt], dim=1
        )

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Compute dx/dt given current state x and input u.

        Args:
            x: States [batch, 7]
            u: Inputs [batch, 3]

        Returns:
            dx/dt [batch, 7]
        """
        if self.use_physics:
            dx_physics = self.physics_derivatives(x, u)
            xu = torch.cat([x, u], dim=1)
            dx_correction = self.net(xu)
            return dx_physics + self.correction_scale * dx_correction
        else:
            xu = torch.cat([x, u], dim=1)
            return self.net(xu)


def _to_device_tensor(
    data: Union[torch.Tensor, np.ndarray, list],
    device: torch.device,
) -> torch.Tensor:
    """Convert data to a float32 tensor on the specified device."""
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)
    return data.to(device)


def _clamp_derivatives(dx_dt: torch.Tensor) -> torch.Tensor:
    """Clamp derivatives to prevent explosive growth."""
    limits = DERIVATIVE_LIMITS
    dx = dx_dt.clone()
    dx[0, 0] = torch.clamp(dx_dt[0, 0], -limits["dh"], limits["dh"])
    dx[0, 1] = torch.clamp(dx_dt[0, 1], -limits["dv"], limits["dv"])
    dx[0, 2] = torch.clamp(dx_dt[0, 2], -limits["dqm"], limits["dqm"])
    for j in range(3, 7):
        dx[0, j] = torch.clamp(dx_dt[0, j], -limits["dz"], limits["dz"])
    return dx


def _clamp_states(x: torch.Tensor) -> torch.Tensor:
    """Clamp states to physical bounds."""
    bounds = STATE_BOUNDS
    x[0, 0] = torch.clamp(x[0, 0], bounds["h"][0], bounds["h"][1])
    x[0, 1] = torch.clamp(x[0, 1], bounds["v"][0], bounds["v"][1])
    x[0, 2] = torch.clamp(x[0, 2], bounds["q_m"][0], bounds["q_m"][1])
    for j in range(3, 7):
        x[0, j] = torch.clamp(x[0, j], bounds["z"][0], bounds["z"][1])
    return x


def _euler_step(model, x, u, dt_i):
    """Perform one Euler step with sub-stepping for stability."""
    dx_dt = model(x, u)
    dx_dt = _clamp_derivatives(dx_dt)

    if dt_i > 0.05:  # If time step > 3 min, use sub-steps
        n_substeps = int(np.ceil(dt_i / 0.05))
        sub_dt = dt_i / n_substeps
        for _ in range(n_substeps):
            dx_dt = model(x, u)
            dx_dt = _clamp_derivatives(dx_dt)
            x = x + sub_dt * dx_dt
            x = _clamp_states(x)
    else:
        x = x + dt_i * dx_dt
        x = _clamp_states(x)

    return x


def integrate_neural_ode(
    model: NeuralODE_GlassMelter,
    x0: Union[torch.Tensor, np.ndarray],
    u_trajectory: Union[torch.Tensor, np.ndarray],
    t_trajectory: Union[torch.Tensor, np.ndarray],
) -> torch.Tensor:
    """
    Integrate the Neural ODE forward in time (open-loop, no correction).

    Uses Euler method with adaptive sub-stepping and state clamping
    for numerical stability.

    Args:
        model: NeuralODE_GlassMelter instance
        x0: Initial state [7]
        u_trajectory: Input trajectory [T, 3]
        t_trajectory: Time points [T]

    Returns:
        x_trajectory: State trajectory [T, 7] on CPU
    """
    model.eval()
    device = next(model.parameters()).device
    T = len(t_trajectory)

    x0 = _to_device_tensor(x0, device)
    u_trajectory = _to_device_tensor(u_trajectory, device)
    t_trajectory = _to_device_tensor(t_trajectory, device)

    x_trajectory = torch.zeros(T, 7, device=device)
    x_trajectory[0] = x0.clone()
    x = x0.unsqueeze(0)  # [1, 7]

    with torch.no_grad():
        for i in range(1, T):
            dt_i = (t_trajectory[i] - t_trajectory[i - 1]).item()
            u = u_trajectory[i - 1 : i]
            x = _euler_step(model, x, u, dt_i)
            x_trajectory[i] = x.squeeze(0)

    return x_trajectory.cpu()


def integrate_neural_ode_with_correction(
    model: NeuralODE_GlassMelter,
    x0: Union[torch.Tensor, np.ndarray],
    u_trajectory: Union[torch.Tensor, np.ndarray],
    t_trajectory: Union[torch.Tensor, np.ndarray],
    h_measured: Optional[Union[torch.Tensor, np.ndarray]] = None,
    correction_gain: float = DEFAULT_CORRECTION_GAIN_H,
    velocity_correction_gain: float = DEFAULT_CORRECTION_GAIN_V,
) -> torch.Tensor:
    """
    Integrate Neural ODE with state feedback correction.

    Implements an observer-like correction inspired by Luenberger observers:
        h_hat = h_hat_pred + L_h * (h_meas - h_hat_pred)
        v_hat = v_hat_pred + L_v * (h_meas - h_hat_pred)

    This prevents velocity-induced level drift over long prediction horizons.

    Args:
        model: NeuralODE_GlassMelter instance
        x0: Initial state [7]
        u_trajectory: Input trajectory [T, 3]
        t_trajectory: Time points [T]
        h_measured: Measured level trajectory [T] (optional)
        correction_gain: Level correction gain L_h (default: 0.3)
        velocity_correction_gain: Velocity correction gain L_v (default: 0.1)

    Returns:
        x_trajectory: Corrected state trajectory [T, 7] on CPU
    """
    model.eval()
    device = next(model.parameters()).device
    T = len(t_trajectory)

    x0 = _to_device_tensor(x0, device)
    u_trajectory = _to_device_tensor(u_trajectory, device)
    t_trajectory = _to_device_tensor(t_trajectory, device)

    x_trajectory = torch.zeros(T, 7, device=device)
    x_trajectory[0] = x0.clone()
    x = x0.unsqueeze(0).clone()

    with torch.no_grad():
        for i in range(1, T):
            dt_i = (t_trajectory[i] - t_trajectory[i - 1]).item()
            u = u_trajectory[i - 1 : i]

            # Forward step
            x = _euler_step(model, x, u, dt_i)

            # State feedback correction
            if h_measured is not None and correction_gain > 0:
                h_pred = x[0, 0].item()
                if isinstance(h_measured, torch.Tensor):
                    h_meas = h_measured[i].item()
                else:
                    h_meas = h_measured[i]
                h_error = h_meas - h_pred

                # Level correction: L_h * e_h
                x[0, 0] = x[0, 0] + correction_gain * h_error

                # Velocity correction: L_v * e_h (derivative action)
                x[0, 1] = x[0, 1] + velocity_correction_gain * h_error

            x_trajectory[i] = x.squeeze(0)

    return x_trajectory.cpu()


def save_model(
    model: NeuralODE_GlassMelter,
    filepath: str,
    training_info: Optional[dict] = None,
):
    """
    Save a trained Neural ODE model to disk.

    Args:
        model: Trained NeuralODE_GlassMelter instance
        filepath: Output .pt file path
        training_info: Optional training metadata
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "hidden_dim": model.net[0].in_features - model.state_dim - model.input_dim
                          if hasattr(model.net[0], "in_features") else DEFAULT_HIDDEN_DIM,
            "n_layers": sum(1 for m in model.net if isinstance(m, nn.Linear)) - 1,
            "use_physics": model.use_physics,
            "correction_scale": model.correction_scale,
        },
        "physics_params": {
            "A": model.A,
            "tau_m": model.tau_m,
            "theta": model.theta,
            "N": model.N,
            "tau_l": model.tau_l,
            "kc": model.kc,
            "w0": model.w0,
            "kw": model.kw,
        },
    }
    if training_info:
        checkpoint["training_info"] = training_info

    torch.save(checkpoint, filepath)


def load_model(filepath: str, device: str = "cpu") -> NeuralODE_GlassMelter:
    """
    Load a trained Neural ODE model from disk.

    Args:
        filepath: Path to .pt checkpoint file
        device: Device to load model onto

    Returns:
        Loaded NeuralODE_GlassMelter model
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    config = checkpoint["model_config"]

    model = NeuralODE_GlassMelter(
        hidden_dim=config.get("hidden_dim", DEFAULT_HIDDEN_DIM),
        n_layers=config.get("n_layers", DEFAULT_N_LAYERS),
        use_physics=config.get("use_physics", True),
        correction_scale=config.get("correction_scale", DEFAULT_NEURAL_CORRECTION_SCALE),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model
