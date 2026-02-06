"""
Training loop for Neural ODE models.

Supports:
    - Multi-step rollout training
    - Mixed Precision (AMP) for GPU acceleration
    - Early stopping based on validation loss
    - Learning rate scheduling (ReduceLROnPlateau)
    - Gradient clipping
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from glass_melter_level.models.neural_ode import (
    NeuralODE_GlassMelter,
    integrate_neural_ode,
)
from glass_melter_level.training.trajectory import trajectories_to_device


class NeuralODETrainer:
    """
    Trainer for Neural ODE glass melter models.

    Handles the full training loop with rollout-based trajectory loss,
    GPU acceleration with AMP, early stopping, and LR scheduling.

    Args:
        model: NeuralODE_GlassMelter instance
        device: Training device ('cuda' or 'cpu')
        lr: Learning rate
        weight_decay: AdamW weight decay
        max_grad_norm: Gradient clipping threshold
    """

    def __init__(
        self,
        model: NeuralODE_GlassMelter,
        device: str = "cuda",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        max_grad_norm: float = 1.0,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.lr = lr
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=25
        )

        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.history: Dict[str, list] = {"train": [], "val": []}

    def train(
        self,
        train_trajectories: List[dict],
        val_trajectories: List[dict],
        n_epochs: int = 200,
        rollout_steps: int = 20,
        n_samples_per_traj: int = 3,
        print_every: int = 25,
    ) -> Dict[str, list]:
        """
        Train the Neural ODE model.

        Args:
            train_trajectories: List of training trajectory dicts
            val_trajectories: List of validation trajectory dicts
            n_epochs: Number of training epochs
            rollout_steps: Rollout length for multi-step training
            n_samples_per_traj: Random windows per trajectory per epoch
            print_every: Print interval

        Returns:
            Training history dict with 'train' and 'val' loss lists
        """
        # Move data to device
        train_data = trajectories_to_device(train_trajectories, self.device)
        val_data = trajectories_to_device(val_trajectories, self.device)

        print(f"Training on {self.device} | AMP: {self.use_amp}")
        print(f"  Trajectories: {len(train_data)} train, {len(val_data)} val")
        print(f"  Rollout steps: {rollout_steps} | Epochs: {n_epochs}")
        print("-" * 60)

        start_time = time.time()

        for epoch in range(n_epochs):
            # Training
            train_loss = self._train_epoch(
                train_data, rollout_steps, n_samples_per_traj
            )
            self.history["train"].append(train_loss)

            # Validation
            val_loss = self._validate(val_data)
            self.history["val"].append(val_loss)

            self.scheduler.step(val_loss)

            # Best model tracking
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }

            if (epoch + 1) % print_every == 0 or epoch == 0:
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch + 1:3d}/{n_epochs}: "
                    f"Train={train_loss:.6f}, Val={val_loss:.6f}, "
                    f"Best={self.best_val_loss:.6f} [{elapsed:.0f}s]"
                )

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s")
        print(f"Best validation loss: {self.best_val_loss:.6f}")

        return self.history

    def train_on_single_trajectory(
        self,
        train_trajectory: dict,
        val_trajectory: dict,
        n_epochs: int = 200,
        rollout_steps: int = 20,
        n_samples: int = 10,
        state_weights: Optional[Dict[int, float]] = None,
        print_every: int = 25,
    ) -> Dict[str, list]:
        """
        Train on a single trajectory (e.g., for v3 direct LQI training).

        Args:
            train_trajectory: Single training trajectory dict
            val_trajectory: Single validation trajectory dict
            n_epochs: Number of training epochs
            rollout_steps: Rollout length
            n_samples: Number of random starting points per epoch
            state_weights: Optional per-state loss weights {0: 10.0, 1: 1.0, 2: 0.5}
            print_every: Print interval

        Returns:
            Training history dict
        """
        if state_weights is None:
            state_weights = {0: 10.0, 1: 1.0, 2: 0.5}  # h, v, q_m

        # Move to device
        train_t = train_trajectory["t"].to(self.device)
        train_x = train_trajectory["x"].to(self.device)
        train_u = train_trajectory["u"].to(self.device)

        val_t = val_trajectory["t"].to(self.device)
        val_x = val_trajectory["x"].to(self.device)
        val_u = val_trajectory["u"].to(self.device)
        val_x0 = val_trajectory["x0"].unsqueeze(0).to(self.device)

        train_dt = train_t[1:] - train_t[:-1]
        n_pts = len(train_t)

        print(f"Training on {self.device} (single trajectory) | AMP: {self.use_amp}")
        print("-" * 60)

        start_time = time.time()

        for epoch in range(n_epochs):
            self.model.train()
            epoch_loss = 0.0

            for _ in range(n_samples):
                max_start = max(1, n_pts - rollout_steps - 1)
                start_idx = np.random.randint(0, max_start)

                self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    x0 = train_x[start_idx].unsqueeze(0)
                    x_true = train_x[start_idx : start_idx + rollout_steps + 1]
                    u_seg = train_u[start_idx : start_idx + rollout_steps + 1]
                    dt_seg = train_dt[start_idx : start_idx + rollout_steps]

                    # Forward rollout
                    x_pred = [x0]
                    x_current = x0
                    for i in range(min(rollout_steps, len(dt_seg))):
                        dt = dt_seg[i]
                        u_i = u_seg[i].unsqueeze(0)
                        dx = self.model(x_current, u_i)
                        x_next = x_current + dt * dx
                        x_pred.append(x_next)
                        x_current = x_next

                    x_pred = torch.cat(x_pred, dim=0)
                    x_true = x_true[: len(x_pred)]

                    # Weighted loss
                    loss = torch.tensor(0.0, device=self.device)
                    for state_idx, weight in state_weights.items():
                        loss += weight * F.mse_loss(
                            x_pred[:, state_idx], x_true[:, state_idx]
                        )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += loss.item()

            epoch_loss /= n_samples
            self.history["train"].append(epoch_loss)

            # Validation
            self.model.eval()
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    x_pred_val = [val_x0]
                    x_current = val_x0
                    for i in range(len(val_t) - 1):
                        dt = val_t[i + 1] - val_t[i]
                        u_i = val_u[i].unsqueeze(0)
                        dx = self.model(x_current, u_i)
                        x_next = x_current + dt * dx
                        x_pred_val.append(x_next)
                        x_current = x_next

                    x_pred_val = torch.cat(x_pred_val, dim=0)
                    val_loss = F.mse_loss(x_pred_val[:, 0], val_x[:, 0]).item()
                    self.history["val"].append(val_loss)

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.best_model_state = {
                            k: v.clone()
                            for k, v in self.model.state_dict().items()
                        }

            self.scheduler.step(val_loss)

            if (epoch + 1) % print_every == 0 or epoch == 0:
                h_rmse = np.sqrt(val_loss)
                print(
                    f"Epoch {epoch + 1:4d}: "
                    f"Train={epoch_loss:.6f}, Val h RMSE={h_rmse:.6f} m"
                )

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s")
        print(f"Best validation h RMSE: {np.sqrt(self.best_val_loss):.6f} m")

        return self.history

    def _train_epoch(
        self,
        train_data: List[dict],
        rollout_steps: int,
        n_samples_per_traj: int,
    ) -> float:
        """Run one training epoch."""
        self.model.train()
        epoch_losses = []

        for traj in train_data:
            t_traj = traj["t"]
            x_true = traj["x"]
            u_traj = traj["u"]
            T = len(t_traj)
            max_start = max(1, T - rollout_steps - 1)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                traj_loss = torch.tensor(0.0, device=self.device)

                for _ in range(n_samples_per_traj):
                    start_idx = np.random.randint(0, max_start)
                    end_idx = min(start_idx + rollout_steps, T)

                    x = x_true[start_idx : start_idx + 1]

                    for i in range(start_idx, end_idx - 1):
                        dt_i = t_traj[i + 1] - t_traj[i]
                        u = u_traj[i : i + 1]
                        dx_dt = self.model(x, u)
                        x = x + dt_i * dx_dt
                        target = x_true[i + 1 : i + 2]
                        traj_loss = traj_loss + torch.mean((x - target) ** 2)

                traj_loss = traj_loss / n_samples_per_traj

            self.scaler.scale(traj_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_losses.append(traj_loss.item())

        return np.mean(epoch_losses)

    def _validate(self, val_data: List[dict]) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                for traj in val_data:
                    x_pred = integrate_neural_ode(
                        self.model, traj["x0"], traj["u"], traj["t"]
                    )
                    x_pred_gpu = x_pred.to(self.device)
                    loss = torch.mean((x_pred_gpu - traj["x"]) ** 2).item()
                    val_losses.append(loss)

        return np.mean(val_losses)
