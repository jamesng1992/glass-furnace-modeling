"""
Train Neural ODE v3: directly on LQI data with 60/20/20 split.

Usage:
    python scripts/train_v3.py --data-path data/LQI_data.xlsx --epochs 200
"""

import argparse
import torch

from glass_melter_level.config import GlassMelterParams
from glass_melter_level.data.loader import load_lqi_data
from glass_melter_level.training.trajectory import (
    prepare_lqi_trajectory,
    split_lqi_data,
)
from glass_melter_level.training.trainer import NeuralODETrainer
from glass_melter_level.models.neural_ode import NeuralODE_GlassMelter, save_model


def main():
    parser = argparse.ArgumentParser(description="Train Neural ODE v3 (LQI direct)")
    parser.add_argument("--data-path", type=str, required=True, help="Path to LQI Excel file")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--rollout", type=int, default=20, help="Rollout steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--subsample", type=int, default=12, help="Subsample factor")
    parser.add_argument("--train-frac", type=float, default=0.6, help="Training fraction")
    parser.add_argument("--val-frac", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--save-path", type=str, default="trained_models/neural_ode_v3.pt")
    args = parser.parse_args()

    # Load and split LQI data
    _, df_est = load_lqi_data(args.data_path)
    df_train, df_val, df_test = split_lqi_data(df_est, args.train_frac, args.val_frac)

    print(f"LQI data split: {len(df_train)} train, {len(df_val)} val, {len(df_test)} test")

    # Prepare trajectories
    train_traj = prepare_lqi_trajectory(df_train, subsample=args.subsample)
    val_traj = prepare_lqi_trajectory(df_val, subsample=args.subsample)

    # Create model and trainer
    params = GlassMelterParams()
    model = NeuralODE_GlassMelter(params=params)
    trainer = NeuralODETrainer(model, lr=args.lr)

    # Train with weighted loss (h=10, v=1, q_m=0.5)
    history = trainer.train_on_single_trajectory(
        train_traj,
        val_traj,
        n_epochs=args.epochs,
        rollout_steps=args.rollout,
        state_weights={0: 10.0, 1: 1.0, 2: 0.5},
    )

    # Save
    save_model(model, args.save_path, metadata={"variant": "v3", "epochs": args.epochs})
    print(f"✅ Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
