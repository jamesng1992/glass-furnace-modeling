"""
Train Neural ODE v1: simulation data within LQI input ranges.

Usage:
    python scripts/train_v1.py --data-path data/LQI_data.xlsx --epochs 200
"""

import argparse
import torch

from glass_melter_level.config import GlassMelterParams
from glass_melter_level.data.loader import load_lqi_data, get_data_ranges
from glass_melter_level.data.scenario_generation import generate_scenarios
from glass_melter_level.training.trajectory import prepare_trajectory
from glass_melter_level.training.trainer import NeuralODETrainer
from glass_melter_level.models.neural_ode import NeuralODE_GlassMelter, save_model


def main():
    parser = argparse.ArgumentParser(description="Train Neural ODE v1 (LQI-range simulation)")
    parser.add_argument("--data-path", type=str, required=True, help="Path to LQI Excel file")
    parser.add_argument("--n-scenarios", type=int, default=300, help="Number of training scenarios")
    parser.add_argument("--t-sim", type=float, default=48.0, help="Simulation horizon (hours)")
    parser.add_argument("--dt", type=float, default=0.01, help="Integration time step (hours)")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--rollout", type=int, default=20, help="Rollout steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save-path", type=str, default="trained_models/neural_ode_v1.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load LQI data to determine input ranges
    df_true, df_est = load_lqi_data(args.data_path)
    ranges = get_data_ranges(df_est)
    print(f"LQI input ranges: {ranges}")

    # Generate simulation scenarios
    params = GlassMelterParams()
    scenarios = generate_scenarios(
        n_scenarios=args.n_scenarios,
        t_sim=args.t_sim,
        dt=args.dt,
        lqi_ic=df_est,
        u1_range=(ranges["u1_min"], ranges["u1_max"]),
        u2_range=(ranges["u2_min"], ranges["u2_max"]),
        seed=args.seed,
    )

    # Prepare trajectories
    trajectories = [prepare_trajectory(sim, subsample=10) for sim in scenarios]
    n_train = int(0.8 * len(trajectories))
    train_traj = trajectories[:n_train]
    val_traj = trajectories[n_train:]

    print(f"Trajectories: {len(train_traj)} train, {len(val_traj)} val")

    # Create model and trainer
    model = NeuralODE_GlassMelter(params=params)
    trainer = NeuralODETrainer(model, lr=args.lr)

    # Train
    history = trainer.train(
        train_traj, val_traj,
        n_epochs=args.epochs,
        rollout_steps=args.rollout,
    )

    # Save
    save_model(model, args.save_path, metadata={"variant": "v1", "epochs": args.epochs})
    print(f"✅ Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
