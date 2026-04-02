import pathlib

import gymnasium as gym
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from cartpole.agents import QLearningAgent


def compare_training_curves(experiment_dir: pathlib.Path) -> bool:
    """Plot the training learning curves side by side. Returns True if data was found."""
    baseline_path = experiment_dir / "episode_history.csv"
    hcrl_path = experiment_dir / "hcrl_episode_history.csv"

    if not baseline_path.exists():
        print("Baseline data not found. Run 'uv run python run.py' first.")
        return False

    if not hcrl_path.exists():
        print("HCRL data not found. Run 'uv run python train_hcrl.py' first.")
        return False

    baseline_df = pd.read_csv(baseline_path, index_col="episode_index")
    hcrl_df = pd.read_csv(hcrl_path, index_col="episode_index")

    # Smooth with moving average for readability
    window_size = 20
    baseline_smooth = baseline_df["episode_length"].rolling(window=window_size, min_periods=1).mean()
    hcrl_smooth = hcrl_df["episode_length"].rolling(window=window_size, min_periods=1).mean()

    plt.figure(figsize=(12, 5))

    # Baseline curve
    plt.plot(baseline_smooth.index, baseline_smooth, label="Baseline (Q-Learning)", color="blue", linewidth=2)
    plt.scatter(baseline_df.index, baseline_df["episode_length"], color="blue", alpha=0.1, s=10)

    # HCRL curve
    plt.plot(hcrl_smooth.index, hcrl_smooth, label="HCRL (Q-Learning + Human)", color="red", linewidth=2)
    plt.scatter(hcrl_df.index, hcrl_df["episode_length"], color="red", alpha=0.1, s=10)

    plt.title("Training Curves: Baseline vs HCRL (Smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Episode Length (steps)")
    plt.axhline(y=195, color="g", linestyle="--", label="Goal: 195")

    plt.legend()
    plt.grid(True)

    output_path = experiment_dir / "comparison_training.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Training chart saved to: {output_path}")
    return True


def compare_gameplay(experiment_dir: pathlib.Path, num_episodes: int = 100) -> bool:
    """Load saved models and compare their actual gameplay performance."""
    baseline_model_path = experiment_dir / "baseline_model.npz"
    hcrl_model_path = experiment_dir / "hcrl_model.npz"

    if not baseline_model_path.exists():
        print(f"Baseline model not found ({baseline_model_path}). Run 'uv run python run.py' first.")
        return False

    if not hcrl_model_path.exists():
        print(f"HCRL model not found ({hcrl_model_path}). Run 'uv run python train_hcrl.py' first.")
        return False

    # Evaluate both models
    print(f"\nEvaluating Baseline model ({num_episodes} episodes)...")
    baseline_agent = QLearningAgent.load(baseline_model_path)
    baseline_lengths = evaluate_model(baseline_agent, num_episodes)

    print(f"Evaluating HCRL model ({num_episodes} episodes)...")
    hcrl_agent = QLearningAgent.load(hcrl_model_path)
    hcrl_lengths = evaluate_model(hcrl_agent, num_episodes)

    # Print statistics
    print("\n" + "=" * 60)
    print(f"{'Metric':<25} {'Baseline':>15} {'HCRL':>15}")
    print("=" * 60)
    print(f"{'Mean':<25} {np.mean(baseline_lengths):>15.1f} {np.mean(hcrl_lengths):>15.1f}")
    print(f"{'Median':<25} {np.median(baseline_lengths):>15.1f} {np.median(hcrl_lengths):>15.1f}")
    print(f"{'Std Dev':<25} {np.std(baseline_lengths):>15.1f} {np.std(hcrl_lengths):>15.1f}")
    print(f"{'Min':<25} {np.min(baseline_lengths):>15d} {np.min(hcrl_lengths):>15d}")
    print(f"{'Max':<25} {np.max(baseline_lengths):>15d} {np.max(hcrl_lengths):>15d}")
    print(f"{'Episodes ≥200 (Perfect)':<25} {sum(1 for l in baseline_lengths if l >= 200):>15d} {sum(1 for l in hcrl_lengths if l >= 200):>15d}")
    print(f"{'Rate ≥195 (Solved %)':<25} {sum(1 for l in baseline_lengths if l >= 195) / num_episodes * 100:>14.1f}% {sum(1 for l in hcrl_lengths if l >= 195) / num_episodes * 100:>14.1f}%")
    print("=" * 60)

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot
    axes[0].boxplot(
        [baseline_lengths, hcrl_lengths],
        labels=["Baseline", "HCRL"],
        patch_artist=True,
        boxprops=[dict(facecolor="lightblue"), dict(facecolor="lightsalmon")],
    )
    axes[0].axhline(y=195, color="g", linestyle="--", alpha=0.7, label="Goal: 195")
    axes[0].set_title("Gameplay Performance Comparison (Box Plot)")
    axes[0].set_ylabel("Episode Length")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Histogram
    axes[1].hist(baseline_lengths, bins=20, alpha=0.6, label="Baseline", color="blue")
    axes[1].hist(hcrl_lengths, bins=20, alpha=0.6, label="HCRL", color="red")
    axes[1].axvline(x=195, color="g", linestyle="--", alpha=0.7, label="Goal: 195")
    axes[1].set_title("Episode Length Distribution (Histogram)")
    axes[1].set_xlabel("Episode Length")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = experiment_dir / "comparison_gameplay.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nGameplay comparison chart saved to: {output_path}")
    return True


def compare_models(experiment_dir: str = "experiment-results", num_episodes: int = 100) -> None:
    dir_path = pathlib.Path(experiment_dir)

    print("=" * 60)
    print(" COMPARING BASELINE vs HCRL")
    print("=" * 60)

    # 1. Training curves
    print("\n[1/2] Comparing training curves...")
    compare_training_curves(dir_path)

    plt.show()


if __name__ == "__main__":
    compare_models()
