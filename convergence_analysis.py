"""
Convergence Speed Analysis
===========================
Measures *how fast* each model reaches performance thresholds during training.

Motivation (Knox & Stone, 2009):
    A key claim of HCRL is not just *final* performance but *learning speed*.
    Human feedback should accelerate convergence to a competent policy.
    This script quantifies that by finding the first episode each model
    crosses performance thresholds of 50, 100, 150, and 200 steps (rolling mean).

Reference:
    Knox, W. B., & Stone, P. (2009). Interactively shaping agents via human
    reinforcement: The TAMER framework. Proceedings of the 5th International
    Conference on Knowledge Capture (K-CAP), pp. 9-16. ACM.

Usage:
    uv run python convergence_analysis.py
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")


def get_models(episodes: int) -> list[dict]:
    base = pathlib.Path(f"experiment-results/ep{episodes}")
    timing = base / "timing-experiment"
    return [
        {
            "key": "baseline",
            "label": "Baseline (Shaped)",
            "color": "blue",
            "history_path": base / "episode_history.csv",
            "is_hcrl": False,
            "window": None,
        },
        {
            "key": "early",
            "label": "HCRL Early (0-20)",
            "color": "green",
            "history_path": timing / "early_episode_history.csv",
            "is_hcrl": True,
            "window": (0, 20),
        },
        {
            "key": "mid",
            "label": "HCRL Mid (40-60)",
            "color": "orange",
            "history_path": timing / "mid_episode_history.csv",
            "is_hcrl": True,
            "window": (40, 60),
        },
        {
            "key": "late",
            "label": "HCRL Late (80-100)",
            "color": "purple",
            "history_path": timing / "late_episode_history.csv",
            "is_hcrl": True,
            "window": (80, 100),
        },
        {
            "key": "full_feedback",
            "label": "HCRL Full Feedback",
            "color": "red",
            "history_path": timing / "full_feedback_episode_history.csv",
            "is_hcrl": True,
            "window": None,
        },
    ]

THRESHOLDS = [50, 100, 150, 200]
ROLLING_WINDOW = 10


def first_crossing(series: pd.Series, threshold: float) -> int | None:
    """Return the first episode index where the rolling mean crosses the threshold."""
    rolling = series.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    crossed = rolling[rolling >= threshold]
    return int(crossed.index[0]) if not crossed.empty else None


def analyze_convergence(episodes: int = 100):
    MODELS = get_models(episodes)
    out_dir = pathlib.Path(f"experiment-results/ep{episodes}")

    # Load histories
    loaded = []
    for m in MODELS:
        if not m["history_path"].exists():
            print(f"  Missing: {m['history_path']}")
            continue
        df = pd.read_csv(m["history_path"], index_col="episode_index")
        loaded.append((m, df))

    if len(loaded) < 2:
        print("Need at least 2 models. Train first.")
        return

    # Print convergence table
    print("\n" + "=" * 72)
    print(f"  {'CONVERGENCE SPEED ANALYSIS (Knox & Stone, 2009)':^70}")
    print(f"  {'First episode where {}-ep rolling mean >= threshold'.format(ROLLING_WINDOW):^70}")
    print("=" * 72)
    col = 16
    header = f"  {'Threshold':<20}" + "".join(f"{m['label']:>{col}}" for m, _ in loaded)
    print(header)
    print("  " + "-" * 70)

    crossing_data: dict[str, dict[int, int | None]] = {}
    for m, df in loaded:
        crossing_data[m["key"]] = {}
        for threshold in THRESHOLDS:
            ep = first_crossing(df["episode_length"], threshold)
            crossing_data[m["key"]][threshold] = ep

    for threshold in THRESHOLDS:
        row = f"  {'>= ' + str(threshold) + ' steps':<20}"
        for m, _ in loaded:
            val = crossing_data[m["key"]][threshold]
            cell = f"ep {val}" if val is not None else "never"
            row += f"{cell:>{col}}"
        print(row)

    # Area under learning curve (proxy for total learning efficiency)
    print("  " + "-" * 70)
    row = f"  {'Area under curve':<20}"
    for m, df in loaded:
        auc = df["episode_length"].rolling(window=ROLLING_WINDOW, min_periods=1).mean().sum()
        row += f"{auc:>{col}.0f}"
    print(row)
    print("=" * 72)
    print("  Note: Lower episode = faster convergence. 'never' = threshold not reached.")
    print("  Area under curve: higher = better overall training performance.")

    # --- Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        "Convergence Speed Analysis — How Fast Does Each Model Learn?\n"
        "Knox & Stone (2009): HCRL should accelerate convergence",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # 1. Learning curves with threshold lines and crossing markers
    ax = axes[0]
    for m, df in loaded:
        rolling = df["episode_length"].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        ls = "--" if not m["is_hcrl"] else "-"
        ax.plot(rolling.index, rolling, color=m["color"], linewidth=2,
                linestyle=ls, label=m["label"])

        # Shade feedback window
        if m["window"]:
            ax.axvspan(m["window"][0], m["window"][1], color=m["color"], alpha=0.07)

        # Mark threshold crossings
        for threshold in THRESHOLDS:
            ep = crossing_data[m["key"]][threshold]
            if ep is not None:
                ax.plot(ep, threshold, marker="x", color=m["color"], markersize=8, markeredgewidth=2)

    for threshold in THRESHOLDS:
        color = "red" if threshold == 200 else "gray"
        ls    = "--"  if threshold == 200 else ":"
        alpha = 0.55  if threshold == 200 else 0.6
        ax.axhline(y=threshold, color=color, linestyle=ls, alpha=alpha, linewidth=1)
        ax.text(episodes + 2, threshold + 1, f"{threshold}", color=color, fontsize=8)
    ax.set_title(f"Learning Curves with Threshold Crossings\n(× = first crossing, {ROLLING_WINDOW}-ep rolling mean)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, episodes + 10)

    # 2. Grouped bar chart: episode of first crossing per threshold
    ax = axes[1]
    n_models = len(loaded)
    n_thresholds = len(THRESHOLDS)
    bar_width = 0.8 / n_models
    x = np.arange(n_thresholds)

    # Find max crossing episode for y-limit
    max_ep = 0
    for i, (m, _) in enumerate(loaded):
        crossing_eps = []
        for threshold in THRESHOLDS:
            ep = crossing_data[m["key"]][threshold]
            val = ep if ep is not None else episodes + 10
            crossing_eps.append(val)
            if ep is not None:
                max_ep = max(max_ep, ep)

        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, crossing_eps, bar_width,
                      label=m["label"], color=m["color"], alpha=0.75)
        for bar, ep_val, threshold in zip(bars, crossing_eps, THRESHOLDS):
            actual_ep = crossing_data[m["key"]][threshold]
            label = str(actual_ep) if actual_ep is not None else "N/A"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    label, ha="center", va="bottom", fontsize=7, fontweight="bold",
                    color=m["color"], rotation=45)

    ax.set_title("Episode of First Threshold Crossing\n(lower = faster convergence)")
    ax.set_xlabel("Performance Threshold")
    ax.set_ylabel("Episode Number")
    ax.set_xticks(x)
    ax.set_xticklabels([f"≥ {t} steps" for t in THRESHOLDS])
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max_ep * 1.25 if max_ep > 0 else episodes + 20)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    output_path = out_dir / "convergence_analysis.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    analyze_convergence(args.episodes)
