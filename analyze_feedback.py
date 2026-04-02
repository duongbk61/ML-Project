import pathlib
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def load_data(experiment_dir: pathlib.Path):
    """Load feedback log and episode history."""
    feedback_path = experiment_dir / "hcrl_feedback_log.csv"
    history_path = experiment_dir / "hcrl_episode_history.csv"

    if not feedback_path.exists():
        print(f"Feedback log not found: {feedback_path}")
        print("Run 'uv run python train_hcrl.py' first.")
        return None, None

    if not history_path.exists():
        print(f"Episode history not found: {history_path}")
        return None, None

    feedback_df = pd.read_csv(feedback_path)
    history_df = pd.read_csv(history_path, index_col="episode_index")

    return feedback_df, history_df


def print_summary(feedback_df: pd.DataFrame, history_df: pd.DataFrame):
    """Print a text summary of feedback statistics."""
    total = len(feedback_df)
    positive = len(feedback_df[feedback_df["feedback"] == "positive"])
    negative = len(feedback_df[feedback_df["feedback"] == "negative"])
    episodes_with_feedback = feedback_df["episode"].nunique()
    total_episodes = len(history_df)

    print("\n" + "=" * 55)
    print(f"  {'HUMAN FEEDBACK ANALYSIS':^53}")
    print("=" * 55)
    print(f"  Total feedback:            {total:>8}")
    print(f"  Positive (↑):              {positive:>8}  ({positive/total*100:.1f}%)")
    print(f"  Negative (↓):              {negative:>8}  ({negative/total*100:.1f}%)")
    print(f"  Positive/Negative ratio:   {positive/max(negative,1):>8.2f}")
    print(f"  Episodes with feedback:    {episodes_with_feedback:>8} / {total_episodes}")
    print(f"  Avg feedback/episode:      {total/episodes_with_feedback:>8.1f}")

    # Feedback over time
    if episodes_with_feedback > 0:
        first_ep = feedback_df["episode"].min()
        last_ep = feedback_df["episode"].max()
        print(f"  Feedback episode range:    {first_ep:>8} → {last_ep}")

    # Training duration
    if "timestamp" in feedback_df.columns and len(feedback_df) > 0:
        duration = feedback_df["timestamp"].max()
        print(f"  Total feedback duration:   {duration:>7.1f}s")

    print("=" * 55)


def plot_feedback_analysis(feedback_df: pd.DataFrame, history_df: pd.DataFrame, output_dir: pathlib.Path):
    """Generate comprehensive feedback analysis plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Human Feedback Analysis", fontsize=14, fontweight="bold")

    # --- 1. Feedback count per episode ---
    ax = axes[0, 0]
    fb_per_ep = feedback_df.groupby("episode").size()
    pos_per_ep = feedback_df[feedback_df["feedback"] == "positive"].groupby("episode").size()
    neg_per_ep = feedback_df[feedback_df["feedback"] == "negative"].groupby("episode").size()

    all_episodes = range(len(history_df))
    pos_counts = [pos_per_ep.get(ep, 0) for ep in all_episodes]
    neg_counts = [neg_per_ep.get(ep, 0) for ep in all_episodes]

    ax.bar(all_episodes, pos_counts, color="green", alpha=0.7, label="Positive (↑)")
    ax.bar(all_episodes, neg_counts, bottom=pos_counts, color="red", alpha=0.7, label="Negative (↓)")
    ax.set_title("Feedback Count per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Feedback count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 2. Cumulative feedback over time ---
    ax = axes[0, 1]
    if len(feedback_df) > 0:
        sorted_fb = feedback_df.sort_values("timestamp")
        cum_pos = (sorted_fb["feedback"] == "positive").cumsum()
        cum_neg = (sorted_fb["feedback"] == "negative").cumsum()
        ax.plot(sorted_fb["timestamp"], cum_pos, color="green", label="Cumulative Positive", linewidth=2)
        ax.plot(sorted_fb["timestamp"], cum_neg, color="red", label="Cumulative Negative", linewidth=2)
        ax.set_title("Cumulative Feedback over Time")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Cumulative feedback")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # --- 3. Episode length vs feedback count ---
    ax = axes[0, 2]
    ep_lengths = history_df["episode_length"].values
    fb_counts = [fb_per_ep.get(ep, 0) for ep in range(len(history_df))]
    has_fb = [c > 0 for c in fb_counts]
    no_fb = [not c for c in has_fb]

    if any(has_fb):
        ax.scatter(
            [fb_counts[i] for i in range(len(fb_counts)) if has_fb[i]],
            [ep_lengths[i] for i in range(len(ep_lengths)) if has_fb[i]],
            color="blue", alpha=0.6, s=30, label="Episodes with feedback",
        )
    ax.set_title("Episode Length vs Feedback Count")
    ax.set_xlabel("Feedback count per episode")
    ax.set_ylabel("Episode length (steps)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 4. Feedback density: moving average of feedback per episode ---
    ax = axes[1, 0]
    window = 10
    fb_series = pd.Series(fb_counts)
    fb_rolling = fb_series.rolling(window=window, min_periods=1).mean()
    ep_rolling = history_df["episode_length"].rolling(window=window, min_periods=1).mean()

    ax2 = ax.twinx()
    ax.bar(range(len(fb_series)), fb_series, color="orange", alpha=0.3, label="Feedback/ep")
    ax.plot(fb_rolling.index, fb_rolling, color="orange", linewidth=2, label=f"Feedback MA({window})")
    ax2.plot(ep_rolling.index, ep_rolling, color="blue", linewidth=2, label=f"Ep Length MA({window})")

    ax.set_title("Feedback Density & Agent Performance")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Feedback count", color="orange")
    ax2.set_ylabel("Episode length", color="blue")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 5. Pole angle distribution at feedback time ---
    ax = axes[1, 1]
    if "pole_angle" in feedback_df.columns:
        pos_angles = feedback_df[feedback_df["feedback"] == "positive"]["pole_angle"]
        neg_angles = feedback_df[feedback_df["feedback"] == "negative"]["pole_angle"]
        bins = np.linspace(-0.25, 0.25, 25)
        if len(pos_angles) > 0:
            ax.hist(pos_angles, bins=bins, alpha=0.6, color="green", label="Positive")
        if len(neg_angles) > 0:
            ax.hist(neg_angles, bins=bins, alpha=0.6, color="red", label="Negative")
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
        ax.set_title("Pole Angle at Feedback Time")
        ax.set_xlabel("Pole Angle (rad)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # --- 6. Cart position distribution at feedback time ---
    ax = axes[1, 2]
    if "cart_position" in feedback_df.columns:
        pos_cart = feedback_df[feedback_df["feedback"] == "positive"]["cart_position"]
        neg_cart = feedback_df[feedback_df["feedback"] == "negative"]["cart_position"]
        bins = np.linspace(-2.4, 2.4, 25)
        if len(pos_cart) > 0:
            ax.hist(pos_cart, bins=bins, alpha=0.6, color="green", label="Positive")
        if len(neg_cart) > 0:
            ax.hist(neg_cart, bins=bins, alpha=0.6, color="red", label="Negative")
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
        ax.set_title("Cart Position at Feedback Time")
        ax.set_xlabel("Cart Position")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "feedback_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFeedback analysis chart saved to: {output_path}")


TIMING_CONDITIONS = [
    {"name": "early",         "label": "Early (0-20)",   "color": "green"},
    {"name": "mid",           "label": "Mid (40-60)",    "color": "orange"},
    {"name": "late",          "label": "Late (80-100)",  "color": "purple"},
    {"name": "full_feedback", "label": "Full Feedback",  "color": "red"},
]


def compare_conditions(timing_dir: str = "experiment-results/timing-experiment"):
    """Compare feedback statistics across all timing conditions side by side."""
    dir_path = pathlib.Path(timing_dir)

    all_data: list[tuple[dict, pd.DataFrame, pd.DataFrame]] = []
    for cond in TIMING_CONDITIONS:
        fb_path = dir_path / f"{cond['name']}_feedback_log.csv"
        hist_path = dir_path / f"{cond['name']}_episode_history.csv"
        if not fb_path.exists() or not hist_path.exists():
            print(f"  Data not found: {cond['label']}")
            continue
        fb_df = pd.read_csv(fb_path)
        hist_df = pd.read_csv(hist_path, index_col="episode_index")
        all_data.append((cond, fb_df, hist_df))

    if not all_data:
        print("No timing experiment data found. Train models first.")
        return

    # Print comparison table
    print("\n" + "=" * 65)
    print(f"  {'COMPARISON: TIMING CONDITIONS':^63}")
    print("=" * 65)
    col = 18
    header = f"  {'Metric':<25}" + "".join(f"{d[0]['label']:>{col}}" for d in all_data)
    print(header)
    print("  " + "-" * 63)
    for stat_name, stat_fn in [
        ("Total feedback",     lambda fb, _: str(len(fb))),
        ("Positive",           lambda fb, _: f"{(fb['feedback']=='positive').sum()} ({(fb['feedback']=='positive').mean()*100:.0f}%)"),
        ("Negative",           lambda fb, _: f"{(fb['feedback']=='negative').sum()} ({(fb['feedback']=='negative').mean()*100:.0f}%)"),
        ("Episodes w/ feedback", lambda fb, hist: f"{fb['episode'].nunique()}/{len(hist)}"),
        ("Avg FB/episode",     lambda fb, _: f"{len(fb)/max(fb['episode'].nunique(),1):.1f}"),
    ]:
        row = f"  {stat_name:<25}" + "".join(f"{stat_fn(d[1], d[2]):>{col}}" for d in all_data)
        print(row)
    print("=" * 65)

    # Plot: feedback density across conditions overlaid
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Feedback Comparison: Timing Conditions", fontsize=13, fontweight="bold")

    ax = axes[0]
    for cond, fb_df, hist_df in all_data:
        fb_per_ep = fb_df.groupby("episode").size()
        counts = pd.Series([fb_per_ep.get(ep, 0) for ep in range(len(hist_df))])
        rolling = counts.rolling(window=5, min_periods=1).mean()
        ax.plot(rolling.index, rolling, color=cond["color"], linewidth=2, label=cond["label"])
    ax.set_title("Feedback Density per Episode (MA-5)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Feedback / episode")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for cond, fb_df, hist_df in all_data:
        ep_lengths = hist_df["episode_length"].rolling(window=10, min_periods=1).mean()
        ax.plot(ep_lengths.index, ep_lengths, color=cond["color"], linewidth=2, label=cond["label"])
    ax.axhline(y=195, color="gray", linestyle="--", alpha=0.5, label="Goal: 195")
    ax.set_title("Learning Curve by Condition (MA-10)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = dir_path / "conditions_feedback_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nConditions comparison chart saved to: {output_path}")
    plt.show()


def analyze_feedback(experiment_dir: str = "experiment-results"):
    dir_path = pathlib.Path(experiment_dir)
    feedback_df, history_df = load_data(dir_path)
    if feedback_df is None or history_df is None:
        return

    print_summary(feedback_df, history_df)
    plot_feedback_analysis(feedback_df, history_df, dir_path)
    plt.show()


if __name__ == "__main__":
    import sys
    if "--compare" in sys.argv:
        timing_dir = sys.argv[sys.argv.index("--compare") + 1] if sys.argv.index("--compare") + 1 < len(sys.argv) else "experiment-results/timing-experiment"
        compare_conditions(timing_dir)
    else:
        exp_dir = sys.argv[1] if len(sys.argv) > 1 else "experiment-results"
        analyze_feedback(exp_dir)
