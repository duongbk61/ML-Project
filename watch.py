"""
watch.py — Watch a single trained model play CartPole.

Usage:
    uv run python watch.py <model_path> [num_episodes]

Examples:
    uv run python watch.py experiment-results/baseline_model.npz
    uv run python watch.py experiment-results/timing-experiment/early_model.npz 5
"""

import sys
import pathlib

import gymnasium as gym
import numpy as np

from cartpole.agents import QLearningAgent


def watch(model_path: str, num_episodes: int = 10) -> None:
    path = pathlib.Path(model_path)
    if not path.exists():
        print(f"Model not found: {path}")
        sys.exit(1)

    agent = QLearningAgent.load(path)
    env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=None)

    print(f"Watching: {path.name}  ({num_episodes} episodes)")
    print("Press Ctrl-C to stop early.\n")

    episode_lengths: list[int] = []

    try:
        for ep in range(num_episodes):
            observation, _ = env.reset()
            action = agent.begin_episode(observation)
            steps = 0

            while True:
                observation, _, terminated, truncated, _ = env.step(action)
                steps += 1
                if terminated or truncated:
                    break
                action = agent.act(observation, reward=0.0)

            episode_lengths.append(steps)
            print(f"  Episode {ep + 1:>2}/{num_episodes}  steps: {steps}")

    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        env.close()

    if episode_lengths:
        print(f"\nResults over {len(episode_lengths)} episode(s):")
        print(f"  Mean:   {np.mean(episode_lengths):.1f}")
        print(f"  Median: {np.median(episode_lengths):.1f}")
        print(f"  Best:   {max(episode_lengths)}")
        print(f"  Worst:  {min(episode_lengths)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    model_path = sys.argv[1]
    num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    watch(model_path, num_episodes)
