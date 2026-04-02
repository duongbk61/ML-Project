import sys

import gymnasium as gym

from cartpole.agents import QLearningAgent


def replay(model_path: str, num_episodes: int = 10) -> None:
    """Load a trained Q-Learning model and let it play CartPole visually."""
    agent = QLearningAgent.load(model_path)
    env = gym.make("CartPole-v1", render_mode="human")

    print(f"Replaying {num_episodes} episodes with model from: {model_path}")
    print("Press Ctrl-C to stop.\n")

    try:
        for episode_index in range(num_episodes):
            observation, _ = env.reset()
            action = agent.begin_episode(observation)

            for timestep_index in range(200):
                observation, _, terminated, _, _ = env.step(action)

                if terminated or timestep_index >= 199:
                    print(
                        f"Episode {episode_index + 1}/{num_episodes} "
                        f"lasted {timestep_index + 1} timesteps."
                    )
                    break

                # Greedy action only (no learning, reward=0 is unused for updates)
                action = agent.act(observation, reward=0.0)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python replay.py <model_path> [num_episodes]")
        print("  e.g: uv run python replay.py experiment-results/baseline_model.npz")
        print("  e.g: uv run python replay.py experiment-results/hcrl_model.npz 20")
        sys.exit(1)

    model_path = sys.argv[1]
    num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    replay(model_path, num_episodes)


if __name__ == "__main__":
    main()
