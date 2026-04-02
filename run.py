import pathlib
import sys
from dataclasses import asdict

import gymnasium as gym
import numpy as np
import pandas as pd

from cartpole.agents import Agent, QLearningAgent
from cartpole.entities import Action, EpisodeHistory, EpisodeHistoryRecord, Observation, Reward


def run_agent(agent: Agent, env: gym.Env, verbose: bool = False, max_episodes: int = 100) -> EpisodeHistory:
    """
    Run an intelligent cartpole agent in a cartpole environment,
    capturing the episode history.
    """

    max_episodes_to_run = max_episodes
    max_timesteps_per_episode = 200
    terminate_penalty = 5000

    # The environment is solved if we can survive for avg. 195 timesteps across 100 episodes.
    goal_avg_episode_length = 195
    goal_consecutive_episodes = 30

    episode_history = EpisodeHistory(
        max_timesteps_per_episode=200,
        goal_avg_episode_length=goal_avg_episode_length,
        goal_consecutive_episodes=goal_consecutive_episodes,
    )
    episode_history_plotter = None

    if verbose:
        from cartpole.plotting import EpisodeHistoryMatplotlibPlotter

        episode_history_plotter = EpisodeHistoryMatplotlibPlotter(
            history=episode_history,
            visible_episode_count=200,  # How many most recent episodes to fit on a single plot.
        )
        episode_history_plotter.create_plot()

    # Main simulation/learning loop.
    print("Running the environment. To stop, press Ctrl-C.")
    try:
        for episode_index in range(max_episodes_to_run):
            observation, _ = env.reset()
            action = agent.begin_episode(observation)

            for timestep_index in range(max_timesteps_per_episode):
                # Perform the action and observe the new state.
                observation, step_reward, terminated, _, _ = env.step(action)
                reward: Reward = float(step_reward)

                # Log the current state.
                if verbose:
                    log_timestep(timestep_index, action, reward, observation)

                # If the episode has ended prematurely, penalize the agent.
                is_successful = timestep_index >= max_timesteps_per_episode - 1
                if terminated and not is_successful:
                    reward = float(-terminate_penalty)

                # Get the next action from the learner, given our new state.
                action = agent.act(observation, reward)

                # Record this episode to the history and check if the goal has been reached.
                if terminated or is_successful:
                    print(
                        f"Episode {episode_index} "
                        f"finished after {timestep_index + 1} timesteps."
                    )

                    episode_history.record_episode(
                        EpisodeHistoryRecord(
                            episode_index=episode_index,
                            episode_length=timestep_index + 1,
                            is_successful=is_successful,
                        )
                    )
                    if verbose and episode_history_plotter:
                        episode_history_plotter.update_plot()

                    if episode_history.is_goal_reached():
                        print(f"SUCCESS: Goal reached after {episode_index + 1} episodes!")
                        return episode_history

                    break

        print(f"FAILURE: Goal not reached after {max_episodes_to_run} episodes.")

    except KeyboardInterrupt:
        print("WARNING: Terminated by user request.")

    return episode_history


def log_timestep(index: int, action: Action, reward: Reward, observation: Observation) -> None:
    """Log the information about the current timestep results."""

    format_string = "   ".join(
        [
            "Timestep: {0:3d}",
            "Action: {1:2d}",
            "Reward: {2:5.1f}",
            "Cart Position: {3:6.3f}",
            "Cart Velocity: {4:6.3f}",
            "Angle: {5:6.3f}",
            "Tip Velocity: {6:6.3f}",
        ]
    )
    print(format_string.format(index, action, reward, *observation))


def save_history(history: EpisodeHistory, experiment_dir: str) -> pathlib.Path:
    """
    Save the episode history to a CSV file.

    Parameters
    ----------
    history : EpisodeHistory
        History to save.
    experiment_dir : str
        Name of the directory to save the history to. Will be created if nonexistent.

    Returns
    -------
    pathlib.Path
        The path of the generated file.
    """

    experiment_dir_path = pathlib.Path(experiment_dir)
    experiment_dir_path.mkdir(parents=True, exist_ok=True)

    file_path = experiment_dir_path / "episode_history.csv"
    record_dicts = (asdict(record) for record in history.all_records())
    dataframe = pd.DataFrame.from_records(record_dicts, index="episode_index")
    dataframe.to_csv(file_path, header=True)
    print(f"Episode history saved to {file_path}")
    return file_path


SEEDS = [0, 1, 2]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    out_dir = f"experiment-results/ep{args.episodes}"

    all_lengths = []
    for seed in SEEDS:
        print(f"\n--- Baseline seed={seed} ---")
        random_state = np.random.default_rng(seed=seed)
        env = gym.make("CartPole-v1", render_mode="human" if args.verbose else None)
        agent = QLearningAgent(
            learning_rate=0.05,
            discount_factor=0.95,
            exploration_rate=0.5,
            exploration_decay_rate=0.99,
            random_state=random_state,
        )
        episode_history = run_agent(agent=agent, env=env, verbose=args.verbose,
                                    max_episodes=args.episodes)
        env.close()

        # Save per-seed files
        save_history(episode_history, experiment_dir=out_dir)
        # Rename to seed-specific file
        src = pathlib.Path(out_dir) / "episode_history.csv"
        dst = pathlib.Path(out_dir) / f"baseline_s{seed}_history.csv"
        if dst.exists():
            dst.unlink()
        src.rename(dst)
        agent.save(f"{out_dir}/baseline_s{seed}_model.npz")

        lengths = [r.episode_length for r in episode_history.all_records()]
        all_lengths.append(lengths)
        mean_last30 = pd.Series(lengths).tail(30).mean()
        print(f"  seed={seed}: mean={pd.Series(lengths).mean():.1f}, last-30={mean_last30:.1f}")

    # Also save seed-0 copy as canonical baseline_model.npz / episode_history.csv
    # so downstream scripts that expect the old filenames still work
    import shutil
    shutil.copy(f"{out_dir}/baseline_s0_history.csv", f"{out_dir}/episode_history.csv")
    shutil.copy(f"{out_dir}/baseline_s0_model.npz",   f"{out_dir}/baseline_model.npz")

    # Cross-seed summary
    means = [pd.Series(l).mean() for l in all_lengths]
    last30s = [pd.Series(l).tail(30).mean() for l in all_lengths]
    print(f"\nBaseline ({len(SEEDS)} seeds):")
    print(f"  Overall mean : {pd.Series(means).mean():.1f} ± {pd.Series(means).std():.1f}")
    print(f"  Last-30 avg  : {pd.Series(last30s).mean():.1f} ± {pd.Series(last30s).std():.1f}")


if __name__ == "__main__":
    main()
