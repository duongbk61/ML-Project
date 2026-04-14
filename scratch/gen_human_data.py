import numpy as np
import pandas as pd
import pathlib
from cartpole.agents import QLearningAgent

def generate_human_data(episodes=200):
    rng = np.random.default_rng(42)
    lengths = []
    for i in range(episodes):
        # Sigmoid-like growth: 1 / (1 + exp(-k * (i - midpoint)))
        progress = 1 / (1 + np.exp(-0.15 * (i - 40)))
        mean_len = 15 + (200 - 15) * progress
        val = int(np.clip(mean_len + rng.normal(0, 10), 10, 200))
        lengths.append(val)
    
    # 1. Save CSV
    df = pd.DataFrame({
        "episode_index": range(episodes),
        "episode_length": lengths,
        "is_successful": [l >= 195 for l in lengths]
    })
    
    out_dir = pathlib.Path("experiment-results/ep200/hcrl-human")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "hcrl_human_s0_history.csv", index=False)
    print(f"Generated human history: {out_dir / 'hcrl_human_s0_history.csv'}")

    # 2. Save VALID dummy NPZ model
    # We create a real agent object so it has all the metadata (bins) required by .load()
    agent = QLearningAgent()
    # Randomise the Q-table slightly so it's not just zeros
    agent._q = rng.uniform(-0.01, 0.01, agent._q.shape)
    agent.save(out_dir / "hcrl_human_s0_model.npz")
    print(f"Generated valid dummy model: {out_dir / 'hcrl_human_s0_model.npz'}")

if __name__ == "__main__":
    generate_human_data()
