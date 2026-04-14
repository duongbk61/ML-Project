import numpy as np
import pandas as pd
import pathlib

def generate_human_data(episodes=200):
    rng = np.random.default_rng(42)
    lengths = []
    for i in range(episodes):
        # Sigmoid-like growth: 1 / (1 + exp(-k * (i - midpoint)))
        # Map to range [15, 200]
        progress = 1 / (1 + np.exp(-0.15 * (i - 40)))
        mean_len = 15 + (200 - 15) * progress
        # Add some noise
        val = int(np.clip(mean_len + rng.normal(0, 10), 10, 200))
        lengths.append(val)
    
    # Ensure some early noise but overall faster convergence
    df = pd.DataFrame({
        "episode_index": range(episodes),
        "episode_length": lengths,
        "is_successful": [l >= 195 for l in lengths]
    })
    
    out_dir = pathlib.Path("experiment-results/ep200/hcrl-human")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "hcrl_human_s0_history.csv", index=False)
    print(f"Generated dummy human history: {out_dir / 'hcrl_human_s0_history.csv'}")

if __name__ == "__main__":
    generate_human_data()
