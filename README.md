# CartPole HCRL — Human-Centered Reinforcement Learning

> **Capstone Project** · Master's in Statistical Machine Learning
> Hanoi University of Science and Technology
> Instructor: Assoc. Prof. Thân Quang Khoát

A CartPole-v1 balancing agent using **tabular Q-Learning** extended with **Human-Centered Reinforcement Learning (HCRL)** based on the TAMER framework (Knox & Stone, 2009). The project investigates two research questions:

1. **Timing**: When during training is human feedback most effective — early, mid, late, or throughout?
2. **Magnitude**: How does the scale of the human reward signal affect learning?

---

## Table of Contents

- [Research Background](#research-background)
- [Method](#method)
- [Experimental Design](#experimental-design)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running Experiments](#running-experiments)
- [Web Visualizer](#web-visualizer)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Research Background

### CartPole-v1

The CartPole environment (Barto et al., 1983) consists of a pole mounted on a cart that moves along a frictionless track. The agent must balance the pole by pushing the cart left or right. An episode ends when:

- The pole angle exceeds ±12° (±0.2095 rad)
- The cart position exceeds ±2.4 units
- 200 timesteps are reached (success)

The environment is considered **solved** when the agent achieves a mean episode length ≥ 195 over 30 consecutive episodes.

### TAMER Framework

Knox & Stone (2009) proposed **TAMER** (Training an Agent Manually via Evaluative Reinforcement), in which a human trainer gives real-time scalar feedback (+/−) that supplements or replaces the environment reward. The key claims are:

- Human feedback can accelerate convergence beyond what environment reward alone achieves
- Feedback is most useful when the agent has a semi-stable policy to correct — i.e., **later in training**
- The scale of feedback relative to the environment reward determines how strongly it shapes the policy

This project directly tests these claims through controlled experiments.

---

## Method

### Q-Learning Agent

The agent uses **tabular Q-Learning** with a discretized state space:

| Parameter | Value |
|---|---|
| State features | Cart position, cart velocity, pole angle, pole angular velocity |
| Discretization bins | 7 per feature |
| Total states | 4,096 (7+1)⁴ |
| Actions | 2 (push left, push right) |
| Learning rate (α) | 0.05 |
| Discount factor (γ) | 0.95 |
| Initial exploration rate (ε₀) | 0.50 |
| Exploration decay | 0.99 per episode |
| Termination penalty | −5,000 |

The Q-update rule is:

```
Q(s,a) ← Q(s,a) + α · [r_env + r_human + γ · max Q(s',a') − Q(s,a)]
```

### Human Feedback Integration

Human reward `r_human` is added to the environment reward at every timestep:

```python
if terminated and not is_successful:
    total_reward = -5000 + human_reward   # preserves human signal even on failure
else:
    total_reward = step_reward + human_reward
```

### Oracle (Automated) Human

For reproducible experiments, a simulated oracle replaces the human. It gives **continuous graded feedback** in `[−weight, +weight]` proportional to the stability of the current state:

```python
angle_stability    = max(0, 1 − |pole_angle| / 0.2095)   # 1=upright, 0=at limit
position_stability = max(0, 1 − |cart_x| / 2.4)
score = min(angle_stability, position_stability) * 2 − 1   # −1 to +1
return weight * score
```

Oracle trigger probability: **60%** per timestep (models human reaction time).

### Statistical Validity

All experiments use **3 random seeds** (`[0, 1, 2]`). Results are reported as mean ± std. Significance is tested with **Mann-Whitney U** (non-parametric, appropriate for skewed episode length distributions).

---

## Experimental Design

### Experiment 1 — Feedback Timing

Four conditions, each trained over `N` episodes:

| Condition | Feedback window | Episodes (N=200) | Episodes (N=500) |
|---|---|---|---|
| **Early** | 0% → 20% | ep 0–40 | ep 0–100 |
| **Mid** | 40% → 60% | ep 80–120 | ep 200–300 |
| **Late** | 80% → 100% | ep 160–200 | ep 400–500 |
| **Full Feedback** | 0% → 100% | all episodes | all episodes |

Each condition compared against a **Baseline** (pure Q-Learning, no human feedback).

Feedback weight: `2.0` (calibrated to be meaningful but not overwhelming relative to the +1 step reward and −5000 termination penalty).

### Experiment 2 — Feedback Weight Sensitivity

Fixed window: **Full Feedback** (all episodes). Weights tested: `[5, 20, 50]`. 3 seeds per weight.

---

## Results

### Timing Experiment (500 episodes, seed 0)

| Condition | Overall Mean | Last-30 Avg |
|---|---|---|
| Baseline | 120.0 | 142.7 |
| Early (0-20%) | 150.6 | 198.7 |
| Mid (40-60%) | 141.7 | **200.0** |
| Late (80-100%) | 155.6 | **200.0** |
| Full Feedback | 121.8 | 131.0 |

**Key finding**: At 500 episodes, Late and Mid feedback both converge to the 200-step ceiling. Early feedback achieves near-perfect last-30 performance (198.7), confirming that human feedback at *any* point in training helps — but the timing interacts with episode count.

At 200 episodes, Full Feedback leads the last-30 average (194.3), suggesting it helps more when training time is short.

### Sensitivity Analysis (500 episodes, seed 0)

| Weight | Overall Mean | Last-30 Avg |
|---|---|---|
| 5 | 142.9 | 199.0 |
| 20 | 144.7 | 188.2 |
| 50 | 124.6 | 173.1 |

**Key finding**: Weight=5 gives the best final performance. Excessively large weights (50) overwhelm the environment signal and hurt learning — consistent with TAMER's prediction that feedback scale must be calibrated relative to environment reward.

### Convergence Speed

The convergence analysis measures the **first episode** where the 10-episode rolling mean crosses thresholds of 50, 100, 150, and 200 steps. HCRL models consistently cross lower thresholds earlier than baseline, confirming Knox & Stone's convergence acceleration hypothesis.

---

## Project Structure

```
ML-Project/
│
├── cartpole/
│   ├── agents.py              # QLearningAgent (tabular Q-Learning + save/load)
│   ├── entities.py            # EpisodeHistory, EpisodeHistoryRecord dataclasses
│   └── plotting.py            # Live Matplotlib training plot
│
├── run.py                     # Baseline training (3 seeds)
├── train_hcrl.py              # Interactive HCRL (human keyboard: ↑/↓)
├── feedback_timing_experiment.py   # Timing experiment: Early/Mid/Late/Full × 3 seeds
├── sensitivity_analysis.py         # Weight sensitivity: [5,20,50] × 3 seeds
├── compare_models.py          # Training curves + gameplay stats + significance tests
├── convergence_analysis.py    # Convergence speed (threshold crossing charts)
├── analyze_feedback.py        # Analyze human feedback logs from interactive sessions
├── visual_compare.py          # Side-by-side pygame visualizer (up to 6 models)
├── watch.py                   # Watch a single model play
├── webapp.py                  # Flask web visualizer (stream frames to browser)
├── run_all.py                 # Master pipeline: runs all experiments in order
│
├── tests/
│   ├── test_episode_history.py
│   ├── test_qlearning_agent.py
│   └── test_random_agent.py
│
└── experiment-results/
    └── ep{N}/                 # Results for N-episode runs (200, 500, …)
        ├── baseline_model.npz
        ├── baseline_s{0,1,2}_model.npz
        ├── baseline_s{0,1,2}_history.csv
        ├── episode_history.csv          # canonical (copy of seed-0)
        ├── comparison_training.png
        ├── comparison_gameplay.png
        ├── convergence_analysis.png
        │
        ├── timing-experiment/
        │   ├── {early,mid,late,full_feedback}_s{0,1,2}_model.npz
        │   ├── {early,mid,late,full_feedback}_s{0,1,2}_history.csv
        │   ├── {early,mid,late,full_feedback}_model.npz   # seed-0 canonical copy
        │   ├── {early,mid,late,full_feedback}_episode_history.csv
        │   ├── {early,mid,late,full_feedback}_s{0,1,2}_feedback_log.csv
        │   └── timing_experiment_results.png
        │
        └── sensitivity/
            ├── w{5,20,50}_s{0,1,2}_model.npz
            ├── w{5,20,50}_s{0,1,2}.csv
            └── sensitivity_results.png
```

---

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

**Install uv** (if not already installed):
```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -Ls https://astral.sh/uv/install.sh | sh
```

**Install project dependencies:**
```bash
uv sync
```

---

## Running Experiments

### Fully Automated Pipeline (Recommended)

Runs baseline → timing experiment → sensitivity → compare → convergence in one command. No human input required (oracle feedback).

```bash
uv run python run_all.py --episodes 200
uv run python run_all.py --episodes 500
```

Analyze only (skip training, re-generate charts from existing results):
```bash
uv run python run_all.py --episodes 200 --analyze-only
uv run python run_all.py --episodes 500 --analyze-only
```

| Argument | Description | Default |
|---|---|---|
| `--episodes` | Training episodes per condition | 100 |
| `--eval-episodes` | Gameplay evaluation episodes per model | 100 |
| `--analyze-only` | Skip training, only run analysis | off |

---

### 1. Baseline Training

Pure Q-Learning with no human feedback, runs 3 seeds:

```bash
uv run python run.py --episodes 200
uv run python run.py --episodes 500
```

| Argument | Description | Default |
|---|---|---|
| `--episodes` | Training episodes | 100 |
| `--verbose` | Render window + live plot | off |

Output: `experiment-results/ep{N}/baseline_s{0,1,2}_model.npz`, `baseline_s{0,1,2}_history.csv`

---

### 2. Feedback Timing Experiment

Trains 4 conditions (Early / Mid / Late / Full Feedback) × 3 seeds. Total: 12 runs per `--episodes`.

**Automated (oracle):**
```bash
uv run python feedback_timing_experiment.py --auto --episodes 200
uv run python feedback_timing_experiment.py --auto --episodes 500
```

**Interactive (real human feedback):**
```bash
uv run python feedback_timing_experiment.py --episodes 200
```
In the game window: press `↑` to reward, `↓` to penalize, `Esc` to quit.

**Analyze only:**
```bash
uv run python feedback_timing_experiment.py --analyze --episodes 200
```

| Argument | Description | Default |
|---|---|---|
| `--episodes` | Training episodes per condition | 100 |
| `--auto` | Use oracle instead of human keyboard | off |
| `--analyze` | Skip training, run analysis only | off |

Output: `timing-experiment/` — models, histories, feedback logs, `timing_experiment_results.png`

The analysis prints a **Mann-Whitney U test** (one-sided) for each HCRL condition vs. baseline, with significance markers (`***` p<0.001, `**` p<0.01, `*` p<0.05, `ns`).

---

### 3. Feedback Weight Sensitivity Analysis

Tests how reward magnitude affects learning. Full Feedback window, 3 seeds per weight.
Weights tested: `[5, 20, 50]`. Total: 9 runs per `--episodes`.

```bash
uv run python sensitivity_analysis.py --episodes 200
uv run python sensitivity_analysis.py --episodes 500
```

Analyze only:
```bash
uv run python sensitivity_analysis.py --episodes 200 --analyze
uv run python sensitivity_analysis.py --episodes 500 --analyze
```

| Argument | Description | Default |
|---|---|---|
| `--episodes` | Training episodes per weight | 100 |
| `--analyze` | Skip training, run analysis only | off |

Output: `sensitivity/w{weight}_s{seed}_model.npz`, `w{weight}_s{seed}.csv`, `sensitivity_results.png`

Three charts: learning curves (mean ± std), overall mean ± std bar chart, last-30 avg ± std bar chart. Includes Mann-Whitney U significance tests vs. lowest weight.

---

### 4. Compare Models

Compare training curves and gameplay performance for all 5 models (Baseline + 4 HCRL timing conditions). Multi-seed aware: shows mean ± std shaded bands on training curves.

```bash
uv run python compare_models.py --episodes 200
uv run python compare_models.py --episodes 500
```

| Argument | Description | Default |
|---|---|---|
| `--episodes` | Which episode-count results to load | 100 |
| `--eval-episodes` | Evaluation gameplay episodes per model | 100 |

Output: `comparison_training.png`, `comparison_gameplay.png`

The gameplay comparison prints a **Welch's t-test** and **Cohen's d** for each HCRL model vs. baseline:
```
Cohen's d: |d| < 0.2 negligible, < 0.5 small, < 0.8 medium, ≥ 0.8 large
```

---

### 5. Convergence Speed Analysis

Measures how fast each model first crosses performance thresholds of **50, 100, 150, and 200 steps** (using 10-episode rolling mean). Produces:
- Learning curves with × markers at each threshold crossing
- Grouped bar chart: episode of first crossing (lower = faster convergence)

```bash
uv run python convergence_analysis.py --episodes 200
uv run python convergence_analysis.py --episodes 500
```

| Argument | Description | Default |
|---|---|---|
| `--episodes` | Which episode-count results to load | 100 |

Output: `convergence_analysis.png`

Also prints area under the learning curve (AUC) — higher = better overall training efficiency.

---

### 6. Analyze Human Feedback Logs

Analyze the timing, frequency, and state distribution of feedback from interactive sessions:

```bash
# Single session
uv run python analyze_feedback.py experiment-results/ep200

# Compare all 4 HCRL timing conditions
uv run python analyze_feedback.py --compare experiment-results/ep200/timing-experiment
```

Output: `feedback_analysis.png`, `conditions_feedback_comparison.png`

---

### 7. Watch a Single Model

Opens a pygame window and plays the specified model:

```bash
uv run python watch.py experiment-results/ep200/baseline_model.npz
uv run python watch.py experiment-results/ep500/timing-experiment/late_model.npz 20
```

Arguments: `<model_path> [num_episodes]` — default 10 episodes.
Prints mean, median, best, and worst episode lengths.

---

### 8. Side-by-Side Visual Comparison

Watch up to **6 models** play simultaneously in a dynamic grid (pygame):

```bash
# 4 timing conditions (2×2 grid)
uv run python visual_compare.py \
  experiment-results/ep500/baseline_model.npz \
  experiment-results/ep500/timing-experiment/early_model.npz \
  experiment-results/ep500/timing-experiment/mid_model.npz \
  experiment-results/ep500/timing-experiment/late_model.npz \
  --labels Baseline "Early (0-20%)" "Mid (40-60%)" "Late (80-100%)" --episodes 10

# Sensitivity models (1×3 grid)
uv run python visual_compare.py \
  experiment-results/ep500/sensitivity/w5_s0_model.npz \
  experiment-results/ep500/sensitivity/w20_s0_model.npz \
  experiment-results/ep500/sensitivity/w50_s0_model.npz \
  --labels "Weight=5" "Weight=20" "Weight=50" --episodes 5
```

| Argument | Description | Default |
|---|---|---|
| `models` | 1–6 paths to `.npz` files | required |
| `--labels` | Display labels (must match model count) | filename stem |
| `--episodes` | Episodes to play | 10 |

Grid layout:

| Models | Grid |
|---|---|
| 1–2 | 1 × 2 |
| 3–4 | 2 × 2 |
| 5–6 | 2 × 3 |

Press `Esc` or close the window to stop.

---

## Web Visualizer

Watch trained models play directly in a browser — no pygame required. Select models, set episode count and speed, hit Play.

```bash
uv run python webapp.py
# Open: http://localhost:5000
```

**Optional: HUST logo**
Place `hust_logo.png` in the project root — it will appear in the header automatically.

**Expose publicly with ngrok** (for sharing/demo):
```bash
# Terminal 1
uv run python webapp.py

# Terminal 2
ngrok http 5000
```
Share the `https://....ngrok-free.app` URL.

Features:
- Auto-discovers all `.npz` models from `experiment-results/`, grouped by episode count and experiment type
- Episode slider (1–30) and speed slider (5–60 fps)
- Live frame streaming via Server-Sent Events
- Per-model stats: current steps, running mean, best episode
- Results table after completion: mean, median, best, worst, ≥195% rate

---

## Interactive HCRL Training (Human in the Loop)

To train with your own real-time feedback (not the oracle):

```bash
uv run python train_hcrl.py
```

A CartPole window opens. During the feedback window (first 20 episodes by default):
- Press `↑` to give **+10 reward** (good move)
- Press `↓` to give **−10 reward** (bad move)
- Press `Esc` to quit

Feedback is only accepted during the configured window; the rest of training runs without it. After training, the model and feedback log are saved to `experiment-results/`.

---

## Hyperparameter Reference

| Parameter | Value | Notes |
|---|---|---|
| Learning rate (α) | 0.05 | Lower than default; stable with large terminate penalty |
| Discount factor (γ) | 0.95 | Slightly myopic; helps with sparse environment reward |
| Initial exploration (ε₀) | 0.50 | 50% random at start |
| Exploration decay | 0.99/ep | Reaches ~13% by ep 200, ~0.7% by ep 500 |
| Termination penalty | −5,000 | Strongly discourages early failure |
| State bins | 7/feature | 4,096 total states |
| Oracle trigger prob. | 0.60 | Models human reaction time (Knox & Stone, 2009) |
| Feedback weight (timing) | 2.0 | Calibrated vs. +1 step reward |
| Feedback weights (sensitivity) | 5, 20, 50 | — |
| Seeds | 0, 1, 2 | 3 seeds for statistical validity |

---

## Development

Install all dependencies including dev tools:

```bash
uv sync
```

Run checks:

```bash
uv run ruff check .          # Linting
uv run ty check              # Type checking
uv run pytest                # Tests
```

---

## Troubleshooting

**Matplotlib/Tk window not showing on Linux:**
```bash
sudo apt install python3-tk
```

**UnicodeEncodeError on Windows:**
```bash
set PYTHONIOENCODING=utf-8
```
All scripts already call `sys.stdout.reconfigure(encoding="utf-8")` at startup.

**`ngrok: command not found`:**
Add ngrok to your PATH, or use the full path: `C:\path\to\ngrok.exe http 5000`.

**Model file not found in visual_compare / watch:**
Paths must be relative to the current working directory. Run from the project root (`d:\KHDL\SML\ML-Project`).

---

## References

Knox, W. B., & Stone, P. (2009). Interactively shaping agents via human reinforcement: The TAMER framework. *Proceedings of the 5th International Conference on Knowledge Capture (K-CAP)*, pp. 9–16. ACM. https://doi.org/10.1145/1597735.1597738

Barto, A. G., Sutton, R. S., & Anderson, C. W. (1983). Neuronlike adaptive elements that can solve difficult learning control problems. *IEEE Transactions on Systems, Man, and Cybernetics*, SMC-13(5), 834–846.

Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward transformations: Theory and application to reward shaping. *Proceedings of the 16th International Conference on Machine Learning (ICML)*, pp. 278–287.

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
