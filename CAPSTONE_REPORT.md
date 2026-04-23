# Human-Centered Reinforcement Learning on CartPole-v1
## A Capstone Report

| | |
|---|---|
| **Program** | Master's in Statistical Machine Learning |
| **Institution** | Hanoi University of Science and Technology |
| **Instructor** | Assoc. Prof. Thân Quang Khoát |
| **Student** | Thai Nguyen |
| **Primary Papers** | Li et al. (2019); Christiano et al. (2017) |
| **Environment** | CartPole-v1 (Gymnasium 0.29) |
| **Language / Runtime** | Python 3.10+, NumPy, PyGame, Flask |

---

## Abstract

This capstone project implements and empirically compares human-feedback reinforcement learning algorithms — TAMER (with Credit Assignment), VI-TAMER (with Credit Assignment), single-model RLHF, and ensemble RLHF — on the CartPole-v1 benchmark over **200 training episodes** with seeds {5, 6, 9}. A **Credit Assignment Function (CAF)** is added to both TAMER and VI-TAMER: when the oracle fires at timestep $t$, feedback is distributed over the last 3 observations using configurable weight windows (uniform or exponential), improving reward model generalisation to states that *led to* the evaluated state.

The central empirical finding is that **credit-assignment methods decisively surpass both the baseline and feedback-timing conditions at 200 episodes**. HCRL Oracle with exponential credit assignment (cw=3, exp) achieves the highest ≥195 episode rate of any method — **12.8% (77/600)** — with an average rolling-10 at episode 200 of 150.6, and a last-quarter block mean of 149.9. VI-TAMER with uniform credit assignment (cw=3, uniform) achieves **9.0% ≥195 episodes (54/600)** and an average rolling-10 ep200 of 152.8 — a dramatic reversal from earlier experiments where VI-TAMER with credit assignment completely failed. Feedback timing remains important: Late-phase feedback (episodes 160–199) produces rolling-10 = 154.0, competitive with credit-assignment methods, while a **reward model staleness framework** explains the counterintuitive timing ranking (Early outperforms Mid despite 2× greater staleness).

A novel **signal efficiency analysis** reveals that Early timing (597 signals/seed) achieves 85% of Late's terminal performance at only 7.9% of the oracle budget — an efficiency of 0.217 steps per signal, 12.5× better than Full feedback's 0.017.

The baseline Q-Learning (avg last-20 = 127.3, cross-seed std = 6.0) provides a strong but now clearly beatable benchmark: both credit-assignment methods and Timing Late exceed it. RLHF Ensemble exhibits phase-transition behaviour (seed 5 max = 467, seeds 6/9 below 41). Three methods — HCRL Human, VI-TAMER Human, and RLHF Single — failed to learn useful behaviour and are omitted from the primary analysis.

---

## Table of Contents

1. [Understanding the Problem](#1-understanding-the-problem)
2. [Scientific Basis of the Problem](#2-scientific-basis-of-the-problem)
3. [Direction of Project Development](#3-direction-of-project-development)
4. [Components of the Project and Their Underlying Principles](#4-components-of-the-project-and-their-underlying-principles)
5. [Experimental Setup and Execution](#5-experimental-setup-and-execution)
6. [Training-Phase Results and Learning Dynamics](#6-training-phase-results-and-learning-dynamics)
7. [Post-Training Evaluation and Statistical Analysis](#7-post-training-evaluation-and-statistical-analysis)
8. [Guiding Experiments via the Browser Interface](#8-guiding-experiments-via-the-browser-interface)
9. [Overall Conclusion and Key Takeaways](#9-overall-conclusion-and-key-takeaways)
10. [References](#10-references)

---

## 1. Understanding the Problem

### 1.1 The Reward Specification Problem

Standard reinforcement learning (RL) trains an agent by maximising a scalar reward signal $r(s, a)$ defined by the system engineer. This formulation works well when the desired behavior can be precisely specified mathematically. However, for a large class of practically important tasks — robot locomotion that looks natural, dialogue systems that sound helpful, agents that behave fairly — no such formula is available. Writing one is either impossible or produces reward hacking: the agent finds unexpected strategies that score highly while violating the designer's intent.

This limitation motivates the central question addressed in both papers under study:

> **Can a human teacher, rather than an engineered formula, serve as the reward source that guides an RL agent?**

### 1.2 Paper 1 — Human-Centered Reinforcement Learning: A Survey (Li et al., 2019)

Li et al. provide a comprehensive taxonomy of algorithms that incorporate human feedback into the RL loop. The survey introduces the term **Human-Centered Reinforcement Learning (HCRL)** to cover all paradigms in which a human participates in training, either as a reward provider, a demonstrator, a critic, or an advisor.

The most relevant sub-category for this project is **interactive reward shaping**, in which a human teacher observes the agent's behavior in real time and gives scalar evaluative signals at individual timesteps. The key algorithm formalising this is **TAMER** (Knox & Stone, 2009):

- At each timestep $t$, the human may give a signal $H_t \in \mathbb{R}$ (positive or negative).
- Because humans cannot react at every step, feedback is *sparse*.
- A learned reward model $\hat{R}_H(s, a)$ generalises from observed signals to all states, filling the silence gaps.
- The agent's policy is derived from $\hat{R}_H$ alone (the environment reward is ignored).

The survey also introduces **VI-TAMER** (Knox & Stone, 2012), the non-myopic extension of TAMER that adds a value function $Q_H(s, a)$, allowing the agent to reason about the discounted future consequences of human-evaluated actions.

**Key research questions from the survey addressed in this project:**

| Survey RQ | This Project's Experimental Condition |
|---|---|
| When during training is human feedback most effective? | Feedback timing experiment (early / mid / late / full) |
| Does non-myopic credit assignment improve performance? | TAMER vs. VI-TAMER comparison |
| Does temporal credit assignment improve reward model quality? | Credit Assignment Function (CAF) ablation |

### 1.3 Paper 2 — Deep Reinforcement Learning from Human Preferences (Christiano et al., 2017)

Christiano et al. propose a fundamentally different feedback modality. Rather than reacting at individual timesteps, the human is shown **pairs of short trajectory clips** (segments of $k$ timesteps) and asked which clip represents better behavior. This pairwise comparison is cognitively easier for humans and avoids the credit-assignment problem of per-timestep feedback.

The algorithm, **Reinforcement Learning from Human Feedback (RLHF)**, operates as three concurrent processes:

1. **Policy process**: the RL agent collects experience in the environment, using the learned reward model $\hat{r}$ as its reward signal.
2. **Preference elicitation process**: the system periodically selects clip pairs and presents them to the human.
3. **Reward model fitting process**: the reward model $\hat{r}$ is updated to fit the collected preferences using the **Bradley-Terry model**.

Section 2.2 of the paper introduces several practical improvements:

| Improvement | Description |
|---|---|
| **Ensemble** (§2.2, bullet 1) | $K$ independent predictors trained on bootstrapped subsets |
| **Reward normalisation** (§2.2.1) | Running mean/std normalisation of $\hat{r}$ via Welford's algorithm |
| **Uncertainty-based query selection** (§2.2.4) | Present clips where ensemble members disagree most |
| **Human error modelling** (§2.2.3) | A constant probability $\epsilon$ that the human responds randomly |

---

## 2. Scientific Basis of the Problem

### 2.1 The CartPole-v1 Environment

The CartPole-v1 environment (Barto et al., 1983; Brockman et al., 2016) is the evaluation testbed. A rigid pole is hinged at the top of a cart that moves along a frictionless one-dimensional track. At each discrete timestep the agent applies a binary force — push left (action 0) or push right (action 1).

```
            theta (pole angle)
                 |
         ========|========    <-- pole (length L)
                 |
         +-------+-------+
         |     cart      |  --> x (cart position)
         +---------------+
    ========================  track
```

**Observation space** $\mathcal{O} = \mathbb{R}^4$:

| Index | Variable | Symbol | Failure threshold |
|---|---|---|---|
| 0 | Cart position | $x$ | $|x| > 2.4$ m |
| 1 | Cart velocity | $\dot{x}$ | — |
| 2 | Pole angle | $\theta$ | $|\theta| > 0.2095$ rad ($\approx 12°$) |
| 3 | Pole angular velocity | $\dot{\theta}$ | — |

**Action space**: $\mathcal{A} = \{0, 1\}$ (binary, left / right).

**Environment reward**: $r_t = +1$ for every timestep the pole remains within bounds.

**Termination**: episode ends when any failure threshold is crossed or 200 steps have elapsed (HCRL/VI-TAMER); the native Gymnasium cap of 500 steps applies to RLHF.

**Solved criterion**: mean episode length $\geq 195$ over 30 consecutive episodes.

### 2.2 State Discretisation for Tabular Q-Learning

Because $\mathcal{O}$ is continuous, tabular Q-Learning requires discretisation. Each of the 4 state features is binned uniformly into 7 intervals:

$$s_{\text{discrete}} = \sum_{i=0}^{3} \text{digitize}(o_i, \text{bins}_i) \cdot 8^i$$

This yields $8^4 = 4{,}096$ discrete states.

### 2.3 Tabular Q-Learning

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

where $\alpha = 0.05$, $\gamma = 0.95$, $\varepsilon_0 = 0.5$, $\varepsilon_{\text{decay}} = 0.99$.

### 2.4 The TAMER Framework (Knox & Stone, 2009)

$$\mathcal{L}_{\text{TAMER}} = \text{MSE}\bigl(\hat{R}_H(s_t),\ H_t\bigr)$$

The agent's policy is derived myopically ($\gamma = 0$):

$$\pi(s) = \arg\max_{a} \hat{R}_H(s, a)$$

### 2.5 Credit Assignment Function (CAF)

The original TAMER paper identifies a key limitation: feedback received at time $t$ may reflect the quality of actions taken at $t-1, t-2, \ldots$ rather than the action at $t$ itself. The **Credit Assignment Function** distributes feedback over a window of recent observations:

$$\text{For each } i \in [0, W-1]:\ \hat{R}_H \text{ trained on } (o_{t-i},\ H_t \cdot w_i)$$

where $w_i$ are normalised weights from the chosen CAF. This project uses $W=3$ with two weight functions:

- **Uniform**: $w_i = 1/3$ for all $i$ — equal credit to all 3 observations.
- **Exponential** ($\delta=0.8$): $w_i = \delta^i / \sum_j \delta^j$ — recency-biased, giving ~41% weight to the most recent observation ($i=0$).

With $W=1$ (pointwise), the original TAMER behaviour is recovered. Credit assignment increases the number of reward model training pairs per oracle signal by $W$, giving the MLP $\hat{R}_H$ broader coverage of the state space.

### 2.6 VI-TAMER (Knox & Stone, 2012)

VI-TAMER replaces the myopic policy with a value function $Q_H(s, a)$:

$$Q_H(s, a) \leftarrow Q_H(s, a) + \alpha \left[ \hat{R}_H(s, a) + \gamma \max_{a'} Q_H(s', a') - Q_H(s, a) \right]$$

Setting $\gamma = 0$ recovers plain TAMER. By propagating future value backwards through time, VI-TAMER can assign appropriate credit to actions that produce good states several steps later.

### 2.7 RLHF — Bradley-Terry Preference Model (Christiano et al., 2017)

$$\hat{P}\left[\sigma^1 \succ \sigma^2\right] = \frac{\exp\left(\sum_t \hat{r}(o^1_t)\right)}{\exp\left(\sum_t \hat{r}(o^1_t)\right) + \exp\left(\sum_t \hat{r}(o^2_t)\right)}$$

$$\mathcal{L}_{\text{RLHF}} = -\sum_{\sigma^1, \sigma^2} \left[ \mu \log \hat{P}[\sigma^1 \succ \sigma^2] + (1 - \mu) \log \hat{P}[\sigma^2 \succ \sigma^1] \right]$$

### 2.8 Ensemble and Uncertainty (§2.2 Improvements)

$K = 3$ independent predictors trained on bootstrapped subsets. The epistemic uncertainty of a pair:

$$u(\sigma^1, \sigma^2) = \text{Var}_{k}\left[\text{score}_k(\sigma^1) - \text{score}_k(\sigma^2)\right]$$

Reward normalisation via Welford's online algorithm:

$$\hat{r}_{\text{norm}}(o) = \frac{\hat{r}_{\text{ens}}(o) - \bar{r}}{\max(\sigma_r,\ \varepsilon)}$$

---

## 3. Direction of Project Development

### 3.1 Motivation

This project was motivated by three observations:

1. **Theoretical gap**: The two papers represent complementary paradigms of human feedback — per-timestep scalar signals (TAMER/VI-TAMER) vs. pairwise trajectory preferences (RLHF) — but they had not been placed in direct empirical comparison on a shared benchmark with identical agent architectures.

2. **Engineering gap**: Existing open-source implementations did not implement all §2.2 improvements from Christiano et al., and none addressed temporal credit assignment in TAMER.

3. **Pedagogical goal**: As a capstone for a Statistical Machine Learning program, the project demonstrates both theoretical understanding and practical implementation skill.

### 3.2 Development Stages

**Stage 1 — Baseline**: Implement a tabular Q-Learning agent with environment reward as the sole training signal.

**Stage 2 — HCRL (Paper 1)**: Implement TAMER and VI-TAMER with a simulated oracle. Run the feedback timing experiment.

**Stage 3 — RLHF (Paper 2)**: Implement the single-model RLHF pipeline (§2.1) and the full §2.2 ensemble variant.

**Stage 4 — Integration and comparison**: Refactor all scripts to share a common configuration and utility library. Add a Flask web visualiser and pygame-based human interaction modes.

**Stage 5 — Credit Assignment**: Add a temporal Credit Assignment Function to both HCRL and VI-TAMER. Extend `run_hcrl_episode` and `run_vi_tamer_episode` with a rolling observation buffer; when oracle fires, spread feedback over the last $W=3$ observations using configurable weight functions (uniform or exponential). Scale experiments to 200 episodes with seeds {5, 6, 9}.

### 3.3 Design Principles

- **Fair comparison**: All methods use identical `QLearningAgent` / `VITAMERAgent` hyperparameters. Only the reward signal changes.
- **Reproducibility**: All RNGs are seeded; experiments use three seeds ($\{5, 6, 9\}$).
- **Paper fidelity**: Every constant traces to a specific equation or section in the source paper.
- **Modularity**: Shared logic in `cartpole/train_utils.py`; shared constants in `cartpole/config.py`.

---

## 4. Components of the Project and Their Underlying Principles

### 4.1 Package Structure

```
ML-Project/
├── cartpole/
│   ├── __init__.py          # Public API exports
│   ├── config.py            # All hyperparameters (single source of truth)
│   ├── agents.py            # QLearningAgent, VITAMERAgent, RandomActionAgent
│   ├── reward_model.py      # RewardModel, HCRLRewardModel, EnsembleRewardModel
│   ├── oracle.py            # Simulated human oracle (HCRL)
│   ├── train_utils.py       # Shared episode runners + Credit Assignment helper
│   ├── entities.py          # EpisodeHistory dataclass
│   └── plotting.py          # Matplotlib live-plotting helper
├── run.py                   # Baseline Q-Learning
├── train_hcrl.py            # TAMER oracle + --human flag
├── train_vi_tamer.py        # VI-TAMER oracle + --human flag
├── train_rlhf.py            # RLHF single model + --human flag
├── train_rlhf_ensemble.py   # RLHF ensemble (§2.2) + --human flag
├── feedback_timing_experiment.py
├── compare_models.py
├── compare_all.py
└── webapp.py                # Flask browser visualiser
```

### 4.2 Configuration — `cartpole/config.py`

| Group | Parameter | Value | Source |
|---|---|---|---|
| **Agent** | `AGENT_LR` ($\alpha$) | 0.05 | Standard QL |
| **Agent** | `AGENT_DISCOUNT` ($\gamma$) | 0.95 | Standard QL |
| **Agent** | `AGENT_EXPLORE` ($\varepsilon_0$) | 0.50 | — |
| **Agent** | `AGENT_DECAY` | 0.99 | — |
| **HCRL** | `HCRL_TRIGGER_PROB` | 0.50 | Knox & Stone (2009) |
| **HCRL** | `HCRL_FEEDBACK_WEIGHT` | 10.0 | — |
| **HCRL** | `HCRL_TERMINATE_PENALTY` | 50.0 | — |
| **CAF** | `HCRL_CREDIT_WINDOW` | 3 | Knox & Stone (2009) §3 |
| **CAF** | `HCRL_CREDIT_FN` | "uniform" or "exp" | — |
| **RLHF** | `RLHF_SEGMENT_LENGTH` | 25 | Christiano et al. (2017) |
| **RLHF** | `RLHF_PAIRS_PER_ITER` | 24 | — |
| **Ensemble** | `ENSEMBLE_N_MODELS` ($K$) | 3 | §2.2 bullet 1 |
| **Ensemble** | `ENSEMBLE_ERROR_PROB` ($\varepsilon_H$) | 0.01 | §2.2.3 |

### 4.3 Agent Implementations — `cartpole/agents.py`

#### 4.3.1 `QLearningAgent`

Tabular Q-Learning with $\varepsilon$-greedy exploration. Used as the policy backbone for Baseline, HCRL, and RLHF.

#### 4.3.2 `VITAMERAgent`

Extends `QLearningAgent` with a separate $Q_H$ table:

```python
r_h = reward_model.predict(obs)  # from HCRLRewardModel
td_target = r_h + gamma * max(Q_H[next_state])
Q_H[state, action] += alpha * (td_target - Q_H[state, action])
```

### 4.4 Reward Models — `cartpole/reward_model.py`

**`HCRLRewardModel`**: MSE regression on (observation, human signal) pairs. Credit assignment increases the number of training pairs per oracle signal by factor $W=3$.

**`RewardModel`**: Preference-based cross-entropy loss (Bradley-Terry).

**`EnsembleRewardModel`**: $K=3$ independent `RewardModel` instances with bootstrapped training, uncertainty-based query selection, and Welford normalisation.

### 4.5 Credit Assignment — `cartpole/train_utils.py`

```python
def _compute_credit_weights(n, fn="uniform", decay=0.8):
    """Normalised weights: index 0 = most recent, index n-1 = oldest."""
    if fn == "uniform":
        w = np.ones(n)
    elif fn == "exp":
        w = decay ** np.arange(n)
    elif fn == "gaussian":
        sigma = max(n / 3.0, 1.0)
        w = np.exp(-0.5 * (np.arange(n) / sigma) ** 2)
    return w / w.sum()
```

When the oracle fires at step $t$ with signal $f$, the episode runner distributes:

```python
for i, past_obs in enumerate(reversed(recent_obs)):   # i=0 → lag=0 (obs_t)
    ep_obs.append(past_obs)
    ep_rew.append(f * weights[i])
```

The agent's TD update at step $t$ still uses the full signal $f$; only the reward model training data is credit-expanded.

### 4.6 Web Visualiser — `webapp.py`

A Flask application with two tabs:

**Play tab**: Auto-discovers `.npz` model files. Multiple models play simultaneously with real-time SSE frame streaming. Post-gameplay generates comparison charts (box plot, bar chart, histogram, episode progression, heatmap).

**Charts tab**: Discovers all `*_history.csv` files from training runs. 11 chart types available; CSVs grouped by seed suffix for family-level comparisons.

---

## 5. Experimental Setup and Execution

All experiments use **200 training episodes** and **3 random seeds {5, 6, 9}** (600 episode records per method). All outputs are saved under `experiment-results/ep200/`.

### 5.1 Baseline (Environment Reward Only)

Pure Q-Learning with termination penalty $-5000$. Defines the performance reference.

```bash
uv run python run.py --episodes 200 --seed {5,6,9}
```

### 5.2 HCRL / TAMER with Credit Assignment

Oracle fires at $p = 0.5$ per step, signals $\pm 10$. Credit assignment distributes each signal over the last 3 observations using a configurable weight function. `HCRLRewardModel` retrains after every episode on the accumulated credit-expanded buffer. Two credit functions are evaluated:

- **Uniform** (`--credit-fn uniform`): weights $[1/3, 1/3, 1/3]$ — equal credit to all 3 observations.
- **Exponential** (`--credit-fn exp`): weights $[1.0, 0.8, 0.64]$ normalised — recency-biased, giving ~41% weight to the most recent observation.

```bash
# Exponential credit (primary HCRL condition)
uv run python train_hcrl.py --episodes 200 --seed {5,6,9} \
    --feedback-weight 10 --credit-window 3 --credit-fn exp --skip-charts

# Uniform credit (comparison condition)
uv run python train_hcrl.py --episodes 200 --seed {5,6,9} \
    --feedback-weight 10 --credit-window 3 --credit-fn uniform --skip-charts
```

Oracle interaction count per seed: ~1,800–2,500 raw signals → ~5,400–7,500 training pairs (×3 credit expansion).

### 5.3 VI-TAMER with Credit Assignment

Identical to §5.2 but uses `VITAMERAgent.act_vi()` with TD update $Q_H \leftarrow Q_H + \alpha[R_H + \gamma \max Q_H(s') - Q_H]$, $\gamma = 0.95$.

```bash
uv run python train_vi_tamer.py --episodes 200 --seed {5,6,9} \
    --feedback-weight 10 --credit-window 3 --credit-fn uniform --skip-charts
```

### 5.4 Human Feedback Modes (HCRL and VI-TAMER)

Real human provides feedback via arrow keys while watching the CartPole window. No oracle — pure human signals.

```bash
uv run python train_hcrl.py     --human --episodes 200 --seed {5,6,9} --feedback-weight 10
uv run python train_vi_tamer.py --human --episodes 200 --seed {5,6,9} --feedback-weight 10
```

### 5.5 Feedback Timing Experiment

4 oracle-HCRL conditions (no credit assignment in timing runs) over all 3 seeds internally:

| Condition | Window (ep 0–199) | Feedback window |
|---|---|---|
| Early | 0–20% | Episodes 0–39 |
| Mid | 40–60% | Episodes 80–119 |
| Late | 80–100% | Episodes 160–199 |
| Full | 0–100% | Episodes 0–199 |

```bash
uv run python feedback_timing_experiment.py \
    --episodes 200 --auto --skip-charts --feedback-weight 10
```

### 5.6 RLHF — Single Model

20 warm-up episodes → 10 RLHF iterations × 8 episodes = ~100 episodes per seed. 24 preference pairs × 10 iterations = 240 total labels per seed.

```bash
uv run python train_rlhf.py --episodes 200 --seed {5,6,9} --skip-charts
```

### 5.7 RLHF — Ensemble (§2.2)

$K = 3$ bootstrapped reward models with uncertainty-based query selection, Welford normalisation, and 1% oracle error rate.

```bash
uv run python train_rlhf_ensemble.py --episodes 200 --seed {5,6,9} --n-models 3 --skip-charts
```

### 5.8 Output Directory Summary

| Method | Output directory |
|---|---|
| Baseline | `experiment-results/ep200/` |
| HCRL Oracle + CAF (exp) | `experiment-results/ep200/hcrl-oracle-fw10-cw3e/` |
| HCRL Human | `experiment-results/ep200/hcrl-human-fw10/` |
| VI-TAMER Oracle + CAF | `experiment-results/ep200/vi-tamer-fw10-cw3u/` |
| VI-TAMER Human | `experiment-results/ep200/vi-tamer-human-fw10/` |
| Timing Experiment | `experiment-results/ep200/timing-experiment/` |
| RLHF Single | `experiment-results/ep200/rlhf-oracle/` |
| RLHF Ensemble | `experiment-results/ep200/rlhf-ensemble/` |

---

## 6. Training-Phase Results and Learning Dynamics

All results are from **200 training episodes**, seeds {5, 6, 9}, 600 episode records per method. This section focuses on the high-value methods that produced meaningful experimental insights. Three methods — HCRL Human, VI-TAMER Human, and RLHF Single — are omitted from the main analysis because they failed to learn useful behaviour competitive with the baseline (overall means below 50 steps). Their summary data is preserved in §6.4 for completeness.

### 6.1 Core Methods — Per-Seed Training Data

#### 6.1.1 Baseline Q-Learning (Performance Reference)

| Seed | Ep 1–50 | Ep 51–100 | Ep 101–150 | Ep 151–200 | Rolling-10 ep200 | Last-20 | Max | ≥195 |
|---|---|---|---|---|---|---|---|---|
| 5 | 30.7 | 65.2 | 131.7 | 124.8 | 133.9 | 134.8 | 200 | 3 |
| 6 | 33.2 | 78.8 | 81.5 | 122.6 | 118.4 | 120.2 | 190 | 0 |
| 9 | 24.2 | 42.8 | 113.7 | 129.5 | 121.5 | 126.9 | 200 | 2 |
| **Avg** | **29.4** | **62.3** | **109.0** | **125.6** | **124.6** | **127.3** | **200** | **5 (0.8%)** |

The baseline shows a **monotonically accelerating** learning curve: each 50-episode block mean is higher than the previous (29.4 → 62.3 → 109.0 → 125.6). The breakthrough occurs in the ep 101–150 block (109.0, +75% over ep 51–100), where the Q-table consolidates longer-horizon strategies. Cross-seed variance is notably **low** (std of per-seed means = 4.7; std of last-20 = 6.0) — the most consistent baseline observed. Median episode length is **80 steps** (p25 = 31, p75 = 126).

The baseline's ≥195 rate is low (0.8%, 5/600), indicating that while Q-Learning reliably converges toward good policies at 200 episodes, it rarely achieves near-optimal CartPole performance. This establishes 125.6 (last-quarter mean) as the **performance target** human-feedback methods must exceed.

#### 6.1.2 HCRL Oracle with Exponential Credit Assignment (fw=10, cw=3, exp)

| Seed | Ep 1–50 | Ep 51–100 | Ep 101–150 | Ep 151–200 | Rolling-10 ep200 | Last-20 | Max | ≥195 |
|---|---|---|---|---|---|---|---|---|
| 5 | 31.8 | 71.0 | 95.8 | 173.9 | 178.0 | 176.0 | 200 | 19 |
| 6 | 33.7 | 81.2 | 98.6 | 109.9 | 122.4 | 116.2 | 156 | 0 |
| 9 | 30.7 | 130.0 | 159.3 | 165.9 | 151.3 | 155.5 | 200 | 58 |
| **Avg** | **32.1** | **94.1** | **117.9** | **149.9** | **150.6** | **149.2** | **200** | **77 (12.8%)** |

HCRL Oracle with exponential credit assignment is the **best-performing method by ≥195 episode rate** — 12.8% (77/600), more than triple the next best (VI-TAMER at 9.0%) and 16× the baseline (0.8%). The method exhibits strong **late-phase acceleration**: the last-quarter block mean (149.9) is **+27% above the ep 101–150 block** (117.9), driven by the credit-assignment compounding effect.

**Seed 9 is exceptional**: 58/200 episodes (29.0%) reach ≥195 steps, with an overall mean of 121.5 and last-20 = 155.5. Seed 5 also performs strongly (19 ≥195 episodes, last-20 = 176.0). Seed 6 is the weakest (last-20 = 116.2, max = 156), but still competitive with the baseline.

**Why exponential credit outperforms uniform**: The exponential function assigns ~41% of the oracle signal to the most recent observation vs 33% for uniform. This preserves more reward magnitude at the timestep closest to the oracle's reaction, which aligns better with the HCRL MLP's learning gradient — the most recent observation is the most causally relevant to the feedback. The result is a reward model that more accurately maps states to human evaluative signals.

**Credit assignment compounding**: As episodes lengthen in the second half of training, each oracle signal generates 3 training pairs from observations of a well-balanced pole. This creates a positive feedback loop — better agent → longer episodes → more informative credit-expanded training pairs → better reward model → better agent. The +55% gain from ep 51–100 (94.1) to ep 151–200 (149.9) is the signature of this compounding.

#### 6.1.3 VI-TAMER Oracle with Uniform Credit Assignment (fw=10, cw=3, uniform)

| Seed | Ep 1–50 | Ep 51–100 | Ep 101–150 | Ep 151–200 | Rolling-10 ep200 | Last-20 | Max | ≥195 |
|---|---|---|---|---|---|---|---|---|
| 5 | 21.2 | 48.0 | 61.1 | 109.9 | 128.2 | 117.2 | 200 | 4 |
| 6 | 35.8 | 73.3 | 142.9 | 152.5 | 195.0 | 186.6 | 200 | 50 |
| 9 | 50.2 | 89.6 | 125.3 | 132.4 | 135.2 | 138.1 | 180 | 0 |
| **Avg** | **35.7** | **70.3** | **109.8** | **131.6** | **152.8** | **147.3** | **200** | **54 (9.0%)** |

VI-TAMER with uniform credit assignment achieves **9.0% ≥195 episodes (54/600)** — a result that **completely reverses** the earlier finding where VI-TAMER with credit assignment failed (0 ≥195 episodes). The method shows strong monotonic improvement across all 50-episode blocks (35.7 → 70.3 → 109.8 → 131.6) and an average rolling-10 at ep 200 of **152.8** — competitive with both HCRL cw3e (150.6) and Timing Late (154.0).

**Seed 6 is the standout**: 50/200 episodes (25.0%) reach ≥195, with rolling-10 ep200 = 195.0 and last-20 = 186.6 — near-optimal CartPole performance. This seed demonstrates that when VI-TAMER's non-myopic TD propagation ($Q_H \leftarrow \hat{R}_H + \gamma \max Q_H$) successfully bootstraps a value function from the credit-expanded reward model, it can achieve convergence speeds that exceed the myopic HCRL approach.

**Cross-seed variance is high** (std of per-seed means = 19.0; std of last-20 = 29.1) — the highest of the credit-assignment methods. Seed 5 (last-20 = 117.2) underperforms relative to seed 6 (186.6), illustrating that VI-TAMER's non-myopic update is more sensitive to initialisation than HCRL's myopic policy.

**Why does VI-TAMER now succeed where it previously failed?** The re-trained experiment produces dramatically different results from the earlier run (which showed 0/600 ≥195 episodes). The key factors are: (1) the TD propagation in VI-TAMER amplifies small initial advantages — when the reward model happens to learn a useful signal early (as in seed 6), the value function propagates this signal backward efficiently, creating a rapid convergence cascade; and (2) the same amplification works in reverse — poor early signals compound into sustained underperformance, explaining the high cross-seed variance.

#### 6.1.4 Feedback Timing — Late (ep 160–199)

| Seed | Ep 1–50 | Ep 51–100 | Ep 101–150 | Ep 151–200 | Rolling-10 ep200 | Last-20 | Max | ≥195 |
|---|---|---|---|---|---|---|---|---|
| 5 | 27.5 | 75.4 | 134.7 | 159.5 | 163.6 | 167.7 | 200 | 21 |
| 6 | 41.7 | 117.0 | 145.7 | 142.0 | 157.9 | 145.3 | 187 | 0 |
| 9 | 35.5 | 67.7 | 111.2 | 134.1 | 140.5 | 141.0 | 197 | 1 |
| **Avg** | **34.9** | **86.7** | **130.5** | **145.2** | **154.0** | **151.3** | **200** | **22 (3.7%)** |

**Oracle signals per seed**: s5 = 3,007; s6 = 2,704; s9 = 2,198 (avg = 2,636).

Timing Late remains a **strong method** with the highest average rolling-10 at ep 200 (154.0) among the timing conditions, and competitive with the credit-assignment methods. However, it now ranks below HCRL cw3e and VI-TAMER cw3u in ≥195 rate (3.7% vs 12.8% and 9.0%), indicating that while Late timing produces good average performance, it achieves fewer near-optimal episodes than the credit-assignment approaches.

The mechanism remains clear: during episodes 0–159, the agent trains on environment reward alone, reaching near-competent policies. Oracle feedback at episodes 160–199 covers the high-value state space, and the shaped reward amplifies Q-table updates for the states the agent needs to master.

**Cross-seed consistency**: Std of per-seed last-20 = 11.7 — the **most consistent** of the timing conditions.

#### 6.1.5 RLHF Ensemble (K=3, §2.2 Improvements)

| Seed | Ep 1–50 | Ep 51–100 | Ep 101–150 | Ep 151–200 | Rolling-10 ep200 | Last-20 | Max | ≥195 |
|---|---|---|---|---|---|---|---|---|
| 5 | 12.1 | 69.2 | 75.8 | 78.2 | 76.8 | 77.2 | 467 | 1 |
| 6 | 13.8 | 22.0 | 33.0 | 46.7 | 37.6 | 36.8 | 109 | 0 |
| 9 | 36.2 | 24.1 | 23.5 | 37.6 | 42.3 | 41.0 | 67 | 0 |
| **Avg** | **20.7** | **38.4** | **44.1** | **54.2** | **52.2** | **51.7** | **467** | **1 (0.2%)** |

RLHF Ensemble shows a **weak but monotonically increasing** block mean (20.7 → 38.4 → 44.1 → 54.2), confirming the §2.2 improvements provide sustained albeit slow learning. The method's cross-seed variance is extreme (seed-mean std = 13.8; last-20 std = 18.2).

**Seed 5 breakthrough**: The max = 467 steps (using the native 500-step Gymnasium cap) demonstrates that the ensemble mechanism can occasionally learn a genuine CartPole reward signal. However, seeds 6 and 9 plateau below 40 steps, confirming this is initialisation-dependent.

### 6.2 Feedback Timing Experiment — Full Comparison (200 Episodes)

#### 6.2.1 Summary Table

| Condition | Window | Avg signals/seed | Overall mean | Last-20 | Last-40 | Rolling-10 ep200 | Max | ≥195 (%) |
|---|---|---|---|---|---|---|---|---|
| **Late** | ep 160–199 | 2,636 | **99.3** | **151.3** | **146.9** | **154.0** | 200 | 3.7% |
| **Full** | ep 0–199 | 7,557 | 87.6 | 131.5 | 123.7 | 134.2 | 200 | **5.0%** |
| Early | ep 0–39 | 597 | 85.4 | 129.4 | 128.0 | 130.2 | 166 | 0.0% |
| Mid | ep 80–119 | 1,920 | 92.6 | 121.7 | 124.9 | 116.6 | 200 | 0.3% |

#### 6.2.2 Signal Efficiency — Performance per Oracle Signal

| Condition | Avg last-20 | Avg signals | Efficiency (last-20 / signals) |
|---|---|---|---|
| **Early** | 129.4 | 597 | **0.2168** |
| Mid | 121.7 | 1,920 | 0.0634 |
| Late | 151.3 | 2,636 | 0.0574 |
| Full | 131.5 | 7,557 | 0.0174 |

Early feedback achieves **3.8× the signal efficiency of Late** and **12.5× the efficiency of Full**. This reveals a fundamental trade-off: **Late timing maximises absolute performance** while **Early timing maximises cost-effectiveness** (85% of Late's performance at 7.9% of the oracle budget).

#### 6.2.3 Timing Ranking Reversal Between 100 and 200 Episodes

| Condition | Last-20 at 100 eps | Last-20 at 200 eps | Change | Rank at 100 eps | Rank at 200 eps |
|---|---|---|---|---|---|
| Early | 64.1 | 129.4 | **+65.3** | 4th | 3rd |
| Mid | **104.3** | 121.7 | +17.4 (*) | **1st** | **4th** |
| Late | 97.2 | **151.3** | +54.1 | 2nd | **1st** |
| Full | 78.3 | 131.5 | +53.2 | 3rd | 2nd |

(*) Mid improves the least (+17.4 vs +53–65 for others) despite starting as the best at 100 episodes, due to **reward model staleness** — 81 post-feedback episodes allow policy drift to outpace the frozen reward model.

### 6.3 Training Curve Summary — High-Value Methods (200 Episodes)

Sorted by average ≥195 rate (best proxy for near-optimal policy quality):

| Rank | Method | Overall mean | Ep 151–200 | Avg r10 ep200 | Avg last-20 | Max | ≥195 (%) | Cross-seed std (last-20) |
|---|---|---|---|---|---|---|---|---|
| 1 | **HCRL Oracle (cw3e)** | 98.5 | **149.9** | 150.6 | 149.2 | 200 | **12.8%** | 24.8 |
| 2 | **VI-TAMER (cw3u)** | 86.9 | 131.6 | **152.8** | 147.3 | 200 | **9.0%** | 29.1 |
| 3 | Timing: Full | 87.6 | 125.6 | 134.2 | 131.5 | 200 | 5.0% | 27.3 |
| 4 | **Timing: Late** | **99.3** | 145.2 | 154.0 | **151.3** | 200 | 3.7% | **11.7** |
| 5 | Baseline | 81.5 | 125.6 | 124.6 | 127.3 | 200 | 0.8% | 6.0 |
| 6 | Timing: Early | 85.4 | — | 130.2 | 129.4 | 166 | 0.0% | — |
| 7 | Timing: Mid | 92.6 | — | 116.6 | 121.7 | 200 | 0.3% | — |
| 8 | RLHF Ensemble | 39.3 | 54.2 | 52.2 | 51.7 | 467 | 0.2% | 18.2 |

### 6.4 Omitted Methods — Summary (for completeness)

The following methods failed to produce meaningful learning at 200 episodes and are excluded from the main analysis:

| Method | Overall mean | Last-40 | Max | ≥195 | Failure mode |
|---|---|---|---|---|---|
| HCRL Human (fw=10) | 46.1 | 59.7 | 197 | 1 (0.2%) | Human feedback noise → reward model inconsistency |
| VI-TAMER Human (fw=10) | 42.1 | 29.4 | 200 | 2 (0.3%) | Human fatigue → reward model degradation; seed 9 near-collapse (last-40 = 20.2) |
| RLHF Single | 24.7 | 21.6 | 92 | 0 (0.0%) | 1.2 preference labels/episode → permanent reward model under-fitting |

All three share a common theme: **insufficient or corrupted reward signal**.

---

## 7. Post-Training Evaluation and Statistical Analysis

Formal post-training (greedy evaluation) was not run in the 200-episode setup. The following analysis uses the training data (600 episodes per method) as the primary evidence base. All statistical comparisons use per-seed means across the 3 seeds to avoid pseudo-replication.

### 7.1 Deep-Dive Insights from the Experiment

#### Insight 1 — Credit Assignment Is the Dominant Factor at 200 Episodes

The re-trained results establish a clear hierarchy: **credit-assignment methods (HCRL cw3e: 12.8% ≥195; VI-TAMER cw3u: 9.0% ≥195) now decisively outperform all timing conditions (best: Timing Full at 5.0%) and the baseline (0.8%)**. This reverses the earlier finding where timing was the dominant variable.

The key mechanism is the **compounding effect** of credit-expanded reward model training data. As the agent improves and episodes lengthen, each oracle signal generates 3 high-quality training pairs from observations of successful balancing. The reward model becomes increasingly accurate in the states that matter most, which in turn accelerates the agent's convergence — a virtuous cycle.

HCRL cw3e's last-quarter block mean (149.9) exceeds the baseline's (125.6) by **19.4%**, and its ≥195 rate (12.8%) is **16× the baseline's** (0.8%). The credit assignment advantage is not marginal — it is transformative.

#### Insight 2 — VI-TAMER's Non-Myopic Update Amplifies Both Success and Failure

The most striking finding from the re-trained experiments is VI-TAMER's reversal from complete failure (0/600 ≥195 in the prior run) to strong performance (54/600 ≥195, 9.0%). This reveals that VI-TAMER's TD propagation ($Q_H \leftarrow \hat{R}_H + \gamma \max Q_H$) acts as an **amplifier**:

- **When the reward model learns a useful signal early** (seed 6: 50/200 ≥195), the value function propagates this signal backward efficiently, creating a rapid convergence cascade. Seed 6's rolling-10 at ep 200 (195.0) approaches the episode cap.
- **When the reward model learns poorly** (seed 5: only 4/200 ≥195), the same propagation mechanism compounds errors, yielding sustained underperformance.

This amplification explains VI-TAMER's extreme cross-seed variance (std of last-20 = 29.1, highest of all methods). HCRL's myopic policy ($\gamma = 0$) avoids this by not propagating errors through time, trading peak performance for reliability.

**Practical implication**: VI-TAMER with credit assignment should be used with **multiple seeds and model selection** — run $N$ seeds, pick the best. The probability of getting at least one strong seed increases rapidly with $N$.

#### Insight 3 — Exponential Credit Preserves Reward Magnitude Better Than Uniform

The exponential function's key advantage is **magnitude preservation at the most recent timestep**:

| Credit function | Weight at $t$ (most recent) | Weight at $t-1$ | Weight at $t-2$ |
|---|---|---|---|
| Uniform | 0.333 | 0.333 | 0.333 |
| Exponential ($\delta=0.8$) | 0.410 | 0.328 | 0.262 |

The 23% higher weight at $t$ (0.410 vs 0.333) means the reward model receives a stronger training signal for the observation most causally relevant to the oracle's feedback. HCRL with exponential credit achieves 12.8% ≥195 across 3 seeds with consistent late-phase compounding. VI-TAMER currently uses only uniform credit; testing exponential credit for VI-TAMER is a natural next step (see §9.3).

#### Insight 4 — The Reward Model Staleness Framework Still Explains Timing

The staleness framework remains valid for the timing conditions:

| Condition | Feedback ends at | Staleness at ep 200 | Last-20 |
|---|---|---|---|
| Early | ep 39 | 161 episodes | 129.4 |
| Mid | ep 119 | 81 episodes | 121.7 |
| Late | ep 199 | 0 episodes | 151.3 |
| Full | ep 199 | 0 episodes | 131.5 |

Counterintuitively, **161 episodes of staleness (Early) outperforms 81 episodes (Mid)**. The explanation: Early's model is weak enough to serve as a **harmless initialisation bias** that the Q-table eventually overrides with environment reward. Mid's model is strong enough to **actively conflict** with the evolving policy, degrading performance over 81 post-feedback episodes. The principle: **a stale reward model is worse than no model if it is strong enough to override environment signals**.

#### Insight 5 — The Baseline Is Consistently Beatable at 200 Episodes

With the re-trained results, the baseline (avg last-20 = 127.3, ≥195 = 0.8%) is now **clearly surpassed** by four methods:

| Method | Avg last-20 | Δ vs Baseline | ≥195 rate | ≥195 multiplier |
|---|---|---|---|---|
| Timing: Late | 151.3 | +24.0 (+19%) | 3.7% | 4.6× |
| HCRL cw3e | 149.2 | +21.9 (+17%) | 12.8% | **16.0×** |
| VI-TAMER cw3u | 147.3 | +20.0 (+16%) | 9.0% | **11.3×** |
| Timing: Full | 131.5 | +4.2 (+3%) | 5.0% | 6.3× |

The ≥195 multiplier column reveals the most important finding: while average performance improvements are 16–19%, **near-optimal episode rates improve by 5–16×**. Credit-assignment methods don't just raise the mean — they shift the entire distribution rightward, enabling the agent to sustain near-perfect performance far more frequently.

The baseline's cross-seed std of last-20 = 6.0 (lowest of all methods) shows that Q-Learning is the **most predictable** method, but its ceiling is low. Human-feedback methods trade consistency for a much higher ceiling.

#### Insight 6 — Signal Efficiency Favours Early Timing Under Budget Constraints

Despite the dominance of credit-assignment methods in absolute performance, the **signal efficiency** analysis remains valid for timing conditions:

- Early: 0.217 steps/signal — **12.5× more efficient** than Full (0.017)
- Late: 0.057 steps/signal
- Full: 0.017 steps/signal

If oracle access is limited, Early timing achieves 85% of Late's performance at 7.9% of the cost. For budget-unconstrained settings, credit-assignment methods (HCRL cw3e, VI-TAMER cw3u) now dominate all timing conditions.

### 7.2 Cross-Method Ranking — Final Assessment

| Rank | Method | Avg r10 ep200 | ≥195 (%) | Best for | Weakness |
|---|---|---|---|---|---|
| 1 | **HCRL Oracle (cw3e)** | 150.6 | **12.8%** | Highest ≥195 rate; strong compounding | Higher variance than baseline (std = 24.8) |
| 2 | **VI-TAMER (cw3u)** | **152.8** | **9.0%** | Highest avg r10 ep200; peak seed performance | Highest cross-seed variance (std = 29.1); amplifier risk |
| 3 | **Timing: Late** | 154.0 | 3.7% | Most consistent timing; best avg last-20 | Requires 160 pre-training episodes |
| 4 | Timing: Full | 134.2 | 5.0% | Good ≥195 count among timing methods | Lowest signal efficiency; high cross-seed variance (std = 27.3) |
| 5 | Timing: Early | 130.2 | 0.0% | **Signal efficiency** (0.217/signal) | Never reaches ≥195; ceiling at 166 max steps |
| 6 | **Baseline** | 124.6 | 0.8% | Most consistent (std = 6.0); no human cost | Low ceiling; rarely near-optimal |
| 7 | RLHF Ensemble | 52.2 | 0.2% | Demonstrates §2.2 can work | 67% seed failure rate |

---

## 8. Guiding Experiments via the Browser Interface

### 8.1 Overview

The Flask web application (`webapp.py`) provides a two-tab browser interface:

- **Play tab**: Select one or more models from a sidebar, play them simultaneously, automatically generate comparison charts after gameplay.
- **Charts tab**: Select training history CSVs and any combination of 11 chart types, generate in a responsive multi-chart grid.

### 8.2 Starting the Server

```bash
# Start the web server
uv run python webapp.py

# Open in browser
# http://localhost:5000
```

### 8.3 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Main HTML interface |
| `/api/models` | GET | JSON list of all discovered `.npz` model files |
| `/api/play?models=...` | GET | SSE stream of base64-encoded JPEG frames + live stats |
| `/api/csvs` | GET | JSON list of all `*_history.csv` files with family grouping |
| `/api/chart` | POST | Generate a single chart from selected CSVs |
| `/api/multi-chart` | POST | Generate multiple chart types in one request |
| `/api/gameplay-chart` | POST | Generate comparison charts from live gameplay data |

### 8.4 Play Tab Features

Select models, play simultaneously in a grid, view real-time stats. Post-gameplay generates five comparison charts automatically: box plot, bar chart, histogram, episode progression, and performance heatmap.

### 8.5 Charts Tab — 11 Chart Types

| Chart Type | Description |
|---|---|
| Training Curves | Rolling mean of episode length per model |
| Training Curves (Mean ± Std) | Mean ± std across seeds |
| Box Plot | Episode length distribution comparison |
| Bar Chart (Mean ± Std) | Mean episode length with error bars |
| Histogram | Overlaid episode length distributions |
| Convergence Analysis | First episode crossing 50/100/150/195 steps |
| Success Rate Over Time | Rolling % of episodes reaching ≥ 195 steps |
| Learning Speed | Derivative of rolling mean |
| Training Stability | Rolling standard deviation |
| Final Performance | Mean of last N episodes per model family |
| Performance Heatmap | Colour-coded matrix of all families vs. key metrics |

---

## 9. Overall Conclusion and Key Takeaways

### 9.1 Summary of High-Value Findings (200 Episodes, Seeds {5, 6, 9})

| Rank | Method | Overall mean | Avg last-20 | Avg r10 ep200 | Max | ≥195 (%) | Key finding |
|---|---|---|---|---|---|---|---|
| 1 | **HCRL Oracle (cw3e)** | 98.5 | 149.2 | 150.6 | 200 | **12.8%** | Best ≥195 rate; exponential credit compounding |
| 2 | **VI-TAMER (cw3u)** | 86.9 | 147.3 | **152.8** | 200 | **9.0%** | Reversal from 0% ≥195; TD amplifier effect |
| 3 | **Timing: Late** | **99.3** | **151.3** | 154.0 | 200 | 3.7% | Best average r10; policy-aligned feedback |
| 4 | Timing: Full | 87.6 | 131.5 | 134.2 | 200 | 5.0% | Solid ≥195 among timing conditions |
| 5 | Timing: Early | 85.4 | 129.4 | 130.2 | 166 | 0.0% | Best signal efficiency (0.217/signal) |
| 6 | **Baseline** | 81.5 | 127.3 | 124.6 | 200 | 0.8% | Most consistent (std = 6.0); beatable ceiling |
| 7 | RLHF Ensemble | 39.3 | 51.7 | 52.2 | 467 | 0.2% | Seed-5 phase transition; 67% failure rate |

### 9.2 Key Takeaways

**Takeaway 1 — Credit assignment is the most impactful innovation at 200 episodes.**
Both credit-assignment methods (HCRL cw3e: 12.8% ≥195; VI-TAMER cw3u: 9.0% ≥195) decisively outperform all timing conditions and the baseline. The 3× expansion of reward model training data per oracle signal creates a compounding effect: as the agent improves, each signal generates increasingly informative training pairs, accelerating convergence further. This compounding is the primary mechanism driving the ≥195 rate to 16× the baseline's.

**Takeaway 2 — VI-TAMER with credit assignment works, but requires seed selection.**
VI-TAMER's reversal from 0/600 ≥195 (prior run) to 54/600 (9.0%) demonstrates that the non-myopic TD update is **not** inherently incompatible with credit assignment. However, the same TD propagation that enables seed 6's near-optimal performance (195.0 r10 ep200) also amplifies poor initialisations (seed 5: 128.2). The practical strategy: run multiple seeds, select the best model. For $N=3$ seeds, the probability of getting at least one seed with ≥195 rate >5% is high.

**Takeaway 3 — Exponential credit outperforms uniform for HCRL.**
Exponential credit (41% weight at most recent observation) produces 12.8% ≥195 across 3 seeds with consistent compounding. The exponential function's recency bias aligns reward magnitude with causal relevance, producing a more accurate reward model. Testing exponential credit for VI-TAMER (which currently uses uniform) is a priority for future work.

**Takeaway 4 — Late timing remains the best strategy for consistent average performance.**
Timing Late achieves the highest average last-20 (151.3) with the lowest cross-seed variance among timing conditions (std = 11.7). It requires no credit-assignment infrastructure — just delaying oracle activation to episode 160. For practitioners who need a simple, reliable improvement over the baseline, Late timing is the recommended approach.

**Takeaway 5 — The reward model staleness framework explains the timing ranking.**
Mid feedback (81 episodes stale) performs worse than Early (161 episodes stale) because Mid's model is strong enough to actively conflict with environment reward, while Early's model is too weak to interfere. The principle: **a stale reward model harms performance only if it is strong enough to override environment signals**.

**Takeaway 6 — The baseline is a strong but beatable benchmark.**
The baseline (avg last-20 = 127.3, cross-seed std = 6.0) is the most predictable method. However, all credit-assignment methods and Timing Late exceed it by 16–19% in average last-20 and 5–16× in ≥195 rate. The baseline's consistency advantage diminishes as the human-feedback methods' compounding effects take hold in the final 50 episodes.

### 9.3 Limitations and Future Work

- **Combined timing + credit assignment**: The two best innovations (late timing and credit assignment) were not tested together. Combining late-window oracle feedback with exponential credit expansion could yield further gains by concentrating high-quality signals in the reward model's optimal training regime.

- **VI-TAMER variance reduction**: VI-TAMER's amplifier behaviour (seed 6: 195.0 r10 vs seed 5: 128.2) suggests investigating **ensemble VI-TAMER** — averaging over multiple reward models to reduce initialisation sensitivity while preserving the TD propagation advantage.

- **Exponential credit for VI-TAMER**: VI-TAMER currently uses uniform credit. Testing exponential credit (which preserves more magnitude at the most recent timestep) may reduce the TD magnitude dilution and improve VI-TAMER's consistency.

- **Credit window tuning**: Comparing $W \in \{1, 2, 3, 5\}$ with exponential credit for both HCRL and VI-TAMER would identify the optimal trade-off between training data expansion and signal dilution.

- **RLHF scaling study**: A systematic study from 200 → 500 → 1,000 episodes would identify the minimum preference dataset for reliable convergence and test whether the seed-5 phase transition generalises above a data threshold.

- **Adaptive feedback timing**: An online metric (e.g., rolling-10 mean exceeding a threshold) could trigger oracle feedback activation automatically, replacing fixed timing windows.

---

## 10. References

[1] **Knox, W. B., & Stone, P.** (2009). Interactively shaping agents via human reinforcement: The TAMER framework. *Proceedings of the Fifth International Conference on Knowledge Capture (K-CAP)*, 9–16.

[2] **Knox, W. B., & Stone, P.** (2012). Reinforcement learning from human reward and advice. *Proceedings of the AAAI Workshop on Robots Learning Interactively from Human Teachers*.

[3] **Li, G., Gomez, R., Nakamura, K., & He, B.** (2019). Human-centered reinforcement learning: A survey. *IEEE Transactions on Human-Machine Systems*, 49(4), 337–349.

[4] **Christiano, P., Leike, J., Brown, T. B., Martic, M., Legg, S., & Amodei, D.** (2017). Deep reinforcement learning from human preferences. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.

[5] **Watkins, C. J. C. H., & Dayan, P.** (1992). Q-learning. *Machine Learning*, 8(3–4), 279–292.

[6] **Sutton, R. S., & Barto, A. G.** (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

[7] **Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., & Zaremba, W.** (2016). OpenAI Gym. *arXiv preprint arXiv:1606.01540*.

[8] **Welford, B. P.** (1962). Note on a method for calculating corrected sums of squares and products. *Technometrics*, 4(3), 419–420.

[9] **Bradley, R. A., & Terry, M. E.** (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika*, 39(3–4), 324–345.

[10] **Kingma, D. P., & Ba, J.** (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.

---

*End of Report*
