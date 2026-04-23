#!/usr/bin/env bash
# run_all.sh — Train every model for 200 episodes across all seeds and feedback weights.
#
# Coverage:
#   Seeds          : 0, 1, 2
#   Feedback weights (HCRL/VI-TAMER): 5, 10, 20, 50
#   Credit variant  : uniform cw=3
#   Models         : Baseline Q-Learning, HCRL, VI-TAMER, RLHF, RLHF Ensemble
#   Timing exp     : 4 conditions × 3 seeds (internal), per feedback weight
#
# Usage:
#   bash run_all.sh

set -e

EPISODES=200
SEEDS="5 6 9"
BSEEDS="0 1 2 3 4 5 6 7 8 9"  # --- IGNORE --- (for timing experiment)
FEEDBACK_WEIGHTS="5 20 50"

echo "========================================================"
echo " Full training run"
echo " episodes=${EPISODES}  seeds=[${SEEDS}]  fw=[${FEEDBACK_WEIGHTS}]"
echo "========================================================"

# ---------------------------------------------------------------------------
# 1. Baseline Q-Learning  (no feedback weight — one run per seed)
# ---------------------------------------------------------------------------
# echo ""
# echo "================================================================"
# echo "  BASELINE Q-LEARNING"
# echo "================================================================"
# for SEED in $BSEEDS; do
#     echo ""
#     echo "  [Baseline] seed=${SEED}"
#     uv run python run.py --episodes $EPISODES --seed $SEED
# done

# ---------------------------------------------------------------------------
# 2. HCRL (TAMER) — all feedback weights × all seeds
# ---------------------------------------------------------------------------
# echo ""
# echo "================================================================"
# echo "  HCRL (TAMER)"
# echo "================================================================"
# for SEED in $SEEDS; do
#     echo ""
#     echo "  [HCRL] fw=${FW}  seed=${SEED}  credit=exp cw=3"
#     uv run python train_hcrl.py --episodes $EPISODES --seed $SEED \
#         --feedback-weight 10 --credit-window 3 --credit-fn exp --skip-charts
# done

# ---------------------------------------------------------------------------
# 3. VI-TAMER — all feedback weights × all seeds
# ---------------------------------------------------------------------------
# echo ""
# echo "================================================================"
# echo "  VI-TAMER"
# echo "================================================================"
for FW in $FEEDBACK_WEIGHTS; do
    echo ""
    echo "  [VI-TAMER] fw=${FW}  seed=${SEED}  credit=exp cw=3"
    uv run python train_hcrl.py --episodes $EPISODES --seed 5 \
        --feedback-weight $FW --credit-window 3 --credit-fn exp --skip-charts
done

# # ---------------------------------------------------------------------------
# # 4. RLHF — no feedback weight, one run per seed
# # ---------------------------------------------------------------------------
# echo ""
# echo "================================================================"
# echo "  RLHF (standard)"
# echo "================================================================"
# for SEED in $SEEDS; do
#     echo ""
#     echo "  [RLHF] seed=${SEED}"
#     uv run python train_rlhf.py --episodes $EPISODES --seed $SEED --skip-charts
# done

# # ---------------------------------------------------------------------------
# # 5. RLHF Ensemble — no feedback weight, one run per seed × ensemble size
# # ---------------------------------------------------------------------------
# echo ""
# echo "================================================================"
# echo "  RLHF Ensemble"
# echo "================================================================"
# for SEED in $SEEDS; do
#     echo ""
#     echo "  [RLHF Ensemble n=3] seed=${SEED}"
#     uv run python train_rlhf_ensemble.py --episodes $EPISODES --seed $SEED --n-models 3 --skip-charts

#     echo ""
#     echo "  [RLHF Ensemble n=5] seed=${SEED}"
#     uv run python train_rlhf_ensemble.py --episodes $EPISODES --seed $SEED --n-models 5 --skip-charts
# done

# ---------------------------------------------------------------------------
# 6. Feedback Timing Experiment — per feedback weight (seeds run internally)
# ---------------------------------------------------------------------------
# echo ""
# echo "================================================================"
# echo "  FEEDBACK TIMING EXPERIMENT"
# echo "================================================================"
# uv run python feedback_timing_experiment.py \
#     --episodes $EPISODES --auto --skip-charts --feedback-weight 10

# echo ""
# echo "========================================================"
# echo " All done. Results in experiment-results/ep${EPISODES}/"
# echo "========================================================"
