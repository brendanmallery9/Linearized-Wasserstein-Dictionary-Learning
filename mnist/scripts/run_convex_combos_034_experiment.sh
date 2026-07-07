#!/usr/bin/env bash
#
# Convex-combinations experiment on MNIST OT maps for digits {0, 3, 4}.
#
# Each data point is a 2-sparse (Dirichlet) mixture of the 3 generating digit
# maps, so the true mixing weights live on a 2-simplex. We learn a 3-atom
# dictionary ("no_heads = 3") and compare two sparsity regimes:
#
#   1. Soft L1 sweep   : activation=relu, L1 in {0.01, 0.005, 0.001, 0.0005}
#   2. Hard 2-sparse   : activation=topk_simplex, k=2 (codes renormalized onto
#                        the simplex; the L1 term is then a no-op, so c=0)
#
# Pipeline:  prep dataset  ->  train L1 sweep  ->  train topk2  ->  evaluate.
# Everything lands under one results tree, and the eval step produces a single
# combined dashboard (atoms + code simplex + MSE) rather than many images.
#
# Override any setting via env vars, e.g.:
#   EPOCHS=100 DEVICE=cpu ./mnist/scripts/run_convex_combos_034_experiment.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # .../wdl_repo
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1

# --- I/O ---
DATA_DIR="${DATA_DIR:-datasets/convex_combos_034}"
MNIST_OT_DIR="${MNIST_OT_DIR:-datasets/mnist_ot}"
OUTPUT_ROOT="${OUTPUT_ROOT:-mnist/results/convex_combos_034}"
LOG_DIR="${LOG_DIR:-logs}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"

# --- Data generation ---
DIGITS="${DIGITS:-0 3 4}"            # generating digits -> 2-simplex
SUBSET_SIZE="${SUBSET_SIZE:-2}"      # 2-sparse combinations
N_SAMPLES="${N_SAMPLES:-10000}"
DIGIT_SAMPLE_INDEX="${DIGIT_SAMPLE_INDEX:-1}"

# --- Model / training (per request) ---
M="${M:-3}"                          # number of dictionary atoms ("no_heads")
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-4}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
L1_SWEEP="${L1_SWEEP:-0.01 0.005 0.001 0.0005}"
TOPK_K="${TOPK_K:-2}"                # enforced sparsity for the topk_simplex run
EPS="${EPS:-0.025}"
LISTA_STEPS="${LISTA_STEPS:-20}"
GRID_MODE="${GRID_MODE:-cloud_mixture}"
GRID_SUPPORT_SIZE="${GRID_SUPPORT_SIZE:-1000}"
GRID_N_CLOUDS="${GRID_N_CLOUDS:-10}"

# --- Modes ---
FORCE_PREP="${FORCE_PREP:-0}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-20000}"

RUNNER="pointcloud/pipeline/pointcloud_run_experiments.py"
L1_DIR="$OUTPUT_ROOT/l1_sweep"
TOPK_DIR="$OUTPUT_ROOT/topk${TOPK_K}"
EVAL_DIR="$OUTPUT_ROOT/eval"

mkdir -p "$LOG_DIR"

echo "=== convex-combos {$DIGITS} ${SUBSET_SIZE}-sparse experiment ==="
echo "data_dir:    $DATA_DIR"
echo "output_root: $OUTPUT_ROOT"
echo "device:      $DEVICE   m=$M  epochs=$EPOCHS  batch=$BATCH_SIZE  lr=$LR"
echo "grad clip:   $GRAD_CLIP_NORM"
echo "L1 sweep:    $L1_SWEEP        topk_simplex k=$TOPK_K"
echo ""

# ------------------------------------------------------------
# 1. Generate the dataset (skip if present unless FORCE_PREP=1)
# ------------------------------------------------------------
if [[ "$FORCE_PREP" -eq 0 && -f "$DATA_DIR/maps/maps.pt" && -f "$DATA_DIR/base_maps.pt" ]]; then
  echo "[data] found existing dataset at $DATA_DIR (set FORCE_PREP=1 to rebuild)"
else
  echo "[data] generating $N_SAMPLES samples"
  "$PYTHON" -u mnist/pipeline/prepare_convex_combos.py \
    --output_dir "$DATA_DIR" \
    --mnist_ot_dir "$MNIST_OT_DIR" \
    --digits $DIGITS \
    --subset_size "$SUBSET_SIZE" \
    --n_samples "$N_SAMPLES" \
    --digit_sample_index "$DIGIT_SAMPLE_INDEX" \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/prepare_convex_combos_034.log"
fi
echo ""

# ------------------------------------------------------------
# Shared runner args
# ------------------------------------------------------------
common_args=(
  --data_dir "$DATA_DIR"
  --m "$M"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --grad_clip_norm "$GRAD_CLIP_NORM"
  --epsilons "$EPS"
  --lista_steps "$LISTA_STEPS"
  --grid_mode "$GRID_MODE"
  --grid_support_size "$GRID_SUPPORT_SIZE"
  --grid_n_clouds "$GRID_N_CLOUDS"
  --methods displacement
  --device "$DEVICE"
  --seed "$SEED"
)

# ------------------------------------------------------------
# 2. L1 sweep (soft sparsity, relu)
# ------------------------------------------------------------
echo "[train] L1 sweep (relu) -> $L1_DIR"
"$PYTHON" -u "$RUNNER" \
  "${common_args[@]}" \
  --output_dir "$L1_DIR" \
  --activation_type relu \
  --sparsity_coeffs $L1_SWEEP \
  2>&1 | tee "$LOG_DIR/train_convex_combos_034_l1_sweep.log"
echo ""

# ------------------------------------------------------------
# 3. Enforced 2-sparse (topk_simplex, k=TOPK_K, L1 no-op so c=0)
# ------------------------------------------------------------
echo "[train] enforced ${TOPK_K}-sparse (topk_simplex) -> $TOPK_DIR"
"$PYTHON" -u "$RUNNER" \
  "${common_args[@]}" \
  --output_dir "$TOPK_DIR" \
  --activation_type topk_simplex \
  --topk_k "$TOPK_K" \
  --sparsity_coeffs 0.0 \
  2>&1 | tee "$LOG_DIR/train_convex_combos_034_topk${TOPK_K}.log"
echo ""

# ------------------------------------------------------------
# 4. Evaluate everything into one dashboard
# ------------------------------------------------------------
echo "[eval] -> $EVAL_DIR"
"$PYTHON" -u mnist/analysis/evaluate_convex_combos.py \
  --results_dir "$OUTPUT_ROOT" \
  --data_dir "$DATA_DIR" \
  --output_dir "$EVAL_DIR" \
  --device "$DEVICE" \
  --max_samples "$EVAL_MAX_SAMPLES" \
  2>&1 | tee "$LOG_DIR/evaluate_convex_combos_034.log"

echo ""
echo "Done."
echo "  dashboard:      $EVAL_DIR/dashboard.png"
echo "  mse comparison: $EVAL_DIR/mse_comparison.png"
echo "  summary:        $EVAL_DIR/summary.csv"
