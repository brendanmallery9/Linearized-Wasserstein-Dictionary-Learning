#!/usr/bin/env bash
#
# Topk2-only sweep for the MNIST convex-combinations experiment on digits {0,3,4}.
#
# This keeps the winning coefficient regularization from run_convex_combos_034:
#   activation_type=topk_simplex, topk_k=2, sparsity_coeff=0.0
#
# Default first-pass sweep:
#   LR  in {1e-4, 3e-4, 1e-3}
#   EPS in {0.0125, 0.025, 0.05}
#
# Override via env vars, e.g.:
#   EPOCHS=1000 LR_SWEEP="3e-4 1e-3" DEVICE=cuda \
#     ./mnist/scripts/run_convex_combos_034_topk2_sweep.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1

# --- I/O ---
DATA_DIR="${DATA_DIR:-datasets/convex_combos_034}"
MNIST_OT_DIR="${MNIST_OT_DIR:-datasets/mnist_ot}"
OUTPUT_ROOT="${OUTPUT_ROOT:-mnist/results/convex_combos_034_topk2_sweep}"
LOG_DIR="${LOG_DIR:-logs}"
DEVICE="${DEVICE:-auto}"
GPU_IDS="${GPU_IDS:-}"
SEED="${SEED:-42}"

# --- Data generation: matches run_convex_combos_034_experiment.sh ---
DIGITS="${DIGITS:-0 3 4}"
SUBSET_SIZE="${SUBSET_SIZE:-2}"
N_SAMPLES="${N_SAMPLES:-10000}"
DIGIT_SAMPLE_INDEX="${DIGIT_SAMPLE_INDEX:-1}"

# --- Fixed model choices ---
M="${M:-3}"
TOPK_K="${TOPK_K:-2}"
ACTIVATION_TYPE="${ACTIVATION_TYPE:-topk_simplex}"
C="${C:-0.0}"
METHOD="${METHOD:-displacement}"

# --- Sweep knobs ---
LR_SWEEP="${LR_SWEEP:-1e-4 3e-4 }"
EPS_SWEEP="${EPS_SWEEP:-0.0125 0.025 }"
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LISTA_STEPS="${LISTA_STEPS:-20}"
GRID_MODE="${GRID_MODE:-uniform}"
GRID_SUPPORT_SIZE="${GRID_SUPPORT_SIZE:-1000}"
GRID_N_CLOUDS="${GRID_N_CLOUDS:-10}"
GRID_SIDE="${GRID_SIDE:-32}"

# --- Optimizer knobs ---
TEST_FRACTION="${TEST_FRACTION:-0.1}"
OPTIMIZER="${OPTIMIZER:-adamw}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
SCHEDULER="${SCHEDULER:-cosine}"
LR_MIN="${LR_MIN:-0.0}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"

# --- Modes ---
FORCE_PREP="${FORCE_PREP:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-20000}"

RUNNER="pointcloud/pipeline/pointcloud_run_experiments.py"
EVAL_DIR="$OUTPUT_ROOT/eval"
MANIFEST="$OUTPUT_ROOT/sweep_manifest.csv"

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

slug() {
  printf '%s' "$1" | sed 's/[^A-Za-z0-9]/_/g'
}

run_complete() {
  local dir="$1"
  local last_ckpt
  local best_ckpt
  [[ -f "$dir/config.json" && -f "$dir/metrics.json" ]] || return 1
  last_ckpt="$(find "$dir" -maxdepth 1 -type f -name '*.pt' ! -name '*_best.pt' -print -quit)"
  best_ckpt="$(find "$dir" -maxdepth 1 -type f -name '*_best.pt' -print -quit)"
  [[ -n "$last_ckpt" && -n "$best_ckpt" ]]
}

echo "=== convex-combos {${DIGITS}} topk${TOPK_K} sweep ==="
echo "data_dir:    $DATA_DIR"
echo "output_root: $OUTPUT_ROOT"
echo "device:      $DEVICE"
if [[ -n "$GPU_IDS" ]]; then echo "gpu_ids:     $GPU_IDS"; fi
echo "fixed:       m=$M activation=$ACTIVATION_TYPE topk_k=$TOPK_K c=$C method=$METHOD"
echo "sweep:       LR=[$LR_SWEEP]  EPS=[$EPS_SWEEP]"
echo "training:    epochs=$EPOCHS batch=$BATCH_SIZE lista_steps=$LISTA_STEPS"
echo "stability:   grad_clip_norm=$GRAD_CLIP_NORM scheduler=$SCHEDULER lr_min=$LR_MIN"
echo "grid:        mode=$GRID_MODE support=$GRID_SUPPORT_SIZE n_clouds=$GRID_N_CLOUDS side=$GRID_SIDE"
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
    2>&1 | tee "$LOG_DIR/prepare_convex_combos_034_topk2_sweep.log"
fi
echo ""

printf 'lr,eps,output_dir,log_file\n' > "$MANIFEST"

# ------------------------------------------------------------
# 2. Train each topk2 run in its own subdirectory
# ------------------------------------------------------------
for lr in $LR_SWEEP; do
  for eps in $EPS_SWEEP; do
    lr_tag="$(slug "$lr")"
    eps_tag="$(slug "$eps")"
    run_dir="$OUTPUT_ROOT/lr${lr_tag}_eps${eps_tag}"
    log_file="$LOG_DIR/train_convex_combos_034_topk2_lr${lr_tag}_eps${eps_tag}.log"

    printf '%s,%s,%s,%s\n' "$lr" "$eps" "$run_dir" "$log_file" >> "$MANIFEST"

    if [[ "$FORCE_TRAIN" -eq 0 ]] && run_complete "$run_dir"; then
      echo "[train] skip existing lr=$lr eps=$eps -> $run_dir"
      continue
    fi

    echo "[train] lr=$lr eps=$eps -> $run_dir"
    train_args=(
      --data_dir "$DATA_DIR"
      --output_dir "$run_dir"
      --m "$M"
      --epochs "$EPOCHS"
      --batch_size "$BATCH_SIZE"
      --lr "$lr"
      --test_fraction "$TEST_FRACTION"
      --optimizer "$OPTIMIZER"
      --weight_decay "$WEIGHT_DECAY"
      --scheduler "$SCHEDULER"
      --lr_min "$LR_MIN"
      --grad_clip_norm "$GRAD_CLIP_NORM"
      --epsilons "$eps"
      --sparsity_coeffs "$C"
      --methods "$METHOD"
      --activation_type "$ACTIVATION_TYPE"
      --topk_k "$TOPK_K"
      --lista_steps "$LISTA_STEPS"
      --grid_mode "$GRID_MODE"
      --grid_side "$GRID_SIDE"
      --grid_support_size "$GRID_SUPPORT_SIZE"
      --grid_n_clouds "$GRID_N_CLOUDS"
      --device "$DEVICE"
      --seed "$SEED"
    )
    if [[ -n "$GPU_IDS" ]]; then
      train_args+=( --gpu_ids $GPU_IDS )
    fi

    "$PYTHON" -u "$RUNNER" "${train_args[@]}" 2>&1 | tee "$log_file"
    echo ""
  done
done

# ------------------------------------------------------------
# 3. Evaluate all sweep runs into one dashboard/CSV
# ------------------------------------------------------------
if [[ "$RUN_EVAL" -eq 1 ]]; then
  echo "[eval] -> $EVAL_DIR"
  "$PYTHON" -u mnist/analysis/evaluate_convex_combos.py \
    --results_dir "$OUTPUT_ROOT" \
    --data_dir "$DATA_DIR" \
    --output_dir "$EVAL_DIR" \
    --device "$DEVICE" \
    --max_samples "$EVAL_MAX_SAMPLES" \
    2>&1 | tee "$LOG_DIR/evaluate_convex_combos_034_topk2_sweep.log"
fi

echo ""
echo "Done."
echo "  manifest:  $MANIFEST"
echo "  dashboard: $EVAL_DIR/dashboard.png"
echo "  summary:   $EVAL_DIR/summary.csv"
