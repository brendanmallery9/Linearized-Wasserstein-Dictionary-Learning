#!/usr/bin/env bash
#
# Reproduce the kept MNIST convex-combos 02378 regular-L1 run.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1

DATA_DIR="${DATA_DIR:-datasets/convex_combos_02378}"
MNIST_OT_DIR="${MNIST_OT_DIR:-datasets/mnist_ot}"
OUTPUT_ROOT="${OUTPUT_ROOT:-mnist/results/convex_combos_02378_best}"
RUN_DIR="${RUN_DIR:-$OUTPUT_ROOT/best_l1_1e-4}"
LOG_DIR="${LOG_DIR:-logs/convex_combos_02378_best}"
EVAL_DIR="${EVAL_DIR:-$OUTPUT_ROOT/eval}"
MANIFEST="$OUTPUT_ROOT/sweep_manifest.csv"

DIGITS="${DIGITS:-0 2 3 7 8}"
SUBSET_SIZE="${SUBSET_SIZE:-2}"
N_SAMPLES="${N_SAMPLES:-60000}"
DIGIT_SAMPLE_INDEX="${DIGIT_SAMPLE_INDEX:-1}"
PURE_DIGIT_REPEATS="${PURE_DIGIT_REPEATS:-0}"

M="${M:-5}"
METHOD="${METHOD:-displacement_centered}"
DISPLACEMENT_CENTER_MODE="${DISPLACEMENT_CENTER_MODE:-data_mean}"
ACTIVATION_TYPE="${ACTIVATION_TYPE:-relu}"
TOPK_K="${TOPK_K:-2}"
LISTA_STEPS="${LISTA_STEPS:-20}"

LR="${LR:-1.5e-4}"
EPS="${EPS:-0.025}"
SPARSITY_COEFF="${SPARSITY_COEFF:-1e-4}"
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
TEST_FRACTION="${TEST_FRACTION:-0.1}"
OPTIMIZER="${OPTIMIZER:-adamw}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
SCHEDULER="${SCHEDULER:-cosine}"
LR_MIN="${LR_MIN:-5e-6}"
PLATEAU_FACTOR="${PLATEAU_FACTOR:-0.5}"
PLATEAU_PATIENCE="${PLATEAU_PATIENCE:-25}"
PLATEAU_THRESHOLD="${PLATEAU_THRESHOLD:-1e-5}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"

GRID_MODE="${GRID_MODE:-uniform}"
GRID_SIDE="${GRID_SIDE:-64}"
GRID_SUPPORT_SIZE="${GRID_SUPPORT_SIZE:-1000}"
GRID_N_CLOUDS="${GRID_N_CLOUDS:-10}"

DEVICE="${DEVICE:-auto}"
GPU_IDS="${GPU_IDS:-}"
SEED="${SEED:-42}"
FORCE_PREP="${FORCE_PREP:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-20000}"

RUNNER="pointcloud/pipeline/pointcloud_run_experiments.py"

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

run_complete() {
  [[ -f "$RUN_DIR/config.json" &&
     -f "$RUN_DIR/metrics.json" &&
     -f "$RUN_DIR/checkpoint_last.pt" &&
     -f "$RUN_DIR/checkpoint_best.pt" ]]
}

normalize_checkpoint_names() {
  local last_ckpt
  local best_ckpt

  if [[ ! -f "$RUN_DIR/checkpoint_last.pt" ]]; then
    last_ckpt="$(find "$RUN_DIR" -maxdepth 1 -type f -name '*.pt' ! -name '*_best.pt' -print -quit)"
    [[ -z "$last_ckpt" ]] || mv "$last_ckpt" "$RUN_DIR/checkpoint_last.pt"
  fi

  if [[ ! -f "$RUN_DIR/checkpoint_best.pt" ]]; then
    best_ckpt="$(find "$RUN_DIR" -maxdepth 1 -type f -name '*_best.pt' -print -quit)"
    [[ -z "$best_ckpt" ]] || mv "$best_ckpt" "$RUN_DIR/checkpoint_best.pt"
  fi
}

echo "=== convex-combos 02378 best run ==="
echo "data_dir: $DATA_DIR"
echo "run_dir:  $RUN_DIR"
echo "logs:     $LOG_DIR"
echo "digits:   $DIGITS"
echo "model:    m=$M method=$METHOD activation=$ACTIVATION_TYPE lista_steps=$LISTA_STEPS"
echo "train:    lr=$LR lr_min=$LR_MIN eps=$EPS l1=$SPARSITY_COEFF epochs=$EPOCHS batch=$BATCH_SIZE"
echo ""

if [[ "$FORCE_PREP" -eq 0 && -f "$DATA_DIR/maps/maps.pt" && -f "$DATA_DIR/base_maps.pt" ]]; then
  echo "[data] found existing dataset at $DATA_DIR"
else
  echo "[data] generating $N_SAMPLES samples"
  "$PYTHON" -u mnist/pipeline/prepare_convex_combos.py \
    --output_dir "$DATA_DIR" \
    --mnist_ot_dir "$MNIST_OT_DIR" \
    --digits $DIGITS \
    --subset_size "$SUBSET_SIZE" \
    --n_samples "$N_SAMPLES" \
    --pure_digit_repeats "$PURE_DIGIT_REPEATS" \
    --digit_sample_index "$DIGIT_SAMPLE_INDEX" \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/prepare.log"
fi
echo ""

printf 'c,output_dir,log_file\n%s,%s,%s\n' "$SPARSITY_COEFF" "$RUN_DIR" "$LOG_DIR/train.log" > "$MANIFEST"

if [[ "$FORCE_TRAIN" -eq 0 ]] && run_complete; then
  echo "[train] found existing run at $RUN_DIR"
else
  echo "[train] -> $RUN_DIR"
  train_args=(
    --data_dir "$DATA_DIR"
    --output_dir "$RUN_DIR"
    --m "$M"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --lr "$LR"
    --test_fraction "$TEST_FRACTION"
    --optimizer "$OPTIMIZER"
    --weight_decay "$WEIGHT_DECAY"
    --scheduler "$SCHEDULER"
    --lr_min "$LR_MIN"
    --plateau_factor "$PLATEAU_FACTOR"
    --plateau_patience "$PLATEAU_PATIENCE"
    --plateau_threshold "$PLATEAU_THRESHOLD"
    --grad_clip_norm "$GRAD_CLIP_NORM"
    --epsilons "$EPS"
    --sparsity_coeffs "$SPARSITY_COEFF"
    --methods "$METHOD"
    --displacement_center_mode "$DISPLACEMENT_CENTER_MODE"
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

  "$PYTHON" -u "$RUNNER" "${train_args[@]}" 2>&1 | tee "$LOG_DIR/train.log"
  normalize_checkpoint_names
fi
echo ""

if [[ "$RUN_EVAL" -eq 1 ]]; then
  echo "[eval] -> $EVAL_DIR"
  "$PYTHON" -u mnist/analysis/evaluate_convex_combos.py \
    --results_dir "$OUTPUT_ROOT" \
    --data_dir "$DATA_DIR" \
    --output_dir "$EVAL_DIR" \
    --device "$DEVICE" \
    --max_samples "$EVAL_MAX_SAMPLES" \
    2>&1 | tee "$LOG_DIR/evaluate.log"
fi

echo ""
echo "Done."
echo "  manifest: $MANIFEST"
echo "  eval:     $EVAL_DIR"
