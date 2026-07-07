#!/usr/bin/env bash
#
# Convex-combinations-of-MNIST-OT-maps SAE pipeline.
#
# Pipeline:
#   1. Generate the convex-combinations dataset (prepare_convex_combos.py),
#      unless it already exists with matching parameters. The prep script
#      itself will fall back to running prepare_mnist_ot.py if the source
#      MNIST OT dataset is missing.
#   2. Train the displacement-field (or raw-map) SAE on the resulting maps
#      (pointcloud_run_experiments.py).
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"

# --- Data / I/O ---
DATA_DIR="${DATA_DIR:-datasets/convex_combos}"
MNIST_OT_DIR="${MNIST_OT_DIR:-datasets/mnist_ot}"
OUTPUT_DIR="${OUTPUT_DIR:-pointcloud/results/convex_combos}"
LOG_DIR="${LOG_DIR:-logs}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"

# --- Convex-combo prep params (must match metadata.json for skip-prep) ---
N_SAMPLES="${N_SAMPLES:-50000}"
SUBSET_SIZE="${SUBSET_SIZE:-3}"
DIGIT_SAMPLE_INDEX="${DIGIT_SAMPLE_INDEX:-0}"

# --- Model / training params ---
M="${M:-10}"
GRID_MODE="${GRID_MODE:-cloud_mixture}"                    # uniform | cloud_mixture
GRID_SIDE="${GRID_SIDE:-32}"                               # used iff GRID_MODE=uniform (32^2=1024 pts)
GRID_N_CLOUDS="${GRID_N_CLOUDS:-10}"                       # used iff GRID_MODE=cloud_mixture
GRID_SUPPORT_SIZE="${GRID_SUPPORT_SIZE:-1000}"             # used iff GRID_MODE=cloud_mixture
LISTA_STEPS="${LISTA_STEPS:-20}"
EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LR="${LR:-0.001}"
TEST_FRACTION="${TEST_FRACTION:-0.1}"
OPTIMIZER="${OPTIMIZER:-adamw}"                            # adamw | adam
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
SCHEDULER="${SCHEDULER:-none}"                             # none | cosine
LR_MIN="${LR_MIN:-0.0}"
EPS="${EPS:-0.025}"
C="${C:-0.00001}"
METHOD="${METHOD:-displacement}"                           # displacement | raw_map
ACTIVATION_TYPE="${ACTIVATION_TYPE:-relu}"                 # relu | jumprelu | topk | topk_simplex
TOPK_K="${TOPK_K:-3}"
GPU_IDS="${GPU_IDS:-}"

# --- Modes ---
PREP_MODE="${PREP_MODE:-auto}"                             # auto | skip | force
SKIP_TRAIN="${SKIP_TRAIN:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

usage() {
  cat <<'EOF'
Usage:
  pointcloud/scripts/run_convex_combos_pipeline.sh [options]

Pipeline:
  1. Generate convex-combinations dataset from MNIST OT maps (unless an
     up-to-date one already exists at DATA_DIR).
  2. Train a transport-map SAE on the resulting maps.

Prep modes:
  --prepare-data, --auto-data  Run prep only if DATA_DIR is incomplete (default).
  --skip-data                  Never run prep; require existing dataset.
  --force-data                 Always rerun prep before training.

Training modes:
  --skip-train                 Do not train.
  --force-train                Allow training when OUTPUT_DIR already has outputs.

Options:
  --data-dir PATH              Default: datasets/convex_combos
  --mnist-ot-dir PATH          Default: datasets/mnist_ot
  --output-dir PATH            Default: pointcloud/results/convex_combos
  --log-dir PATH               Default: logs
  --device auto|cuda|mps|cpu
  --gpu-ids "0 1 2"
  --seed N

  --n-samples N                Default: 10000
  --subset-size N              Distinct digits per mixture. Default: 3
  --digit-sample-index N       Which sample to take from each digit_<d>. Default: 0

  --m N                        Dictionary atoms. Default: 30
  --grid-mode uniform|cloud_mixture
  --grid-side N                Side length when --grid-mode uniform (dataset is 2D).
  --grid-n-clouds N
  --grid-support-size N
  --lista-steps N              Default: 20
  --epochs N                   Default: 2000
  --batch-size N               Default: 64
  --lr X                       Default: 0.001
  --test-fraction X            Default: 0.1
  --optimizer adamw|adam       Default: adamw
  --weight-decay X             Default: 0.1
  --scheduler none|cosine      Default: none
  --lr-min X                   Floor for cosine schedule. Default: 0.0
  --eps X                      Default: 0.025
  --c X                        Sparsity coefficient. Default: 0.0001
  --method displacement|raw_map
  --activation-type relu|jumprelu|topk|topk_simplex
                               Default: relu
  --topk-k N                   k for topk / topk_simplex. Default: 3

  -h, --help                   Show this help

Environment overrides use the uppercase option names, e.g.:
  DEVICE=cuda EPOCHS=500 ./wdl_repo/mnist/scripts/run_convex_combos_pipeline.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prepare-data|--auto-data) PREP_MODE="auto"; shift ;;
    --skip-data)                PREP_MODE="skip"; shift ;;
    --force-data)               PREP_MODE="force"; shift ;;
    --skip-train)               SKIP_TRAIN=1; shift ;;
    --force-train)              FORCE_TRAIN=1; shift ;;
    --data-dir)                 DATA_DIR="$2"; shift 2 ;;
    --mnist-ot-dir)             MNIST_OT_DIR="$2"; shift 2 ;;
    --output-dir)               OUTPUT_DIR="$2"; shift 2 ;;
    --log-dir)                  LOG_DIR="$2"; shift 2 ;;
    --device)                   DEVICE="$2"; shift 2 ;;
    --gpu-ids)                  GPU_IDS="$2"; shift 2 ;;
    --seed)                     SEED="$2"; shift 2 ;;
    --n-samples)                N_SAMPLES="$2"; shift 2 ;;
    --subset-size)              SUBSET_SIZE="$2"; shift 2 ;;
    --digit-sample-index)       DIGIT_SAMPLE_INDEX="$2"; shift 2 ;;
    --m)                        M="$2"; shift 2 ;;
    --grid-mode)                GRID_MODE="$2"; shift 2 ;;
    --grid-side)                GRID_SIDE="$2"; shift 2 ;;
    --grid-n-clouds)            GRID_N_CLOUDS="$2"; shift 2 ;;
    --grid-support-size)        GRID_SUPPORT_SIZE="$2"; shift 2 ;;
    --lista-steps)              LISTA_STEPS="$2"; shift 2 ;;
    --epochs)                   EPOCHS="$2"; shift 2 ;;
    --batch-size)               BATCH_SIZE="$2"; shift 2 ;;
    --lr)                       LR="$2"; shift 2 ;;
    --test-fraction)            TEST_FRACTION="$2"; shift 2 ;;
    --optimizer)                OPTIMIZER="$2"; shift 2 ;;
    --weight-decay)             WEIGHT_DECAY="$2"; shift 2 ;;
    --scheduler)                SCHEDULER="$2"; shift 2 ;;
    --lr-min)                   LR_MIN="$2"; shift 2 ;;
    --eps)                      EPS="$2"; shift 2 ;;
    --c)                        C="$2"; shift 2 ;;
    --method)                   METHOD="$2"; shift 2 ;;
    --activation-type)          ACTIVATION_TYPE="$2"; shift 2 ;;
    --topk-k)                   TOPK_K="$2"; shift 2 ;;
    -h|--help)                  usage; exit 0 ;;
    *)                          echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$PREP_MODE" != "auto" && "$PREP_MODE" != "skip" && "$PREP_MODE" != "force" ]]; then
  echo "ERROR: PREP_MODE must be one of: auto, skip, force. Got: $PREP_MODE" >&2
  exit 2
fi
if [[ "$GRID_MODE" != "uniform" && "$GRID_MODE" != "cloud_mixture" ]]; then
  echo "ERROR: --grid-mode must be one of: uniform, cloud_mixture. Got: $GRID_MODE" >&2
  exit 2
fi
if [[ "$OPTIMIZER" != "adamw" && "$OPTIMIZER" != "adam" ]]; then
  echo "ERROR: --optimizer must be one of: adamw, adam. Got: $OPTIMIZER" >&2
  exit 2
fi
if [[ "$SCHEDULER" != "none" && "$SCHEDULER" != "cosine" ]]; then
  echo "ERROR: --scheduler must be one of: none, cosine. Got: $SCHEDULER" >&2
  exit 2
fi
if [[ "$METHOD" != "displacement" && "$METHOD" != "raw_map" ]]; then
  echo "ERROR: --method must be one of: displacement, raw_map. Got: $METHOD" >&2
  exit 2
fi
case "$ACTIVATION_TYPE" in
  relu|jumprelu|topk|topk_simplex) ;;
  *) echo "ERROR: --activation-type must be relu|jumprelu|topk|topk_simplex. Got: $ACTIVATION_TYPE" >&2; exit 2 ;;
esac


# ============================================================
# Inline Python: check whether DATA_DIR already has a matching dataset.
# ============================================================
data_complete() {
  "$PYTHON" - "$DATA_DIR" "$N_SAMPLES" "$SUBSET_SIZE" \
              "$DIGIT_SAMPLE_INDEX" "$SEED" <<'PY'
import json, sys
from pathlib import Path
import torch

(data_dir, n_samples, subset_size, digit_sample_index, seed) = sys.argv[1:6]
data_dir = Path(data_dir)

meta_path = data_dir / "metadata.json"
base_path = data_dir / "base_measure.pt"
maps_path = data_dir / "maps" / "maps.pt"
coeffs_path = data_dir / "coefficients" / "coefficients.pt"
for p in (meta_path, base_path, maps_path, coeffs_path):
    if not p.exists():
        sys.exit(1)

try:
    meta = json.loads(meta_path.read_text())
except Exception:
    sys.exit(1)

expected = {
    "n_samples":          int(n_samples),
    "subset_size":        int(subset_size),
    "digit_sample_index": int(digit_sample_index),
    "seed":               int(seed),
}
for k, v in expected.items():
    if meta.get(k) != v:
        sys.exit(1)

try:
    maps = torch.load(maps_path, map_location="cpu")
    coeffs = torch.load(coeffs_path, map_location="cpu")
except Exception:
    sys.exit(1)

if maps.shape[0] != int(n_samples):
    sys.exit(1)
if coeffs.shape[0] != int(n_samples):
    sys.exit(1)

sys.exit(0)
PY
}


train_outputs_exist() {
  [[ -d "$OUTPUT_DIR" ]] || return 1
  [[ -f "$OUTPUT_DIR/metrics.json" || -f "$OUTPUT_DIR/config.json" ]] && return 0
  find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | grep -q .
}


run_prep() {
  echo "[data] generating convex-combinations dataset"
  echo "  data_dir:           $DATA_DIR"
  echo "  mnist_ot_dir:       $MNIST_OT_DIR"
  echo "  n_samples:          $N_SAMPLES"
  echo "  subset_size:        $SUBSET_SIZE"
  echo "  digit_sample_index: $DIGIT_SAMPLE_INDEX"
  echo "  seed:               $SEED"

  prep_args=(
    --output_dir "$DATA_DIR"
    --mnist_ot_dir "$MNIST_OT_DIR"
    --n_samples "$N_SAMPLES"
    --subset_size "$SUBSET_SIZE"
    --digit_sample_index "$DIGIT_SAMPLE_INDEX"
    --seed "$SEED"
    --python "$PYTHON"
  )

  log_tag="$(basename "$DATA_DIR")"
  "$PYTHON" -u mnist/pipeline/prepare_convex_combos.py "${prep_args[@]}" \
    2>&1 | tee "$LOG_DIR/prepare_convex_combos_${log_tag}.log"
}


run_training() {
  if [[ "$SKIP_TRAIN" -eq 1 ]]; then
    echo "[train] skipped"
    return
  fi
  if [[ "$FORCE_TRAIN" -eq 0 ]] && train_outputs_exist; then
    echo "ERROR: OUTPUT_DIR already has training outputs: $OUTPUT_DIR" >&2
    echo "Use --force-train to overwrite, or pick another --output-dir." >&2
    exit 1
  fi

  echo "[train] convex-combos SAE"
  echo "  data_dir:        $DATA_DIR"
  echo "  output_dir:      $OUTPUT_DIR"
  echo "  device:          $DEVICE"
  if [[ -n "$GPU_IDS" ]]; then echo "  gpu_ids:         $GPU_IDS"; fi
  echo "  method:          $METHOD"
  echo "  m:               $M"
  echo "  grid_mode:       $GRID_MODE"
  echo "  grid_side:       $GRID_SIDE"
  echo "  grid_n_clouds:   $GRID_N_CLOUDS"
  echo "  grid_supp_size:  $GRID_SUPPORT_SIZE"
  echo "  lista_steps:     $LISTA_STEPS"
  echo "  epochs:          $EPOCHS"
  echo "  batch_size:      $BATCH_SIZE"
  echo "  lr:              $LR"
  echo "  test_fraction:   $TEST_FRACTION"
  echo "  optimizer:       $OPTIMIZER"
  echo "  weight_decay:    $WEIGHT_DECAY"
  echo "  scheduler:       $SCHEDULER"
  echo "  lr_min:          $LR_MIN"
  echo "  eps:             $EPS"
  echo "  c:               $C"
  echo "  activation_type: $ACTIVATION_TYPE"
  if [[ "$ACTIVATION_TYPE" == "topk" || "$ACTIVATION_TYPE" == "topk_simplex" ]]; then
    echo "  topk_k:          $TOPK_K"
  fi

  train_args=(
    --data_dir "$DATA_DIR"
    --output_dir "$OUTPUT_DIR"
    --m "$M"
    --grid_mode "$GRID_MODE"
    --grid_side "$GRID_SIDE"
    --grid_n_clouds "$GRID_N_CLOUDS"
    --grid_support_size "$GRID_SUPPORT_SIZE"
    --lista_steps "$LISTA_STEPS"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --lr "$LR"
    --test_fraction "$TEST_FRACTION"
    --optimizer "$OPTIMIZER"
    --weight_decay "$WEIGHT_DECAY"
    --scheduler "$SCHEDULER"
    --lr_min "$LR_MIN"
    --epsilons "$EPS"
    --sparsity_coeffs "$C"
    --methods "$METHOD"
    --activation_type "$ACTIVATION_TYPE"
    --topk_k "$TOPK_K"
    --device "$DEVICE"
    --seed "$SEED"
  )
  if [[ -n "$GPU_IDS" ]]; then
    train_args+=( --gpu_ids $GPU_IDS )
  fi

  log_tag="$(basename "$OUTPUT_DIR")"
  "$PYTHON" -u pointcloud/pipeline/pointcloud_run_experiments.py "${train_args[@]}" \
    2>&1 | tee "$LOG_DIR/train_convex_combos_SAE_${log_tag}.log"
}


mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1

echo "=== convex-combos SAE pipeline ==="
echo "data_dir:     $DATA_DIR"
echo "mnist_ot_dir: $MNIST_OT_DIR"
echo "output_dir:   $OUTPUT_DIR"
echo "device:       $DEVICE"
echo "prep_mode:    $PREP_MODE"
echo ""

case "$PREP_MODE" in
  auto)
    if data_complete; then
      echo "[data] existing matching dataset found; skipping prep"
      echo "  data_dir: $DATA_DIR"
    else
      echo "[data] missing, incomplete, or stale; running prep"
      run_prep
    fi
    ;;
  skip)
    echo "[data] skipped"
    echo "  data_dir: $DATA_DIR"
    if ! data_complete; then
      echo "ERROR: DATA_DIR is missing, incomplete, or does not match the requested params." >&2
      echo "Run without --skip-data, or use --force-data to rebuild." >&2
      exit 1
    fi
    ;;
  force)
    run_prep
    ;;
esac

run_training

echo ""
echo "Done."
echo "Checkpoint and metrics: $OUTPUT_DIR"
