#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-datasets/mnist_ot}"
OUTPUT_DIR="${OUTPUT_DIR:-datasets/mnist_ot/SAE_params}"
LOG_DIR="${LOG_DIR:-logs}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
MAP_MODE="${MAP_MODE:-auto}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
RUN_EVAL="${RUN_EVAL:-0}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-}"

BASE_SUPP_SIZE="${BASE_SUPP_SIZE:-400}"
MAX_PER_DIGIT="${MAX_PER_DIGIT:-2000}"
BASE_MODE="${BASE_MODE:-uniform}"
BASE_DIGIT="${BASE_DIGIT:-0}"
BASE_NOISE_STD="${BASE_NOISE_STD:-0.02}"
BASE_N_COMPONENTS="${BASE_N_COMPONENTS:-100}"

M="${M:-30}"
LISTA_STEPS="${LISTA_STEPS:-20}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EPS="${EPS:-0.025}"
C="${C:-0.0001}"

usage() {
  cat <<'EOF'
Usage:
  mnist/scripts/run_mnist_sae_pipeline.sh [options]

Pipeline:
  1. Prepare MNIST OT maps, unless complete matching maps already exist.
  2. Train the MNIST displacement-field SAE.
  3. Optionally evaluate and render checkpoint visualizations.

Map-prep modes:
  --prepare-maps, --auto-maps  Prepare maps only if DATA_DIR is incomplete. Default.
  --skip-maps                  Never run prepare_mnist_ot.py; require existing maps.
  --force-maps                 Always rerun prepare_mnist_ot.py before training.

Training / eval modes:
  --skip-train                 Do not train; useful with --eval.
  --force-train                Allow training when OUTPUT_DIR already has outputs.
  --eval                       Run mnist/analysis/evaluate_mnist_sae.py after training.

Options:
  --data-dir PATH              Default: datasets/mnist_ot
  --output-dir PATH            Default: datasets/mnist_ot/SAE_params
  --log-dir PATH               Default: logs
  --device auto|cuda|mps|cpu   Default: auto
  --seed N                     Default: 42
  --progress-every N           Default: 10
  --base-supp-size N           Default: 400
  --max-per-digit N            Default: 2000
  --base-mode MODE             uniform, digit, or mixture. Default: uniform
  --base-digit N               Default: 0
  --base-noise-std X           Default: 0.02
  --base-n-components N        Default: 100
  --m N                        Dictionary atoms. Default: 30
  --lista-steps N              Default: 20
  --epochs N                   Default: 100
  --batch-size N               Default: 128
  --eps X                      Gibbs epsilon. Default: 0.025
  --c X                        Sparsity coefficient. Default: 0.0001
  --eval-output-dir PATH       Default: OUTPUT_DIR/eval
  -h, --help                   Show this help

Environment overrides use the uppercase option names, e.g.:
  DEVICE=cuda PROGRESS_EVERY=10 ./mnist/scripts/run_mnist_sae_pipeline.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prepare-maps|--auto-maps)
      MAP_MODE="auto"
      shift
      ;;
    --skip-maps)
      MAP_MODE="skip"
      shift
      ;;
    --force-maps)
      MAP_MODE="force"
      shift
      ;;
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    --force-train)
      FORCE_TRAIN=1
      shift
      ;;
    --eval)
      RUN_EVAL=1
      shift
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --progress-every)
      PROGRESS_EVERY="$2"
      shift 2
      ;;
    --base-supp-size)
      BASE_SUPP_SIZE="$2"
      shift 2
      ;;
    --max-per-digit)
      MAX_PER_DIGIT="$2"
      shift 2
      ;;
    --base-mode)
      BASE_MODE="$2"
      shift 2
      ;;
    --base-digit)
      BASE_DIGIT="$2"
      shift 2
      ;;
    --base-noise-std)
      BASE_NOISE_STD="$2"
      shift 2
      ;;
    --base-n-components)
      BASE_N_COMPONENTS="$2"
      shift 2
      ;;
    --m)
      M="$2"
      shift 2
      ;;
    --lista-steps)
      LISTA_STEPS="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --eps)
      EPS="$2"
      shift 2
      ;;
    --c)
      C="$2"
      shift 2
      ;;
    --eval-output-dir)
      EVAL_OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$EVAL_OUTPUT_DIR" ]]; then
  EVAL_OUTPUT_DIR="$OUTPUT_DIR/eval"
fi

if [[ "$MAP_MODE" != "auto" && "$MAP_MODE" != "skip" && "$MAP_MODE" != "force" ]]; then
  echo "ERROR: MAP_MODE must be one of: auto, skip, force. Got: $MAP_MODE" >&2
  exit 2
fi

if [[ "$BASE_MODE" != "uniform" && "$BASE_MODE" != "digit" && "$BASE_MODE" != "mixture" ]]; then
  echo "ERROR: --base-mode must be one of: uniform, digit, mixture. Got: $BASE_MODE" >&2
  exit 2
fi

maps_complete() {
  "$PYTHON" - "$DATA_DIR" "$BASE_SUPP_SIZE" "$MAX_PER_DIGIT" "$SEED" \
    "$BASE_MODE" "$BASE_DIGIT" "$BASE_NOISE_STD" "$BASE_N_COMPONENTS" <<'PY'
import json
import math
import sys
from pathlib import Path

import torch

data_dir = Path(sys.argv[1])
base_supp_size = int(sys.argv[2])
max_per_digit = int(sys.argv[3])
seed = int(sys.argv[4])
base_mode = sys.argv[5]
base_digit = int(sys.argv[6])
base_noise_std = float(sys.argv[7])
base_n_components = int(sys.argv[8])

metadata_path = data_dir / "metadata.json"
base_path = data_dir / "base_measure.pt"
if not metadata_path.exists() or not base_path.exists():
    sys.exit(1)

try:
    metadata = json.loads(metadata_path.read_text())
    params = metadata["parameters"]
except Exception:
    sys.exit(1)

expected = {
    "base_supp_size": base_supp_size,
    "max_per_digit": max_per_digit,
    "seed": seed,
    "base_mode": base_mode,
    "base_digit": base_digit,
    "base_n_components": base_n_components,
}
for key, value in expected.items():
    if params.get(key) != value:
        sys.exit(1)
try:
    observed_noise_std = float(params.get("base_noise_std"))
except (TypeError, ValueError):
    sys.exit(1)
if not math.isclose(observed_noise_std, base_noise_std, rel_tol=0.0, abs_tol=1e-12):
    sys.exit(1)

try:
    base = torch.load(base_path, map_location="cpu")
except Exception:
    sys.exit(1)
if tuple(base.shape) != (base_supp_size, 2):
    sys.exit(1)

for digit in range(10):
    path = data_dir / f"digit_{digit}" / "mappings.pt"
    if not path.exists():
        sys.exit(1)
    try:
        maps = torch.load(path, map_location="cpu")
    except Exception:
        sys.exit(1)
    if tuple(maps.shape) != (max_per_digit, base_supp_size, 2):
        sys.exit(1)
    if int(metadata.get("digit_counts", {}).get(str(digit), -1)) != max_per_digit:
        sys.exit(1)

sys.exit(0)
PY
}

train_outputs_exist() {
  [[ -d "$OUTPUT_DIR" ]] || return 1
  [[ -f "$OUTPUT_DIR/metrics.json" || -f "$OUTPUT_DIR/config.json" ]] && return 0
  find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | grep -q .
}

run_prepare_maps() {
  echo "[maps] computing MNIST OT maps"
  echo "  data_dir: $DATA_DIR"
  echo "  base_supp_size: $BASE_SUPP_SIZE"
  echo "  max_per_digit: $MAX_PER_DIGIT"
  echo "  seed: $SEED"
  echo "  base_mode: $BASE_MODE"
  echo "  progress_every: $PROGRESS_EVERY"
  "$PYTHON" -u mnist/pipeline/prepare_mnist_ot.py \
    --output_dir "$DATA_DIR" \
    --base_supp_size "$BASE_SUPP_SIZE" \
    --max_per_digit "$MAX_PER_DIGIT" \
    --seed "$SEED" \
    --base_mode "$BASE_MODE" \
    --base_digit "$BASE_DIGIT" \
    --base_noise_std "$BASE_NOISE_STD" \
    --base_n_components "$BASE_N_COMPONENTS" \
    --progress_every "$PROGRESS_EVERY" \
    2>&1 | tee "$LOG_DIR/prepare_mnist_ot_${MAX_PER_DIGIT}.log"
}

run_training() {
  if [[ "$SKIP_TRAIN" -eq 1 ]]; then
    echo "[train] skipped"
    return
  fi
  if [[ "$FORCE_TRAIN" -eq 0 ]] && train_outputs_exist; then
    echo "ERROR: OUTPUT_DIR already has training outputs: $OUTPUT_DIR" >&2
    echo "Use --force-train to write a new run into this directory, or choose --output-dir." >&2
    exit 1
  fi

  echo "[train] MNIST SAE"
  echo "  data_dir: $DATA_DIR"
  echo "  output_dir: $OUTPUT_DIR"
  echo "  device: $DEVICE"
  echo "  m: $M"
  echo "  lista_steps: $LISTA_STEPS"
  echo "  epochs: $EPOCHS"
  echo "  batch_size: $BATCH_SIZE"
  echo "  eps: $EPS"
  echo "  c: $C"
  "$PYTHON" -u mnist/pipeline/train_mnist_sae.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --m "$M" \
    --lista_steps "$LISTA_STEPS" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --epsilons "$EPS" \
    --sparsity_coeffs "$C" \
    2>&1 | tee "$LOG_DIR/train_mnist_SAE.log"
}

run_eval() {
  if [[ "$RUN_EVAL" -eq 0 ]]; then
    return
  fi
  if [[ ! -f "$OUTPUT_DIR/metrics.json" || ! -f "$OUTPUT_DIR/config.json" ]]; then
    echo "ERROR: Cannot evaluate; missing metrics.json/config.json in $OUTPUT_DIR" >&2
    exit 1
  fi

  echo "[eval] MNIST SAE visualizations"
  echo "  checkpoint_dir: $OUTPUT_DIR"
  echo "  output_dir: $EVAL_OUTPUT_DIR"
  "$PYTHON" -u mnist/analysis/evaluate_mnist_sae.py \
    --checkpoint_dir "$OUTPUT_DIR" \
    --data_dir "$DATA_DIR" \
    --output_dir "$EVAL_OUTPUT_DIR" \
    --device "$DEVICE" \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/eval_mnist_SAE.log"
}

mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1

echo "=== MNIST SAE pipeline ==="
echo "data_dir: $DATA_DIR"
echo "output_dir: $OUTPUT_DIR"
echo "device: $DEVICE"
echo "map_mode: $MAP_MODE"
echo "run_eval: $RUN_EVAL"
echo ""

case "$MAP_MODE" in
  auto)
    if maps_complete; then
      echo "[maps] existing matching MNIST OT maps found; skipping"
      echo "  data_dir: $DATA_DIR"
    else
      echo "[maps] missing, incomplete, or stale; preparing"
      run_prepare_maps
    fi
    ;;
  skip)
    echo "[maps] skipped"
    echo "  data_dir: $DATA_DIR"
    if ! maps_complete; then
      echo "ERROR: DATA_DIR is missing, incomplete, or does not match requested map params." >&2
      echo "Run without --skip-maps, or use --force-maps to rebuild." >&2
      exit 1
    fi
    ;;
  force)
    run_prepare_maps
    ;;
esac

run_training
run_eval

echo ""
echo "Done."
echo "Checkpoint and metrics: $OUTPUT_DIR"
if [[ "$RUN_EVAL" -eq 1 ]]; then
  echo "Evaluation outputs:     $EVAL_OUTPUT_DIR"
fi
