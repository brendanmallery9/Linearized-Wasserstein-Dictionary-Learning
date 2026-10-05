#!/usr/bin/env bash
# Launch the deterministic 02378 train-mean seed sweep.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-mnist/results/convex_combos_02378_train_mean_seed_sweep}"
N_SEEDS="${N_SEEDS:-16}"
SCREEN_EPOCHS="${SCREEN_EPOCHS:-350}"
FINAL_EPOCHS="${FINAL_EPOCHS:-1000}"
N_FINALISTS="${N_FINALISTS:-4}"
DEVICE="${DEVICE:-mps}"

if [[ -n "${WAIT_FOR_PID:-}" ]]; then
  echo "Waiting for PID $WAIT_FOR_PID before starting the sweep..."
  while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do
    sleep 30
  done
  echo "PID $WAIT_FOR_PID finished; starting sweep."
fi

exec "$PYTHON" -u mnist/scripts/run_convex_combos_02378_nn_seed_sweep.py \
  --output_root "$OUTPUT_ROOT" \
  --n_seeds "$N_SEEDS" \
  --screen_epochs "$SCREEN_EPOCHS" \
  --final_epochs "$FINAL_EPOCHS" \
  --n_finalists "$N_FINALISTS" \
  --device "$DEVICE"
