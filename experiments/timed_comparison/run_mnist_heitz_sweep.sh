#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUN_DIR="${RUN_DIR:-${REPO_ROOT}/experiments/results/mnist_heitz_sweep_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-mps}"
SEED="${SEED:-42}"

MAX_PER_DIGIT="${MAX_PER_DIGIT:-10}"
BASE_SUPP_SIZE="${BASE_SUPP_SIZE:-400}"
ATOMS="${ATOMS:-10}"
LISTA_STEPS="${LISTA_STEPS:-20}"
EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EPS="${EPS:-0.025}"
C="${C:-0.0001}"

HEITZ_LOSS_TYPE="${HEITZ_LOSS_TYPE:-2}"
HEITZ_SCALE_DICT_FACTOR="${HEITZ_SCALE_DICT_FACTOR:-100.0}"
HEITZ_AVX="${HEITZ_AVX:-auto}"

PYTHON="${PYTHON:-python}"

mkdir -p "${RUN_DIR}"
RUN_DIR="$(cd "${RUN_DIR}" && pwd)"

cat > "${RUN_DIR}/sweep_manifest.json" <<EOF
{
  "max_per_digit": ${MAX_PER_DIGIT},
  "base_supp_size": ${BASE_SUPP_SIZE},
  "atoms": ${ATOMS},
  "lista_steps": ${LISTA_STEPS},
  "epochs": ${EPOCHS},
  "batch_size": ${BATCH_SIZE},
  "eps": ${EPS},
  "c": ${C},
  "seed": ${SEED},
  "device": "${DEVICE}",
  "heitz_loss_type": ${HEITZ_LOSS_TYPE},
  "heitz_scale_dict_factor": ${HEITZ_SCALE_DICT_FACTOR},
  "heitz_avx": "${HEITZ_AVX}",
  "heitz_trials": [
    {"gamma": 0.5, "sinkhorn_iters": 5, "max_optim_iter": 10},
    {"gamma": 0.5, "sinkhorn_iters": 25, "max_optim_iter": 50},
    {"gamma": 2.0, "sinkhorn_iters": 5, "max_optim_iter": 10},
    {"gamma": 2.0, "sinkhorn_iters": 25, "max_optim_iter": 50}
  ]
}
EOF

echo "Run directory: ${RUN_DIR}"

echo
echo "=== Preparing shared MNIST PNGs for Heitz variants ==="
"${PYTHON}" "${SCRIPT_DIR}/prepare_mnist_png.py" \
  --output-dir "${RUN_DIR}/shared_mnist_png" \
  --mnist-root "${REPO_ROOT}/mnist_raw" \
  --max-per-digit "${MAX_PER_DIGIT}" \
  --force

echo
echo "=== Running MNIST OT-SAE once ==="
"${PYTHON}" "${SCRIPT_DIR}/run_mnist_sae_timed.py" \
  --run-dir "${RUN_DIR}/mnist_ot_sae" \
  --map-mode force \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --base-supp-size "${BASE_SUPP_SIZE}" \
  --max-per-digit "${MAX_PER_DIGIT}" \
  --m "${ATOMS}" \
  --lista-steps "${LISTA_STEPS}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --eps "${EPS}" \
  --c "${C}" \
  --force-output

run_heitz_trial() {
  local gamma="$1"
  local sinkhorn_iters="$2"
  local max_optim_iter="$3"
  local gamma_label="${gamma/./p}"
  local run_name="heitz_gamma${gamma_label}_sink${sinkhorn_iters}_optim${max_optim_iter}"

  echo
  echo "=== Running ${run_name} ==="
  "${PYTHON}" "${SCRIPT_DIR}/run_heitz_wdl.py" \
    --run-dir "${RUN_DIR}/${run_name}" \
    --input-dir "${RUN_DIR}/shared_mnist_png/all" \
    --k "${ATOMS}" \
    --loss-type "${HEITZ_LOSS_TYPE}" \
    --sinkhorn-iters "${sinkhorn_iters}" \
    --max-optim-iter "${max_optim_iter}" \
    --gamma "${gamma}" \
    --scale-dict-factor "${HEITZ_SCALE_DICT_FACTOR}" \
    --avx "${HEITZ_AVX}" \
    --deterministic
}

run_heitz_trial 0.5 5 10
run_heitz_trial 0.5 25 50
run_heitz_trial 2.0 5 10
run_heitz_trial 2.0 25 50

echo
echo "=== Plotting all loss curves ==="
"${PYTHON}" "${SCRIPT_DIR}/plot_mnist_sweep_curves.py" \
  --run-dir "${RUN_DIR}" \
  --output "${RUN_DIR}/loss_vs_wall_time_sweep.png"

echo
echo "Sweep complete: ${RUN_DIR}"
echo "Combined loss plot: ${RUN_DIR}/loss_vs_wall_time_sweep.png"
