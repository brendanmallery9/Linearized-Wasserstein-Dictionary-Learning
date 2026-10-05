#!/usr/bin/env bash
# Corruption robustness sweep across ALL methods and selected corruption types:
#
#   Methods evaluated per run:
#     - Transport-map SAE  (OT-embedded cube → JumpReLU SAE, monotone weights)
#     - Linear SAE      (raw cube → JumpReLU SAE, nonneg weights)
#     - NMF             (baseline; always run automatically by inner script)
#
#   Corruption types:  drop_random, drop_contiguous, log_warp
#   Dropout severities: k = 0.1, 0.2, 0.3, 0.4, 0.5
#   Log-warp severities: k = 1, 10, 100, 500, 1000
#   Corruption seeds:  0 – 4
#   Training seeds:    0 – 4
#
# Device detection (at runtime):
#   CUDA available  →  distribute across all CUDA GPUs (up to 3, one per type)
#   MPS available   →  run all types sequentially on MPS
#   otherwise       →  run all types sequentially on CPU
#
# Prerequisites: SAE models for the selected --sae-mode must already be trained.
#
# Usage:
#   mkdir -p hsi/results
#   nohup bash hsi/scripts/full_corruption_sweep_hyperspec.sh --root datasets/hsi_data --sae-mode both >& hsi/results/full_sweep.log &
#   tail -f hsi/results/full_sweep.log

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_DIR="$SCRIPT_DIR/../pipeline"
WRAPPER="$PIPELINE_DIR/corruption_sweep_wrapper.py"
ROOT="${ROOT:-datasets/hsi_data}"
SAE_MODE="${SAE_MODE:-both}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/../results}"
DATASETS="${DATASETS:-pavia botswana salinas_a}"
SEEDS="${SEEDS:-0 1 2 3 4}"
CORRUPTION_SEEDS="${CORRUPTION_SEEDS:-0 1 2 3 4}"
DROP_K_VALUES="${DROP_K_VALUES:-0.1 0.2 0.3 0.4 0.5}"
LOG_WARP_K_VALUES="${LOG_WARP_K_VALUES:-1 10 100 500 1000}"

usage() {
    cat <<'EOF'
Usage:
  hsi/scripts/full_corruption_sweep_hyperspec.sh [options]

Options:
  --root PATH          Hyperspectral dataset root. Default: datasets/hsi_data
  --sae-mode MODE      transport_maps, linear, or both. Default: both
  --results-dir PATH   Directory for JSONs, tables, and slot logs. Default: hsi/results
  --datasets "NAMES"   Space-separated dataset directories. Default: "pavia botswana salinas_a"
  --seeds "N ..."      Space-separated SAE training seeds. Default: "0 1 2 3 4"
  --corruption-seeds "N ..."
                       Space-separated corruption RNG seeds. Default: "0 1 2 3 4"
  --drop-k-values "K ..."
                       Dropout severities. Default: "0.1 0.2 0.3 0.4 0.5"
  --log-warp-k-values "K ..."
                       Log-warp severities. Default: "1 10 100 500 1000"
  -h, --help           Show this help

Environment overrides:
  ROOT=datasets/hsi_data
  SAE_MODE=both
  RESULTS_DIR=hsi/results
  DATASETS="pavia botswana salinas_a"
  SEEDS="0 1 2 3 4"
  CORRUPTION_SEEDS="0 1 2 3 4"
  DROP_K_VALUES="0.1 0.2 0.3 0.4 0.5"
  LOG_WARP_K_VALUES="1 10 100 500 1000"
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            ROOT="$2"
            shift 2
            ;;
        --sae-mode)
            SAE_MODE="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --corruption-seeds)
            CORRUPTION_SEEDS="$2"
            shift 2
            ;;
        --drop-k-values)
            DROP_K_VALUES="$2"
            shift 2
            ;;
        --log-warp-k-values)
            LOG_WARP_K_VALUES="$2"
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

if [[ "$SAE_MODE" == "potentials" ]]; then
    echo "Note: --sae-mode potentials is deprecated for HSI; using transport_maps." >&2
    SAE_MODE="transport_maps"
fi

if [[ "$SAE_MODE" != "transport_maps" && "$SAE_MODE" != "linear" && "$SAE_MODE" != "both" ]]; then
    echo "ERROR: --sae-mode must be one of: transport_maps, linear, both. Got: $SAE_MODE" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Detect compute device and number of parallel slots
# ---------------------------------------------------------------------------
DEVICE_INFO=$(python3 -c "
import torch

if torch.cuda.is_available():
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    print(f'cuda {n}')
    print(f'  {n} CUDA GPU(s) detected:', flush=True)
    for i, name in enumerate(names):
        print(f'    GPU {i}: {name}', flush=True)
elif torch.backends.mps.is_available():
    print('mps 1')
    print('  MPS (Apple Silicon) detected', flush=True)
else:
    print('cpu 1')
    print('  No GPU detected — falling back to CPU', flush=True)
" 2>/dev/null)

DEVICE_TOKEN=$(echo "$DEVICE_INFO" | grep -m1 -E '^(cuda|mps|cpu) [0-9]+')
DEVICE_TYPE=$(echo "$DEVICE_TOKEN" | awk '{print $1}')
N_SLOTS=$(echo  "$DEVICE_TOKEN" | awk '{print $2}')

echo "--- Device check ---"
echo "$DEVICE_INFO" | tail -n +2
echo ""
echo "Root: $ROOT"
echo "SAE mode: $SAE_MODE"
echo "Datasets: $DATASETS"
echo "Training seeds: $SEEDS"
echo "Corruption seeds: $CORRUPTION_SEEDS"
mkdir -p "$RESULTS_DIR"
RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"
echo "Results dir: $RESULTS_DIR"
echo ""

# ---------------------------------------------------------------------------
# Paper key pairs: transport_maps (mon) + linear (nonneg) for the three
# reported datasets.  Hidden dimension identifies the dataset here.
#   NMF baseline is run automatically by the inner clustering script.
#
#   Botswana      → hidden_dim 15
#   Pavia         → hidden_dim 10
#   Salinas_A     → hidden_dim  7
# ---------------------------------------------------------------------------
TRANSPORT_MAP_KEY_PAIRS=(
    # --- transport maps (OT-embedded) ---
    transport_maps JUMPRELUAE_15_1e-1_mon
    transport_maps JUMPRELUAE_15_5e-1_mon
    transport_maps JUMPRELUAE_10_5e-1_mon
    transport_maps JUMPRELUAE_10_1e-2_mon
    transport_maps JUMPRELUAE_7_1e-5_mon
    transport_maps JUMPRELUAE_7_1e-3_mon
)

LINEAR_KEY_PAIRS=(
    # --- linear (raw cube) ---
    linear JUMPRELUAE_15_1e-3_nonneg
    linear JUMPRELUAE_15_1e-5_nonneg
    linear JUMPRELUAE_10_1e-3_nonneg
    linear JUMPRELUAE_10_1e-4_nonneg
    linear JUMPRELUAE_7_1e-3_nonneg
    linear JUMPRELUAE_7_1e-5_nonneg
)

KEY_PAIRS=()
case "$SAE_MODE" in
    transport_maps)
        KEY_PAIRS=("${TRANSPORT_MAP_KEY_PAIRS[@]}")
        ;;
    linear)
        KEY_PAIRS=("${LINEAR_KEY_PAIRS[@]}")
        ;;
    both)
        KEY_PAIRS=("${TRANSPORT_MAP_KEY_PAIRS[@]}" "${LINEAR_KEY_PAIRS[@]}")
        ;;
esac

COMMON_ARGS=(
    --root "$ROOT"
    --datasets $DATASETS
    --corruption_seeds $CORRUPTION_SEEDS
    --seeds $SEEDS
    --key_pairs "${KEY_PAIRS[@]}"
)

# ---------------------------------------------------------------------------
# Assign corruption types to slots (round-robin)
# ---------------------------------------------------------------------------
CORRUPTION_TYPES=(drop_random drop_contiguous log_warp)
N_TYPES=${#CORRUPTION_TYPES[@]}

if [ "$N_SLOTS" -gt "$N_TYPES" ]; then
    N_SLOTS=$N_TYPES
fi

echo "Parallel slots: $N_SLOTS  (corruption types: ${CORRUPTION_TYPES[*]})"
echo ""

# ---------------------------------------------------------------------------
# Launch one background subshell per slot
# ---------------------------------------------------------------------------
declare -a PIDS=()

for slot in $(seq 0 $(( N_SLOTS - 1 ))); do
    types=()
    for i in "${!CORRUPTION_TYPES[@]}"; do
        if [ $(( i % N_SLOTS )) -eq "$slot" ]; then
            types+=("${CORRUPTION_TYPES[$i]}")
        fi
    done
    [ "${#types[@]}" -eq 0 ] && continue

    log_file="$RESULTS_DIR/slot${slot}_full_sweep.log"

    (
        if [ "$DEVICE_TYPE" = "cuda" ]; then
            export CUDA_VISIBLE_DEVICES=$slot
            label="GPU$slot"
        else
            label="$DEVICE_TYPE"
        fi

        for ctype in "${types[@]}"; do
            echo "[$label] Starting $ctype sweep"
            if [[ "$ctype" == "log_warp" ]]; then
                k_values=( $LOG_WARP_K_VALUES )
            else
                k_values=( $DROP_K_VALUES )
            fi
            python -u "$WRAPPER" "${COMMON_ARGS[@]}" --k_values "${k_values[@]}" \
                --corruption_type "$ctype" \
                --output "$RESULTS_DIR/${ctype}.json"
            echo "[$label] $ctype done"
        done
    ) &> "$log_file" &

    pid=$!
    PIDS+=("$pid")
    echo "Slot $slot ($(IFS=+; echo "${types[*]}")) → PID $pid, log: $log_file"
done

echo ""
echo "Waiting for all slots to finish..."
_failed_slots=0
for _pid in "${PIDS[@]}"; do
    wait "$_pid" || (( _failed_slots++ )) || true
done
echo ""
if [ "$_failed_slots" -gt 0 ]; then
    echo "WARNING: $_failed_slots slot(s) exited with errors. Check per-slot logs in $RESULTS_DIR."
else
    echo "All sweeps complete."
fi
echo "Output JSONs : $RESULTS_DIR/drop_random.json $RESULTS_DIR/drop_contiguous.json $RESULTS_DIR/log_warp.json"
echo "Visualizations: $RESULTS_DIR/drop_random/ $RESULTS_DIR/drop_contiguous/ $RESULTS_DIR/log_warp/"
