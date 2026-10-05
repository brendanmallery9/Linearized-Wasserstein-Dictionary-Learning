#!/usr/bin/env bash
# Prepare the three paper HSI datasets, compute their 1-D transport maps, and
# train every SAE configuration used by the corruption study.
#
# This is intentionally long-running.  Its default root is separate from the
# retained artifacts so a reproduction run cannot overwrite them accidentally.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ROOT="${ROOT:-datasets/hsi_paper_reproduction}"
SEEDS="${SEEDS:-0 1 2 3 4}"
EPOCHS="${EPOCHS:-400}"
FORCE_TRAIN=0

usage() {
  cat <<'EOF'
Usage: hsi/scripts/run_paper_hsi_models.sh [options]

Options:
  --root PATH          Output root. Default: datasets/hsi_paper_reproduction
  --seeds "0 1 ..."    Training seeds. Default: "0 1 2 3 4"
  --epochs N           Training epochs. Default: 400
  --force-train        Permit writing into existing SAE output directories
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --force-train) FORCE_TRAIN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

PIPELINE=(bash hsi/scripts/run_hsi_sae_pipeline.sh --root "$ROOT" --epochs "$EPOCHS" --seeds "$SEEDS")
FORCE_ARGS=()
if [[ "$FORCE_TRAIN" -eq 1 ]]; then
  FORCE_ARGS+=(--force-train)
fi

train_dataset() {
  local dataset="$1"
  local transport_params="$2"
  local linear_params="$3"

  echo "=== $dataset: data, transport maps, and transport-map SAEs ==="
  "${PIPELINE[@]}" \
    --dataset "$dataset" \
    --sae-mode transport_maps \
    --sae-params "$transport_params" \
    --train \
    "${FORCE_ARGS[@]}"

  echo "=== $dataset: linear SAEs ==="
  "${PIPELINE[@]}" \
    --dataset "$dataset" \
    --skip-download \
    --skip-maps \
    --sae-mode linear \
    --sae-params "$linear_params" \
    --train \
    --force-train
}

train_dataset botswana \
  '{"JUMPRELUAE_15_1e-1_mon":{"architecture":"JumpReLU_monotone","l1":"1e-1","lr":1e-4,"hidden_dim":15,"top_K":0},"JUMPRELUAE_15_5e-1_mon":{"architecture":"JumpReLU_monotone","l1":"5e-1","lr":1e-4,"hidden_dim":15,"top_K":0}}' \
  '{"JUMPRELUAE_15_1e-3_nonneg":{"architecture":"JumpReLU_nonneg","l1":"1e-3","lr":1e-4,"hidden_dim":15,"top_K":0},"JUMPRELUAE_15_1e-5_nonneg":{"architecture":"JumpReLU_nonneg","l1":"1e-5","lr":1e-4,"hidden_dim":15,"top_K":0}}'

train_dataset pavia \
  '{"JUMPRELUAE_10_5e-1_mon":{"architecture":"JumpReLU_monotone","l1":"5e-1","lr":1e-4,"hidden_dim":10,"top_K":0},"JUMPRELUAE_10_1e-2_mon":{"architecture":"JumpReLU_monotone","l1":"1e-2","lr":1e-4,"hidden_dim":10,"top_K":0}}' \
  '{"JUMPRELUAE_10_1e-3_nonneg":{"architecture":"JumpReLU_nonneg","l1":"1e-3","lr":1e-4,"hidden_dim":10,"top_K":0},"JUMPRELUAE_10_1e-4_nonneg":{"architecture":"JumpReLU_nonneg","l1":"1e-4","lr":1e-4,"hidden_dim":10,"top_K":0}}'

# The code and retained checkpoints use m=7 for Salinas A.  This resolves the
# manuscript's isolated m=15 entry in the uniform-dropout table in favor of the
# configuration used everywhere else.
train_dataset salinas_a \
  '{"JUMPRELUAE_7_1e-3_mon":{"architecture":"JumpReLU_monotone","l1":"1e-3","lr":1e-4,"hidden_dim":7,"top_K":0},"JUMPRELUAE_7_1e-5_mon":{"architecture":"JumpReLU_monotone","l1":"1e-5","lr":1e-5,"hidden_dim":7,"top_K":0}}' \
  '{"JUMPRELUAE_7_1e-3_nonneg":{"architecture":"JumpReLU_nonneg","l1":"1e-3","lr":1e-4,"hidden_dim":7,"top_K":0},"JUMPRELUAE_7_1e-5_nonneg":{"architecture":"JumpReLU_nonneg","l1":"1e-5","lr":1e-4,"hidden_dim":7,"top_K":0}}'

echo "Paper HSI models are under $ROOT."
echo "Next: bash hsi/scripts/full_corruption_sweep_hyperspec.sh --root '$ROOT' --sae-mode both"
