#!/bin/bash
# Sweep over N_GAUSSIAN values, running the potentials pipeline for each.
# Assumes Step 1 (embeddings) is already done.
# Steps 2-3 (CPU-bound) run sequentially to avoid memory/IO contention.
# Step 4 (SAE training, GPU-bound) is opt-in and launches all variants in parallel on separate GPUs.
#
# Usage:
#   bash run_pile100k_sweep.sh                  # steps 2-3 only; no training by default
#   bash run_pile100k_sweep.sh --train          # steps 2-3-4
#   bash run_pile100k_sweep.sh --train --force-train
#   bash run_pile100k_sweep.sh --sae-only       # skip steps 2-3, train SAEs only
#
# Example:
#   nohup bash run_pile100k_sweep.sh --sae-only >& run_sweep.log &

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_DIR="$SCRIPT_DIR/../pipeline"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- config ----
ROOT="${ROOT:-$REPO_ROOT/datasets/pile-100k}"
PCA_COMPONENTS=150
N_WORKERS=16
EPOCHS=6000
BATCH_SIZE=1024
LR=5e-6
SEED=42

N_GAUSSIAN_VALUES=(200 400 600 800)
NUM_GPUS=4
# ----------------

# ---- parse flags ----
SAE_ONLY=false
TRAIN=false
FORCE_TRAIN=false
for arg in "$@"; do
    case "$arg" in
        --train) TRAIN=true ;;
        --skip-train) TRAIN=false ;;
        --force-train) FORCE_TRAIN=true ;;
        --sae-only) SAE_ONLY=true; TRAIN=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

dir_has_files() {
    [ -d "$1" ] && [ -n "$(find "$1" -mindepth 1 -type f -print -quit 2>/dev/null)" ]
}

# ---- Phase 1: Steps 2-3 sequentially (CPU-bound) ----
if [ "$SAE_ONLY" = false ]; then
    for N_GAUSSIAN in "${N_GAUSSIAN_VALUES[@]}"; do
        SOURCE="$ROOT/source_${N_GAUSSIAN}.pt"
        PCA="$ROOT/pca_${N_GAUSSIAN}.pt"
        POTENTIALS="$ROOT/potentials_${N_GAUSSIAN}"

        echo ""
        echo "################################################################"
        echo "# N_GAUSSIAN = $N_GAUSSIAN - Steps 2-3 (CPU)"
        echo "################################################################"

        echo "=== Step 2: Fit Gaussian source + PCA (N=$N_GAUSSIAN) ==="
        python "$PIPELINE_DIR/compute_gaussian_source.py" \
            --activations_dir "$ROOT/activations/pile" \
            --source_output   "$SOURCE" \
            --pca_output      "$PCA" \
            --n_samples       "$N_GAUSSIAN" \
            --pca_components  "$PCA_COMPONENTS"

        echo "=== Step 3: Compute Brenier potentials (N=$N_GAUSSIAN) ==="
        python "$PIPELINE_DIR/multi_dir_brenier_embedding.py" \
            --activations_root "$ROOT/activations" \
            --output_root      "$POTENTIALS" \
            --source_path      "$SOURCE" \
            --object           potential \
            --method           emd \
            --eps_reg          0.0 \
            --pca_transform    "$PCA" \
            --num_workers      "$N_WORKERS"

        echo "=== Done with steps 2-3 for N_GAUSSIAN=$N_GAUSSIAN ==="
    done
else
    echo "Skipping steps 2-3 (--sae-only)"
fi

# ---- Phase 2: Step 4 in parallel on separate GPUs ----
echo ""
if [ "$TRAIN" = false ]; then
    echo "################################################################"
    echo "# Skipping SAE training (default; pass --train or --sae-only)"
    echo "################################################################"
    echo ""
    echo "All sweeps complete."
    echo "SAE training was skipped."
    exit 0
fi

echo "################################################################"
echo "# Launching SAE training on ${NUM_GPUS} GPUs"
echo "################################################################"

PIDS=()
for i in "${!N_GAUSSIAN_VALUES[@]}"; do
    N_GAUSSIAN=${N_GAUSSIAN_VALUES[$i]}
    GPU_ID=$(( i % NUM_GPUS ))
    POTENTIALS="$ROOT/potentials_${N_GAUSSIAN}"
    SAE_OUT="$ROOT/SAE_params_${N_GAUSSIAN}"
    STACK_DIR="$ROOT/stacked_${N_GAUSSIAN}"
    LOG="$ROOT/sae_train_${N_GAUSSIAN}.log"

    if [ "$FORCE_TRAIN" = false ] && dir_has_files "$SAE_OUT"; then
        echo "ERROR: SAE outputs already exist under $SAE_OUT"
        echo "Use --force-train with --train to intentionally write new models there, or set ROOT to a new run directory."
        exit 1
    fi

    echo "Launching SAE training N_GAUSSIAN=$N_GAUSSIAN on cuda:$GPU_ID (log: $LOG)"
    python "$PIPELINE_DIR/potential_SAE_analysis_script.py" \
        --data_dir    "$POTENTIALS" \
        --out_root    "$SAE_OUT" \
        --stack_dir   "$STACK_DIR" \
        --n_per_dir   -1 \
        --epochs      $EPOCHS \
        --lr          $LR \
        --batch_size  $BATCH_SIZE \
        --device      "cuda:$GPU_ID" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

# Wait for all SAE training jobs
FAIL=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    N_GAUSSIAN=${N_GAUSSIAN_VALUES[$i]}
    if wait "$PID"; then
        echo "N_GAUSSIAN=$N_GAUSSIAN (PID $PID) SAE training completed successfully."
    else
        echo "ERROR: N_GAUSSIAN=$N_GAUSSIAN (PID $PID) SAE training failed. See $ROOT/sae_train_${N_GAUSSIAN}.log"
        FAIL=1
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo "One or more SAE training jobs failed."
    exit 1
fi

echo ""
echo "All sweeps complete."
echo "SAE outputs: $ROOT/SAE_params_200"
