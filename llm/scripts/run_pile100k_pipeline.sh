#!/bin/bash
# pile-100k pipeline: embed -> Gaussian source + PCA -> Brenier potentials -> optional SAE training
#
# Usage:
#   bash llm/scripts/run_pile100k_pipeline.sh                          # reuse existing activations/SAE_params by default
#   bash llm/scripts/run_pile100k_pipeline.sh --device mps             # Apple Silicon
#   bash llm/scripts/run_pile100k_pipeline.sh --device cpu             # CPU only
#   bash llm/scripts/run_pile100k_pipeline.sh --num-gpus 4             # multi-GPU embed (cuda only)
#   bash llm/scripts/run_pile100k_pipeline.sh --skip-embed             # reuse existing activations
#   bash llm/scripts/run_pile100k_pipeline.sh --skip-gaussian          # reuse source.pt + pca.pt
#   bash llm/scripts/run_pile100k_pipeline.sh --skip-brenier           # reuse potentials/
#   bash llm/scripts/run_pile100k_pipeline.sh --train                  # train SAE_params/
#   bash llm/scripts/run_pile100k_pipeline.sh --train --force-train    # overwrite existing SAE_params/
#   bash llm/scripts/run_pile100k_pipeline.sh --skip-train             # reuse SAE_params/
#   nohup bash llm/scripts/run_pile100k_pipeline.sh >& run_pile100k.log &

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_DIR="$SCRIPT_DIR/../pipeline"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- defaults (overridable via flags or env) ----
ROOT="${ROOT:-$REPO_ROOT/datasets/pile-100k}"
DATASET="${DATASET:-jannikbrinkmann/pile-100k}"
N_DOCS="${N_DOCS:-100}"             # -1 means all docs
STREAMING="${STREAMING:-true}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-200}"
NUM_GPUS="${NUM_GPUS:-1}"          # parallel embed processes (cuda only)

MODEL="${MODEL:-EleutherAI/pythia-410m-deduped}"
N_GAUSSIAN="${N_GAUSSIAN:-1000}"   # Gaussian source sample size; becomes SAE INPUT_DIM
PCA_COMPONENTS="${PCA_COMPONENTS:-150}"
N_WORKERS="${N_WORKERS:-16}"       # parallel workers for Brenier step

# DEVICE auto-detect (cuda > cpu). Override via --device or DEVICE env.
DEVICE="${DEVICE:-}"

SKIP_EMBED=false
SKIP_GAUSSIAN=false
SKIP_BRENIER=false
SKIP_TRAIN=true
FORCE_TRAIN=false

# ---- parse flags ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)         DEVICE="$2"; shift 2 ;;
        --num-gpus)       NUM_GPUS="$2"; shift 2 ;;
        --epochs)         EPOCHS="$2"; shift 2 ;;
        --n-docs)         N_DOCS="$2"; shift 2 ;;
        --root)           ROOT="$2"; shift 2 ;;
        --skip-embed)     SKIP_EMBED=true; shift ;;
        --skip-gaussian)  SKIP_GAUSSIAN=true; shift ;;
        --skip-brenier)   SKIP_BRENIER=true; shift ;;
        --train)          SKIP_TRAIN=false; shift ;;
        --force-train)    FORCE_TRAIN=true; shift ;;
        --skip-train)     SKIP_TRAIN=true; shift ;;
        -h|--help)        sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Auto-detect device if not set
if [ -z "$DEVICE" ]; then
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        DEVICE="cuda"
    else
        DEVICE="cpu"
    fi
fi

# Force single-process embed when not on cuda (gpu_id splitting only makes sense for cuda)
if [ "$DEVICE" != "cuda" ] && [ "$NUM_GPUS" -gt 1 ]; then
    echo "Note: --num-gpus=$NUM_GPUS ignored on device=$DEVICE; using 1 process"
    NUM_GPUS=1
fi

ACT_DIR="$ROOT/activations"
ACT_PILE_DIR="$ACT_DIR/pile"
SOURCE_PT="$ROOT/source.pt"
PCA_PT="$ROOT/pca.pt"
POTENTIALS_DIR="$ROOT/potentials"
SAE_OUT="$ROOT/SAE_params"
SAE_STACK="$ROOT/non_normalized_stacked"

mkdir -p "$ACT_PILE_DIR" "$POTENTIALS_DIR" "$SAE_OUT"

dir_has_files() {
    [ -d "$1" ] && [ -n "$(find "$1" -mindepth 1 -type f -print -quit 2>/dev/null)" ]
}

PIDS=""

cleanup_children() {
    if [ -n "$PIDS" ]; then
        kill $PIDS 2>/dev/null || true
    fi
}

trap cleanup_children INT TERM

echo "=== Config ==="
echo "ROOT       = $ROOT"
echo "DATASET    = $DATASET"
echo "MODEL      = $MODEL"
echo "DEVICE     = $DEVICE"
echo "NUM_GPUS   = $NUM_GPUS"
echo "N_DOCS     = $N_DOCS  (-1 = all)"
echo "EPOCHS     = $EPOCHS"
echo

# ----------------------------------------------------------------------------
# Step 1 — embed pile docs
# ----------------------------------------------------------------------------
if [ "$SKIP_EMBED" = true ]; then
    echo "=== Step 1: SKIPPED (--skip-embed) ==="
else
    echo "=== Step 1: Embed pile-100k docs (${NUM_GPUS} process(es), device=$DEVICE) ==="
    EMBED_EXTRA_FLAGS=""
    [ "$STREAMING" = true ] && EMBED_EXTRA_FLAGS="$EMBED_EXTRA_FLAGS --streaming"
    [ "$N_DOCS" -gt 0 ]    && EMBED_EXTRA_FLAGS="$EMBED_EXTRA_FLAGS --n_docs $N_DOCS"

    PIDS=""
    for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
        # shellcheck disable=SC2086
        python "$PIPELINE_DIR/embed_document_from_hf.py" \
            --out_dir    "$ACT_DIR" \
            --model_name "$MODEL" \
            --dataset    "$DATASET" \
            --seed       "$SEED" \
            --subdir     pile \
            --device     "$DEVICE" \
            --gpu_id     "$GPU_ID" \
            --num_gpus   "$NUM_GPUS" \
            $EMBED_EXTRA_FLAGS &
        PID=$!
        PIDS="${PIDS:+$PIDS }$PID"
        echo "  Launched worker $GPU_ID (PID $PID)"
    done

    EMBED_FAIL=0
    for PID in $PIDS; do
        if ! wait "$PID"; then
            echo "ERROR: Embedding process $PID failed"
            EMBED_FAIL=1
        fi
    done
    PIDS=""
    if [ "$EMBED_FAIL" -ne 0 ]; then
        echo "One or more embedding processes failed. Aborting."
        exit 1
    fi
    echo "All $NUM_GPUS embedding process(es) completed successfully."
fi

# ----------------------------------------------------------------------------
# Step 2 — Gaussian source + PCA
# ----------------------------------------------------------------------------
echo
if [ "$SKIP_GAUSSIAN" = true ]; then
    echo "=== Step 2: SKIPPED (--skip-gaussian) ==="
elif [ -f "$SOURCE_PT" ] && [ -f "$PCA_PT" ]; then
    echo "=== Step 2: source.pt and pca.pt already exist — skipping fit ==="
    echo "    (delete $SOURCE_PT and/or $PCA_PT to force a refit)"
else
    echo "=== Step 2: Fit Gaussian source + PCA ==="
    python "$PIPELINE_DIR/compute_gaussian_source.py" \
        --activations_dir "$ACT_PILE_DIR" \
        --source_output   "$SOURCE_PT" \
        --pca_output      "$PCA_PT" \
        --n_samples       "$N_GAUSSIAN" \
        --pca_components  "$PCA_COMPONENTS"
fi

# ----------------------------------------------------------------------------
# Step 3 — Brenier potentials
# ----------------------------------------------------------------------------
echo
if [ "$SKIP_BRENIER" = true ]; then
    echo "=== Step 3: SKIPPED (--skip-brenier) ==="
else
    echo "=== Step 3: Compute Brenier potentials (parallel) ==="
    python "$PIPELINE_DIR/multi_dir_brenier_embedding.py" \
        --activations_root "$ACT_DIR" \
        --output_root      "$POTENTIALS_DIR" \
        --source_path      "$SOURCE_PT" \
        --object           potential \
        --method           emd \
        --eps_reg          0.0 \
        --pca_transform    "$PCA_PT" \
        --num_workers      "$N_WORKERS"
fi

# ----------------------------------------------------------------------------
# Step 4 — Train SAEs
# ----------------------------------------------------------------------------
echo
if [ "$SKIP_TRAIN" = true ]; then
    echo "=== Step 4: SKIPPED (--skip-train) ==="
else
    if [ "$FORCE_TRAIN" = false ] && dir_has_files "$SAE_OUT"; then
        echo "ERROR: SAE outputs already exist under $SAE_OUT"
        echo "Use --force-train with --train to intentionally write new models there, or set ROOT to a new run directory."
        exit 1
    fi
    echo "=== Step 4: Train SAEs -> $SAE_OUT ==="
    python "$PIPELINE_DIR/potential_SAE_analysis_script.py" \
        --data_dir    "$POTENTIALS_DIR" \
        --out_root    "$SAE_OUT" \
        --stack_dir   "$SAE_STACK" \
        --n_per_dir   -1 \
        --epochs      "$EPOCHS" \
        --lr          5e-6 \
        --batch_size  1024 \
        --device      "$DEVICE"
fi

echo
echo "Done. SAE outputs: $SAE_OUT"
