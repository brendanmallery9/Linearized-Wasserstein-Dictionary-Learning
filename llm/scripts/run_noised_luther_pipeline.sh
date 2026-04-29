#!/usr/bin/env bash
# noised-luther pipeline:
#   1. Generate noised text variants (skipped if already present)
#   2. Embed the base text into activations
#   3. Embed each noised-corruption subdir into its own activation subdir
#   4. Compute Brenier potentials with base activations as the source (no PCA)
#   5. Optionally train SAEs (TopK10/k=2, TopK40/k=3, JumpReLU10/l1=1e-3, JumpReLU40/l1=1e-3)
#
# Usage:
#   bash llm/scripts/run_noised_luther_pipeline.sh                    # reuse existing SAE_params by default
#   bash llm/scripts/run_noised_luther_pipeline.sh --max-docs 1       # smoke test: 1 seed per (n,type)
#   bash llm/scripts/run_noised_luther_pipeline.sh --device mps       # run on Apple Silicon
#   bash llm/scripts/run_noised_luther_pipeline.sh --skip-noise       # skip step 1
#   bash llm/scripts/run_noised_luther_pipeline.sh --train            # train SAE_params
#   bash llm/scripts/run_noised_luther_pipeline.sh --train --force-train
#   bash llm/scripts/run_noised_luther_pipeline.sh --skip-train       # skip step 5 (reuse existing SAE_params)
#   MAX_DOCS=1 DEVICE=cpu bash llm/scripts/run_noised_luther_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_DIR="$SCRIPT_DIR/../pipeline"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- defaults (overridable by env or flags) ----
ROOT="${ROOT:-$REPO_ROOT/datasets/noised_luther}"
MODEL="${MODEL:-EleutherAI/pythia-410m-deduped}"
MODEL_SLUG="${MODEL//\//__}"

DEVICE="${DEVICE:-cpu}"          # cuda | mps | cpu
MAX_DOCS="${MAX_DOCS:-20}"        # -1 = all; >0 = N per corruption subdir (and base)
EPOCHS="${EPOCHS:-5}"
LR="${LR:-5e-6}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

SKIP_NOISE=false
SKIP_TRAIN=true
FORCE_TRAIN=false

# ---- parse flags ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-docs)   MAX_DOCS="$2"; shift 2 ;;
        --device)     DEVICE="$2"; shift 2 ;;
        --epochs)     EPOCHS="$2"; shift 2 ;;
        --root)       ROOT="$2"; shift 2 ;;
        --skip-noise) SKIP_NOISE=true; shift ;;
        --train)      SKIP_TRAIN=false; shift ;;
        --force-train) FORCE_TRAIN=true; shift ;;
        --skip-train) SKIP_TRAIN=true; shift ;;
        -h|--help)    sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---- derived paths ----
TXT_BASE_DIR="$ROOT/txt/base"
TXT_NOISED_DIR="$ROOT/txt/noised_docs"

ACT_BASE_DIR="$ROOT/activations/base"
ACT_NOISED_ROOT="$ROOT/activations/noised_activations/$MODEL_SLUG"

POTENTIALS_DIR="$ROOT/potentials/$MODEL_SLUG/raw_potentials"

SAE_OUT="$ROOT/SAE_params"
SAE_STACK="$ROOT/stacked"

BASE_TXT="$TXT_BASE_DIR/luther.txt"
BASE_ACT_PT="$ACT_BASE_DIR/luther_L2_residual.pt"

EMBED_FLAGS=(--model "$MODEL" --device "$DEVICE")
[ "$MAX_DOCS" -gt 0 ] && EMBED_FLAGS+=(--max-docs "$MAX_DOCS")

dir_has_files() {
    [ -d "$1" ] && [ -n "$(find "$1" -mindepth 1 -type f -print -quit 2>/dev/null)" ]
}

# ----------------------------------------------------------------------------
echo "=== Config ==="
echo "ROOT          = $ROOT"
echo "MODEL         = $MODEL"
echo "DEVICE        = $DEVICE"
echo "MAX_DOCS      = $MAX_DOCS  (-1 = all)"
echo "EPOCHS        = $EPOCHS"
echo

# ----------------------------------------------------------------------------
# Step 1 — generate noised docs (skip if already populated)
# ----------------------------------------------------------------------------
if [ "$SKIP_NOISE" = true ]; then
    echo "=== Step 1: SKIPPED (--skip-noise) ==="
elif [ -d "$TXT_NOISED_DIR" ] && [ -n "$(find "$TXT_NOISED_DIR" -mindepth 2 -name 'seed_*.txt' -print -quit 2>/dev/null)" ]; then
    echo "=== Step 1: noised docs already present at $TXT_NOISED_DIR — skipping ==="
else
    echo "=== Step 1: Generating noised docs ==="
    [ -f "$BASE_TXT" ] || { echo "Missing base text: $BASE_TXT"; exit 1; }
    python "$PIPELINE_DIR/generate_noised_docs.py" \
        --base-text "$BASE_TXT" \
        --out-root  "$TXT_NOISED_DIR"
fi

# ----------------------------------------------------------------------------
# Step 2 — embed the base text
# ----------------------------------------------------------------------------
echo
echo "=== Step 2: Embed base text -> $ACT_BASE_DIR ==="
python "$PIPELINE_DIR/embed_documents_from_dir.py" \
    --data-dir "$TXT_BASE_DIR" \
    --out-dir  "$ACT_BASE_DIR" \
    "${EMBED_FLAGS[@]}"

[ -f "$BASE_ACT_PT" ] || { echo "Expected base activation not found: $BASE_ACT_PT"; exit 1; }

# ----------------------------------------------------------------------------
# Step 3 — embed each noised-corruption subdir into its own activation subdir
# ----------------------------------------------------------------------------
echo
echo "=== Step 3: Embed noised docs -> $ACT_NOISED_ROOT ==="
mkdir -p "$ACT_NOISED_ROOT"

python "$PIPELINE_DIR/embed_documents_from_dir.py" \
    --data-dir "$TXT_NOISED_DIR" \
    --out-dir  "$ACT_NOISED_ROOT" \
    --multi-dir \
    "${EMBED_FLAGS[@]}"

# ----------------------------------------------------------------------------
# Step 4 — Brenier potentials, source = base activations, NO PCA
# ----------------------------------------------------------------------------
echo
echo "=== Step 4: Brenier potentials -> $POTENTIALS_DIR ==="
python "$PIPELINE_DIR/multi_dir_brenier_embedding.py" \
    --activations_root "$ACT_NOISED_ROOT" \
    --output_root      "$POTENTIALS_DIR" \
    --source_path      "$BASE_ACT_PT" \
    --object           potential \
    --method           emd \
    --eps_reg          0.0 \
    --num_workers      "$NUM_WORKERS"

# ----------------------------------------------------------------------------
# Step 5 — train SAEs on potentials
# ----------------------------------------------------------------------------
echo
if [ "$SKIP_TRAIN" = true ]; then
    echo "=== Step 5: SKIPPED (--skip-train) ==="
else
    if [ "$FORCE_TRAIN" = false ] && dir_has_files "$SAE_OUT"; then
        echo "ERROR: SAE outputs already exist under $SAE_OUT"
        echo "Use --force-train with --train to intentionally write new models there, or set ROOT to a new run directory."
        exit 1
    fi
    echo "=== Step 5: Train SAEs -> $SAE_OUT ==="
    python "$PIPELINE_DIR/potential_SAE_analysis_script.py" \
        --data_dir   "$POTENTIALS_DIR" \
        --out_root   "$SAE_OUT" \
        --stack_dir  "$SAE_STACK" \
        --n_per_dir  -1 \
        --epochs     "$EPOCHS" \
        --lr         "$LR" \
        --batch_size "$BATCH_SIZE" \
        --device     "$DEVICE"
fi

echo
echo "Done. SAE checkpoints under $SAE_OUT/{TOPKAE_10,TOPKAE_40,JUMPRELUAE_10,JUMPRELUAE_40}/sparse_ae.pt"
