#!/usr/bin/env bash
#
# End-to-end minimal point-cloud ablation on a CUDA machine.
#
# A fresh clone of this repo has the CODE but NOT the data or the trained model
# (datasets/, pointcloud/results/, and pointcloud_raw/ are all gitignored).  So
# this script is self-contained: it (1) prepares the point-cloud OT dataset with
# raw clouds, (2) trains the LWDL-EOT model on the GPU, and (3) runs the
# PCA / sparse-coding / LWDL ablation on the GPU.  Steps that already
# have their outputs on disk are skipped, so re-runs are cheap.
#
# Usage:
#   pointcloud/scripts/run_minimal_ablation_cuda.sh
#   EPOCHS=2000 M=20 ./pointcloud/scripts/run_minimal_ablation_cuda.sh
#   PYTHON=.venv/bin/python DEVICE=cuda ./pointcloud/scripts/run_minimal_ablation_cuda.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
M="${M:-20}"
EPOCHS="${EPOCHS:-1000}"                 # LWDL training epochs
SEED="${SEED:-42}"
CLASSES="${CLASSES:-bed chair monitor sofa table toilet}"

DATA_DIR="${DATA_DIR:-datasets/modelnet10_6cls_ot_withraw}"
LWDL_DIR="${LWDL_DIR:-pointcloud/results/modelnet10_6cls_m${M}}"
OUT_DIR="${OUT_DIR:-pointcloud/results/minimal_ablation_cuda}"

echo "=== minimal ablation (CUDA) ==="
echo "python:   $PYTHON"
echo "device:   $DEVICE"
echo "m:        $M    epochs: $EPOCHS"
echo "data_dir: $DATA_DIR"
echo "lwdl_dir: $LWDL_DIR"
echo "out_dir:  $OUT_DIR"
echo ""

# 0. Sanity-check the GPU (fail loudly rather than silently using CPU).
if [ "$DEVICE" = "cuda" ]; then
  "$PYTHON" - <<'PY'
import torch, sys
if not torch.cuda.is_available():
    sys.exit("ERROR: torch.cuda.is_available() is False -- install a CUDA build "
             "of torch, or re-run with DEVICE=cpu.")
print(f"CUDA OK: {torch.cuda.get_device_name(0)} (torch {torch.__version__})")
PY
fi

# 1. Prepare the OT dataset WITH raw clouds (downloads ModelNet10 on first run).
if [ ! -f "$DATA_DIR/class_chair/raw_clouds.pt" ]; then
  echo ">>> [1/3] Preparing dataset with raw clouds -> $DATA_DIR"
  "$PYTHON" -u pointcloud/pipeline/prepare_pointcloud_ot.py \
    --dataset modelnet10 --output_dir "$DATA_DIR" \
    --base_source uniform --base_supp_size 1000 --cloud_supp_size 1024 \
    --max_per_class 300 --samples_per_mesh 1 --split train --ot_method emd \
    --seed "$SEED" --progress_every 50 --classes $CLASSES
else
  echo ">>> [1/3] Dataset already present ($DATA_DIR); skipping prep."
fi

# 2. Train the LWDL-EOT model (m atoms) on the GPU.
if [ ! -f "$LWDL_DIR/config.json" ]; then
  echo ">>> [2/3] Training LWDL-EOT (m=$M, epochs=$EPOCHS) -> $LWDL_DIR"
  "$PYTHON" -u pointcloud/pipeline/pointcloud_run_experiments.py \
    --data_dir "$DATA_DIR" --output_dir "$LWDL_DIR" \
    --m "$M" --epochs "$EPOCHS" --seed "$SEED" \
    --device "$DEVICE" --classes $CLASSES
else
  echo ">>> [2/3] LWDL checkpoint already present ($LWDL_DIR); skipping training."
fi

# 3. Run the ablation (PCA / sparse coding / LWDL) on the GPU.
echo ">>> [3/3] Running ablation -> $OUT_DIR"
"$PYTHON" -u pointcloud/pipeline/run_pointcloud_minimal_ablation.py \
  --data_dir "$DATA_DIR" --lwdl_results_dir "$LWDL_DIR" \
  --output_dir "$OUT_DIR" --m "$M" --seed "$SEED" \
  --device "$DEVICE"

echo ""
echo "Done. Table:"
cat "$OUT_DIR/ablation_table.csv"
