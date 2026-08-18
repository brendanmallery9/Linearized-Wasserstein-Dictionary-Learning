#!/usr/bin/env bash
#
# Smoke test for the minimal point-cloud ablation (LWDL-EOT vs PCA vs sparse coding).
#
# Runs the orchestrator end-to-end on a small stratified subsample with a tiny
# dictionary width and a capped Wasserstein sample count, so it finishes quickly
# and only exercises the plumbing -- it is NOT a meaningful benchmark.
#
# On success it writes:
#   <OUTPUT_DIR>/split_indices.json
#   <OUTPUT_DIR>/ablation_metrics.json
#   <OUTPUT_DIR>/ablation_table.csv
#   <OUTPUT_DIR>/ablation_table.tex
#
# Usage:
#   pointcloud/scripts/run_minimal_ablation_smoke.sh
#   PYTHON=.venv/bin/python DEVICE=cpu ./pointcloud/scripts/run_minimal_ablation_smoke.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-datasets/modelnet10_6cls_ot}"
LWDL_RESULTS_DIR="${LWDL_RESULTS_DIR:-pointcloud/results/modelnet10_6cls_uniform}"
OUTPUT_DIR="${OUTPUT_DIR:-pointcloud/results/minimal_ablation_smoke}"
DEVICE="${DEVICE:-auto}"
M="${M:-5}"
MAX_SAMPLES="${MAX_SAMPLES:-120}"
MAX_WASS="${MAX_WASS:-20}"
SPARSE_MAX_ITER="${SPARSE_MAX_ITER:-30}"

echo "=== minimal ablation smoke ==="
echo "data_dir:         $DATA_DIR"
echo "lwdl_results_dir: $LWDL_RESULTS_DIR"
echo "output_dir:       $OUTPUT_DIR"
echo "m=$M  max_samples=$MAX_SAMPLES  max_wasserstein_samples=$MAX_WASS"
echo ""

"$PYTHON" -u pointcloud/pipeline/run_pointcloud_minimal_ablation.py \
  --data_dir "$DATA_DIR" \
  --lwdl_results_dir "$LWDL_RESULTS_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --m "$M" \
  --max_samples "$MAX_SAMPLES" \
  --max_wasserstein_samples "$MAX_WASS" \
  --sparse_max_iter "$SPARSE_MAX_ITER" \
  --device "$DEVICE"

echo ""
echo "Smoke run complete. Table:"
cat "$OUTPUT_DIR/ablation_table.csv"
