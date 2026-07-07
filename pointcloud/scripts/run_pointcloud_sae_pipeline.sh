#!/usr/bin/env bash
#
# 3D point-cloud SAE pipeline -- analog of mnist/scripts/run_mnist_sae_pipeline.sh.
#
# Pipeline:
#   1. Prepare point-cloud OT maps (prepare_pointcloud_ot.py), unless complete
#      maps with matching prep parameters already exist at DATA_DIR.
#   2. Train the displacement-field (or raw-map) SAE
#      (pointcloud_run_experiments.py).
#   3. Evaluation is not yet automated for point clouds; analysis lives in
#      pointcloud/analysis/notebooks/pointcloud_WDL_analysis.ipynb.
#
# Defaults reproduce the modelnet10 6-class chair-base experiment that lives
# at datasets/modelnet10_6cls_chair_ot / pointcloud/results/modelnet10_6cls_chair.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"

# --- Data / I/O ---
DATASET="${DATASET:-modelnet10}"                                    # geometric_shapes | modelnet10
DATA_DIR="${DATA_DIR:-datasets/modelnet10_6cls_chair_ot}"
OUTPUT_DIR="${OUTPUT_DIR:-pointcloud/results/modelnet10_6cls_chair}"
LOG_DIR="${LOG_DIR:-logs}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"
PROGRESS_EVERY="${PROGRESS_EVERY:-5}"

# --- Map-prep params (must match what's recorded in DATA_DIR/metadata.json
#     for the maps_complete check to consider DATA_DIR up-to-date) ---
CLASSES="${CLASSES:-bed chair monitor sofa table toilet}"
BASE_SOURCE="${BASE_SOURCE:-class_mesh}"                            # uniform | class_mesh
BASE_CLASS="${BASE_CLASS:-chair}"                                   # only used if BASE_SOURCE=class_mesh
BASE_MESH_INDEX="${BASE_MESH_INDEX:-0}"
BASE_SUPP_SIZE="${BASE_SUPP_SIZE:-1000}"
CLOUD_SUPP_SIZE="${CLOUD_SUPP_SIZE:-1024}"
MAX_PER_CLASS="${MAX_PER_CLASS:-300}"
SAMPLES_PER_MESH="${SAMPLES_PER_MESH:-1}"
SPLIT="${SPLIT:-train}"
OT_METHOD="${OT_METHOD:-emd}"

# --- Model / training params ---
M="${M:-30}"
GRID_MODE="${GRID_MODE:-uniform}"                                   # uniform | cloud_mixture
GRID_SIDE="${GRID_SIDE:-17}"                                        # used iff GRID_MODE=uniform (17^3 = 4913 pts)
GRID_N_CLOUDS="${GRID_N_CLOUDS:-10}"                                # used iff GRID_MODE=cloud_mixture
GRID_SUPPORT_SIZE="${GRID_SUPPORT_SIZE:-3000}"                      # used iff GRID_MODE=cloud_mixture
LISTA_STEPS="${LISTA_STEPS:-20}"
EPOCHS="${EPOCHS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-0.001}"
EPS="${EPS:-0.025}"
C="${C:-0.0001}"
METHOD="${METHOD:-displacement}"                                    # displacement | raw_map
GPU_IDS="${GPU_IDS:-}"                                              # optional, e.g. "0" or "0 1 2 3"

# --- Modes ---
MAP_MODE="${MAP_MODE:-auto}"                                        # auto | skip | force
SKIP_TRAIN="${SKIP_TRAIN:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

usage() {
  cat <<'EOF'
Usage:
  pointcloud/scripts/run_pointcloud_sae_pipeline.sh [options]

Pipeline:
  1. Prepare OT maps from a base measure to each class's point clouds (unless
     complete matching maps already exist).
  2. Train a transport-map SAE on the resulting maps.
  3. Eval lives in pointcloud_WDL_analysis.ipynb (not part of this script).

Map-prep modes:
  --prepare-maps, --auto-maps  Prepare maps only if DATA_DIR is incomplete (default).
  --skip-maps                  Never run prepare_pointcloud_ot.py; require existing maps.
  --force-maps                 Always rerun prepare_pointcloud_ot.py before training.

Training modes:
  --skip-train                 Do not train (useful if you only want to (re)prep).
  --force-train                Allow training when OUTPUT_DIR already has outputs.

Options:
  --dataset {modelnet10,geometric_shapes}
  --data-dir PATH              Default: datasets/modelnet10_6cls_chair_ot
  --output-dir PATH            Default: pointcloud/results/modelnet10_6cls_chair
  --log-dir PATH               Default: logs
  --device auto|cuda|mps|cpu
  --gpu-ids "0 1 2"            Optional CUDA gpu ids (space-separated).
  --seed N
  --progress-every N

  --classes "bed chair ..."    Class names to include (default: 6 modelnet10 classes).
  --base-source uniform|class_mesh
                               'uniform': iid uniform on [0,1]^3 (default for non-pointcloud bases).
                               'class_mesh': surface-sample one mesh from --base-class.
  --base-class NAME            Class name to source the base mesh from (e.g. 'chair').
  --base-mesh-index N          Index into the sorted mesh list for --base-class.
  --base-supp-size N           Default: 1000
  --cloud-supp-size N          Default: 1024
  --max-per-class N            Default: 300
  --samples-per-mesh N         Default: 1
  --split train|test
  --ot-method emd|entropic

  --m N                        Dictionary atoms. Default: 30
  --grid-mode uniform|cloud_mixture
  --grid-side N                Cube grid side when --grid-mode uniform.
  --grid-n-clouds N
  --grid-support-size N
  --lista-steps N              Default: 20
  --epochs N                   Default: 2000
  --batch-size N               Default: 64
  --lr X                       Default: 0.001
  --eps X                      Default: 0.025
  --c X                        Sparsity coefficient. Default: 0.0001
  --method displacement|raw_map

  -h, --help                   Show this help

Environment overrides use the uppercase option names, e.g.:
  DEVICE=cuda EPOCHS=2000 ./pointcloud/scripts/run_pointcloud_sae_pipeline.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prepare-maps|--auto-maps) MAP_MODE="auto"; shift ;;
    --skip-maps)                MAP_MODE="skip"; shift ;;
    --force-maps)               MAP_MODE="force"; shift ;;
    --skip-train)               SKIP_TRAIN=1; shift ;;
    --force-train)              FORCE_TRAIN=1; shift ;;
    --dataset)                  DATASET="$2"; shift 2 ;;
    --data-dir)                 DATA_DIR="$2"; shift 2 ;;
    --output-dir)               OUTPUT_DIR="$2"; shift 2 ;;
    --log-dir)                  LOG_DIR="$2"; shift 2 ;;
    --device)                   DEVICE="$2"; shift 2 ;;
    --gpu-ids)                  GPU_IDS="$2"; shift 2 ;;
    --seed)                     SEED="$2"; shift 2 ;;
    --progress-every)           PROGRESS_EVERY="$2"; shift 2 ;;
    --classes)                  CLASSES="$2"; shift 2 ;;
    --base-source)              BASE_SOURCE="$2"; shift 2 ;;
    --base-class)               BASE_CLASS="$2"; shift 2 ;;
    --base-mesh-index)          BASE_MESH_INDEX="$2"; shift 2 ;;
    --base-supp-size)           BASE_SUPP_SIZE="$2"; shift 2 ;;
    --cloud-supp-size)          CLOUD_SUPP_SIZE="$2"; shift 2 ;;
    --max-per-class)            MAX_PER_CLASS="$2"; shift 2 ;;
    --samples-per-mesh)         SAMPLES_PER_MESH="$2"; shift 2 ;;
    --split)                    SPLIT="$2"; shift 2 ;;
    --ot-method)                OT_METHOD="$2"; shift 2 ;;
    --m)                        M="$2"; shift 2 ;;
    --grid-mode)                GRID_MODE="$2"; shift 2 ;;
    --grid-side)                GRID_SIDE="$2"; shift 2 ;;
    --grid-n-clouds)            GRID_N_CLOUDS="$2"; shift 2 ;;
    --grid-support-size)        GRID_SUPPORT_SIZE="$2"; shift 2 ;;
    --lista-steps)              LISTA_STEPS="$2"; shift 2 ;;
    --epochs)                   EPOCHS="$2"; shift 2 ;;
    --batch-size)               BATCH_SIZE="$2"; shift 2 ;;
    --lr)                       LR="$2"; shift 2 ;;
    --eps)                      EPS="$2"; shift 2 ;;
    --c)                        C="$2"; shift 2 ;;
    --method)                   METHOD="$2"; shift 2 ;;
    -h|--help)                  usage; exit 0 ;;
    *)                          echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$MAP_MODE" != "auto" && "$MAP_MODE" != "skip" && "$MAP_MODE" != "force" ]]; then
  echo "ERROR: MAP_MODE must be one of: auto, skip, force. Got: $MAP_MODE" >&2
  exit 2
fi
if [[ "$BASE_SOURCE" != "uniform" && "$BASE_SOURCE" != "class_mesh" ]]; then
  echo "ERROR: --base-source must be one of: uniform, class_mesh. Got: $BASE_SOURCE" >&2
  exit 2
fi
if [[ "$GRID_MODE" != "uniform" && "$GRID_MODE" != "cloud_mixture" ]]; then
  echo "ERROR: --grid-mode must be one of: uniform, cloud_mixture. Got: $GRID_MODE" >&2
  exit 2
fi
if [[ "$METHOD" != "displacement" && "$METHOD" != "raw_map" ]]; then
  echo "ERROR: --method must be one of: displacement, raw_map. Got: $METHOD" >&2
  exit 2
fi


# ============================================================
# Inline Python: check whether DATA_DIR already has matching maps.
#   Exit 0 if existing maps match the requested prep parameters,
#   exit non-zero otherwise.
# ============================================================
maps_complete() {
  "$PYTHON" - "$DATA_DIR" "$BASE_SUPP_SIZE" "$CLOUD_SUPP_SIZE" "$MAX_PER_CLASS" \
              "$SAMPLES_PER_MESH" "$SEED" "$BASE_SOURCE" "$BASE_CLASS" \
              "$BASE_MESH_INDEX" "$DATASET" "$SPLIT" "$OT_METHOD" \
              "$CLASSES" <<'PY'
import json, sys
from pathlib import Path
import torch

(data_dir, base_supp_size, cloud_supp_size, max_per_class,
 samples_per_mesh, seed, base_source, base_class, base_mesh_index,
 dataset, split, ot_method, classes_str) = sys.argv[1:14]

data_dir = Path(data_dir)
classes = classes_str.split() if classes_str.strip() else None

metadata_path = data_dir / "metadata.json"
base_path = data_dir / "base_measure.pt"
if not metadata_path.exists() or not base_path.exists():
    sys.exit(1)

try:
    metadata = json.loads(metadata_path.read_text())
    params = metadata["parameters"]
except Exception:
    sys.exit(1)

# Coerce numeric params before comparison
expected = {
    "base_supp_size":   int(base_supp_size),
    "cloud_supp_size":  int(cloud_supp_size),
    "max_per_class":    int(max_per_class),
    "samples_per_mesh": int(samples_per_mesh),
    "seed":             int(seed),
    "split":            split,
    "ot_method":        ot_method,
    "base_source":      base_source,
}
if base_source == "class_mesh":
    expected["base_class"]      = base_class
    expected["base_mesh_index"] = int(base_mesh_index)

for key, value in expected.items():
    if params.get(key) != value:
        sys.exit(1)

if metadata.get("dataset") != dataset:
    sys.exit(1)

# Verify base measure shape (n, d) where d is 2 or 3
try:
    base = torch.load(base_path, map_location="cpu")
except Exception:
    sys.exit(1)
if base.ndim != 2 or base.shape[0] != int(base_supp_size):
    sys.exit(1)
d = base.shape[1]

# Verify per-class mappings exist with correct shape
if classes is None:
    classes = sorted(metadata.get("classes", []))
if not classes:
    sys.exit(1)

class_counts = metadata.get("class_counts", {})
for c in classes:
    path = data_dir / f"class_{c}" / "mappings.pt"
    if not path.exists():
        sys.exit(1)
    try:
        m = torch.load(path, map_location="cpu")
    except Exception:
        sys.exit(1)
    expected_n = int(max_per_class) * int(samples_per_mesh)
    if tuple(m.shape) != (expected_n, int(base_supp_size), d):
        sys.exit(1)
    if int(class_counts.get(c, -1)) != expected_n:
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
  echo "[maps] computing point-cloud OT maps"
  echo "  dataset:           $DATASET"
  echo "  data_dir:          $DATA_DIR"
  echo "  classes:           $CLASSES"
  echo "  base_source:       $BASE_SOURCE"
  if [[ "$BASE_SOURCE" == "class_mesh" ]]; then
    echo "  base_class:        $BASE_CLASS"
    echo "  base_mesh_index:   $BASE_MESH_INDEX"
  fi
  echo "  base_supp_size:    $BASE_SUPP_SIZE"
  echo "  cloud_supp_size:   $CLOUD_SUPP_SIZE"
  echo "  max_per_class:     $MAX_PER_CLASS"
  echo "  samples_per_mesh:  $SAMPLES_PER_MESH"
  echo "  split / ot_method: $SPLIT / $OT_METHOD"

  prep_args=(
    --dataset "$DATASET"
    --output_dir "$DATA_DIR"
    --base_source "$BASE_SOURCE"
    --base_supp_size "$BASE_SUPP_SIZE"
    --cloud_supp_size "$CLOUD_SUPP_SIZE"
    --max_per_class "$MAX_PER_CLASS"
    --samples_per_mesh "$SAMPLES_PER_MESH"
    --split "$SPLIT"
    --ot_method "$OT_METHOD"
    --seed "$SEED"
    --progress_every "$PROGRESS_EVERY"
    --classes $CLASSES
  )
  if [[ "$BASE_SOURCE" == "class_mesh" ]]; then
    prep_args+=( --base_class "$BASE_CLASS" --base_mesh_index "$BASE_MESH_INDEX" )
  fi

  log_tag="$(basename "$DATA_DIR")"
  "$PYTHON" -u pointcloud/pipeline/prepare_pointcloud_ot.py "${prep_args[@]}" \
    2>&1 | tee "$LOG_DIR/prepare_pointcloud_ot_${log_tag}.log"
}


run_training() {
  if [[ "$SKIP_TRAIN" -eq 1 ]]; then
    echo "[train] skipped"
    return
  fi
  if [[ "$FORCE_TRAIN" -eq 0 ]] && train_outputs_exist; then
    echo "ERROR: OUTPUT_DIR already has training outputs: $OUTPUT_DIR" >&2
    echo "Use --force-train to overwrite, or pick another --output-dir." >&2
    exit 1
  fi

  echo "[train] point-cloud SAE"
  echo "  data_dir:    $DATA_DIR"
  echo "  output_dir:  $OUTPUT_DIR"
  echo "  device:      $DEVICE"
  if [[ -n "$GPU_IDS" ]]; then echo "  gpu_ids:     $GPU_IDS"; fi
  echo "  classes:     $CLASSES"
  echo "  method:      $METHOD"
  echo "  m:           $M"
  echo "  grid_mode:   $GRID_MODE"
  echo "  grid_side:   $GRID_SIDE"
  echo "  lista_steps: $LISTA_STEPS"
  echo "  epochs:      $EPOCHS"
  echo "  batch_size:  $BATCH_SIZE"
  echo "  lr:          $LR"
  echo "  eps:         $EPS"
  echo "  c:           $C"

  train_args=(
    --data_dir "$DATA_DIR"
    --output_dir "$OUTPUT_DIR"
    --classes $CLASSES
    --m "$M"
    --grid_mode "$GRID_MODE"
    --grid_side "$GRID_SIDE"
    --grid_n_clouds "$GRID_N_CLOUDS"
    --grid_support_size "$GRID_SUPPORT_SIZE"
    --lista_steps "$LISTA_STEPS"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --lr "$LR"
    --epsilons "$EPS"
    --sparsity_coeffs "$C"
    --methods "$METHOD"
    --device "$DEVICE"
    --seed "$SEED"
  )
  if [[ -n "$GPU_IDS" ]]; then
    train_args+=( --gpu_ids $GPU_IDS )
  fi

  log_tag="$(basename "$OUTPUT_DIR")"
  "$PYTHON" -u pointcloud/pipeline/pointcloud_run_experiments.py "${train_args[@]}" \
    2>&1 | tee "$LOG_DIR/train_pointcloud_SAE_${log_tag}.log"
}


mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1

echo "=== point-cloud SAE pipeline ==="
echo "dataset:    $DATASET"
echo "data_dir:   $DATA_DIR"
echo "output_dir: $OUTPUT_DIR"
echo "device:     $DEVICE"
echo "map_mode:   $MAP_MODE"
echo ""

case "$MAP_MODE" in
  auto)
    if maps_complete; then
      echo "[maps] existing matching point-cloud OT maps found; skipping"
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
      echo "ERROR: DATA_DIR is missing, incomplete, or does not match the requested prep params." >&2
      echo "Run without --skip-maps, or use --force-maps to rebuild." >&2
      exit 1
    fi
    ;;
  force)
    run_prepare_maps
    ;;
esac

run_training

echo ""
echo "Done."
echo "Checkpoint and metrics: $OUTPUT_DIR"
echo "Analysis lives in pointcloud/analysis/notebooks/pointcloud_WDL_analysis.ipynb."
