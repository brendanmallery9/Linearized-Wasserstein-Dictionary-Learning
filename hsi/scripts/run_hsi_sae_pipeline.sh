#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
ROOT="${ROOT:-datasets/hsi_data}"
DATASET=""
EPOCHS="${EPOCHS:-1000}"
SOURCE_SUPP_SIZE="${SOURCE_SUPP_SIZE:-auto}"
SEEDS="${SEEDS:-0}"
SAE_MODE="${SAE_MODE:-transport_maps}"
SAE_PARAMS="${SAE_PARAMS:-}"
OT_WORKERS="${OT_WORKERS:-}"
LOG_DIR="${LOG_DIR:-logs}"
SKIP_DOWNLOAD=0
SKIP_MAPS=0
SKIP_TRAIN=1
FORCE_DOWNLOAD=0
FORCE_MAPS=0
FORCE_TRAIN=0

usage() {
  cat <<'EOF'
Usage:
  hsi/scripts/run_hsi_sae_pipeline.sh --dataset DATASET [options]

Pipeline:
  1. Download DATASET into ROOT/DATASET
  2. Compute 1-D OT/Brenier transport maps into ROOT/DATASET/transport_maps
  3. Optionally train SAE(s) into ROOT/DATASET/SAE_params/<mode>/unreg/seed_<seed>/

Required:
  --dataset NAME             One of: indian_pines, salinas_a, cuprite, botswana, pavia

Options:
  --root PATH                Dataset root. Default: datasets/hsi_data
  --epochs N                 SAE epochs. Default: 1000
  --source-supp-size N|auto  Transport-map support size. Default: auto, use cube band count
  --seeds "0 1 2"            Space-separated training seeds. Default: "0"
  --sae-mode MODE            transport_maps, linear, or both. Default: transport_maps
  --sae-params JSON          JSON dict passed to the SAE training wrapper
  --ot-workers N             Workers for transport maps. Default: script default
  --skip-download            Do not download/convert dataset
  --force-download           Download/convert even if ROOT/DATASET/data has a cube
  --skip-maps                Do not compute transport maps
  --force-maps               Recompute transport maps even if outputs exist
  --train                    Train SAEs. Default is to reuse existing SAE_params
  --force-train              Allow --train to write into existing SAE output dirs
  --skip-train               Do not train SAEs
  -h, --help                 Show this help

Environment overrides:
  PYTHON=python
  ROOT=datasets/hsi_data
  EPOCHS=1000
  SOURCE_SUPP_SIZE=auto
  SEEDS="0"
  SAE_MODE=transport_maps
  SAE_PARAMS='{"JUMPRELUAE_7_1e-3_mon":{...}}'
  OT_WORKERS=8
  LOG_DIR=logs
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --root)
      ROOT="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --source-supp-size)
      SOURCE_SUPP_SIZE="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --sae-mode)
      SAE_MODE="$2"
      shift 2
      ;;
    --sae-params)
      SAE_PARAMS="$2"
      shift 2
      ;;
    --ot-workers)
      OT_WORKERS="$2"
      shift 2
      ;;
    --skip-download)
      SKIP_DOWNLOAD=1
      shift
      ;;
    --force-download)
      FORCE_DOWNLOAD=1
      shift
      ;;
    --skip-maps)
      SKIP_MAPS=1
      shift
      ;;
    --force-maps)
      FORCE_MAPS=1
      shift
      ;;
    --train)
      SKIP_TRAIN=0
      shift
      ;;
    --force-train)
      FORCE_TRAIN=1
      shift
      ;;
    --skip-train)
      SKIP_TRAIN=1
      shift
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

if [[ -z "$DATASET" ]]; then
  echo "ERROR: --dataset is required." >&2
  usage >&2
  exit 2
fi

if [[ "$SAE_MODE" == "potentials" ]]; then
  echo "Note: --sae-mode potentials is deprecated for HSI; using transport_maps." >&2
  SAE_MODE="transport_maps"
fi

if [[ "$SAE_MODE" != "transport_maps" && "$SAE_MODE" != "linear" && "$SAE_MODE" != "both" ]]; then
  echo "ERROR: --sae-mode must be one of: transport_maps, linear, both. Got: $SAE_MODE" >&2
  exit 2
fi

DATASET_DIR="$ROOT/$DATASET"
DATA_DIR="$DATASET_DIR/data"
TRANSPORT_MAPS_DIR="$DATASET_DIR/transport_maps"
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1

default_hidden_dim() {
  case "$DATASET" in
    salinas_a) echo 7 ;;
    pavia) echo 10 ;;
    botswana) echo 15 ;;
    indian_pines) echo 17 ;;
    cuprite) echo 20 ;;
    *) echo 10 ;;
  esac
}

default_sae_params() {
  local mode="$1"
  local hdim
  hdim="$(default_hidden_dim)"
  if [[ "$mode" == "linear" ]]; then
    printf '{"JUMPRELUAE_%s_1e-3_nonneg":{"architecture":"JumpReLU_nonneg","l1":"1e-3","lr":1e-4,"hidden_dim":%s,"top_K":0}}' "$hdim" "$hdim"
  else
    printf '{"JUMPRELUAE_%s_1e-3_mon":{"architecture":"JumpReLU_monotone","l1":"1e-3","lr":1e-5,"hidden_dim":%s,"top_K":0}}' "$hdim" "$hdim"
  fi
}

cube_path() {
  local found
  found="$(find "$DATA_DIR" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | sort | head -n 1 || true)"
  [[ -n "$found" ]] && printf '%s\n' "$found"
}

dataset_assets_complete() {
  [[ -n "$(cube_path)" ]]
}

resolve_source_supp_size() {
  if [[ "$SOURCE_SUPP_SIZE" != "auto" ]]; then
    printf '%s\n' "$SOURCE_SUPP_SIZE"
    return
  fi
  local cube
  cube="$(cube_path)"
  if [[ -z "$cube" ]]; then
    echo "ERROR: Cannot infer source support size before dataset cube exists." >&2
    exit 1
  fi
  "$PYTHON" - "$cube" <<'PY'
import sys
import torch

obj = torch.load(sys.argv[1], map_location="cpu")
if isinstance(obj, dict):
    obj = obj.get("cube", obj.get("data"))
print(int(obj.shape[-1]))
PY
}

transport_maps_complete() {
  local cube
  cube="$(cube_path)"
  [[ -n "$cube" ]] || return 1
  local expected="$TRANSPORT_MAPS_DIR/$(basename "$cube")"
  [[ -f "$expected" ]]
}

sae_outputs_exist() {
  local out_root="$1"
  [[ -d "$out_root" ]] || return 1
  [[ -n "$(find "$out_root" -mindepth 1 -type f -print -quit 2>/dev/null)" ]]
}

run_download() {
  if [[ "$SKIP_DOWNLOAD" -eq 1 ]]; then
    echo "[download] skipped"
    return
  fi
  if [[ "$FORCE_DOWNLOAD" -eq 0 ]] && dataset_assets_complete; then
    echo "[download] existing dataset cube found; skipping"
    echo "  data_dir: $DATA_DIR"
    echo "  cube: $(cube_path)"
    return
  fi
  echo "[download] dataset=$DATASET root=$ROOT"
  "$PYTHON" -u hsi/pipeline/download_hyperspec_data.py \
    --root "$ROOT" \
    --datasets "$DATASET" \
    2>&1 | tee "$LOG_DIR/hsi_${DATASET}_download.log"
}

run_maps() {
  if [[ "$SKIP_MAPS" -eq 1 ]]; then
    echo "[maps] skipped"
    return
  fi
  if [[ "$FORCE_MAPS" -eq 0 ]] && transport_maps_complete; then
    echo "[maps] existing transport maps found; skipping"
    echo "  transport_maps_dir: $TRANSPORT_MAPS_DIR"
    return
  fi

  local supp_size
  supp_size="$(resolve_source_supp_size)"
  local cmd=(
    "$PYTHON" -u hsi/pipeline/compute_transport_maps.py
    --data_root "$DATA_DIR"
    --output_root "$TRANSPORT_MAPS_DIR"
    --source_supp_size "$supp_size"
  )
  if [[ -n "$OT_WORKERS" ]]; then
    cmd+=(--num_workers "$OT_WORKERS")
  fi
  if [[ "$FORCE_MAPS" -eq 1 ]]; then
    cmd+=(--overwrite)
  fi

  echo "[maps] computing transport maps"
  echo "  data_root: $DATA_DIR"
  echo "  output_root: $TRANSPORT_MAPS_DIR"
  echo "  source_supp_size: $supp_size"
  "${cmd[@]}" 2>&1 | tee "$LOG_DIR/hsi_${DATASET}_transport_maps.log"
}

train_one_mode() {
  local mode="$1"
  local train_data_dir
  local out_root
  local params

  if [[ "$mode" == "linear" ]]; then
    train_data_dir="$DATA_DIR"
    out_root="$DATASET_DIR/SAE_params/linear"
    params="${SAE_PARAMS:-$(default_sae_params linear)}"
  else
    train_data_dir="$TRANSPORT_MAPS_DIR"
    out_root="$DATASET_DIR/SAE_params/transport_maps"
    params="${SAE_PARAMS:-$(default_sae_params transport_maps)}"
  fi

  if [[ "$FORCE_TRAIN" -eq 0 ]] && sae_outputs_exist "$out_root"; then
    echo "ERROR: SAE outputs already exist under $out_root" >&2
    echo "Use --force-train to intentionally write new models there, or choose a different --root." >&2
    exit 1
  fi

  for seed in $SEEDS; do
    echo "[train] mode=$mode seed=$seed epochs=$EPOCHS"
    echo "  data_dir: $train_data_dir"
    echo "  out_root: $out_root"
    echo "  sae_params: $params"
    "$PYTHON" -u hsi/pipeline/train_hsi_sae.py \
      --out_root "$out_root" \
      --data_dir "$train_data_dir" \
      --epochs "$EPOCHS" \
      --seed "$seed" \
      --sae_params "$params" \
      2>&1 | tee "$LOG_DIR/hsi_${DATASET}_${mode}_sae_seed${seed}.log"
  done
}

run_training() {
  if [[ "$SKIP_TRAIN" -eq 1 ]]; then
    echo "[train] skipped"
    return
  fi
  if [[ "$FORCE_TRAIN" -eq 0 ]] && sae_outputs_exist "$DATASET_DIR/SAE_params"; then
    echo "ERROR: SAE outputs already exist under $DATASET_DIR/SAE_params" >&2
    echo "Use --force-train to intentionally write new models there, or choose a different --root." >&2
    exit 1
  fi
  case "$SAE_MODE" in
    transport_maps)
      train_one_mode transport_maps
      ;;
    linear)
      train_one_mode linear
      ;;
    both)
      train_one_mode transport_maps
      train_one_mode linear
      ;;
  esac
}

echo "=== HSI SAE pipeline ==="
echo "dataset: $DATASET"
echo "root: $ROOT"
echo "sae_mode: $SAE_MODE"
echo "seeds: $SEEDS"
echo "epochs: $EPOCHS"
echo ""

run_download
run_maps
run_training

echo ""
echo "Done."
echo "Dataset dir: $DATASET_DIR"
echo "Transport maps: $TRANSPORT_MAPS_DIR"
echo "SAEs:        $DATASET_DIR/SAE_params"
