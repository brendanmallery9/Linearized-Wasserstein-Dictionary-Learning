"""
Multi-trial reconstruction MSE: SAE (linear + transport maps) vs NMF on Pavia.

Evaluates clean data plus ALL corruption types swept over a range of strengths,
with MSE averaged over multiple corruption RNG seeds (in addition to model seeds).

Corruption types run automatically:
    drop_random, drop_contiguous, shift_global, shift_split, log_warp

Output structure
----------------
JSON:
  conditions:
    "clean":                   { aggregated_sae, aggregated_nmf, raw }
    "drop_random_k0.1":        { aggregated_sae, aggregated_nmf, raw }  <- avg over cseed
    "drop_random_k0.2":        ...
    ...
    "log_warp_k0.5":           ...

PNG:
  One subtable per corruption type (+ one for clean).
  Rows = models.  Columns = strengths (0.1 … 0.5), plus a "clean" column.
  MSE values are mean ± std over BOTH model seeds and corruption seeds.

EXAMPLES
--------
# Default: all types, strengths 0.1-0.5, corruption seeds 0 1 2
python reconstruction_mse_multitrial.py \
    --root datasets/hsi_data \
    --seeds 0 1 2 3 4 \
    --device cpu \
    --output_json recon_mse_sweep.json \
    --output_png  recon_mse_sweep.png

# Custom strengths and corruption seeds
python reconstruction_mse_multitrial.py \\
    --root datasets/hsi_data \\
    --seeds 0 1 2 3 4 \\
    --corruption_strengths 0.1 0.3 0.5 \\
    --corruption_seeds 0 1 2 3 4 \\
    --device cpu \\
    --output_json recon_mse_sweep.json \\
    --output_png  recon_mse_sweep.png

# Specific keys, skip NMF
python reconstruction_mse_multitrial.py \\
    --root datasets/hsi_data \\
    --seeds 0 1 2 3 4 \\
    --potential_keys JUMPRELUAE_10_1e-2_mon \\
    --linear_keys   JUMPRELUAE_10_1e-2_nonneg \\
    --skip_nmf \\
    --device cpu \\
    --output_json recon_mse_sweep.json \\
    --output_png  recon_mse_sweep.png
"""

import argparse
import json
import time as _time
from collections import defaultdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
HSI_PIPELINE_DIR = REPO_ROOT / "hsi" / "pipeline"
if str(HSI_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(HSI_PIPELINE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import NMF

from brenier_embedding_functions import wass_map_1D
from hsi_utils import (
    SAE_PARAMETERS,
    load_model,
    model_path_for,
    drop_random_bands,
    drop_contiguous_bands,
    shift_mass_global,
    shift_mass_random_split,
    pushforward_log_warp,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_CORRUPTION_TYPES = [
    "drop_random",
    "drop_contiguous",
    "shift_global",
    "shift_split",
    "log_warp",
]

DEFAULT_STRENGTHS = [0.1, 0.2, 0.3, 0.4, 0.5]


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def resolve_device(device_str=None):
    if device_str is not None:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def find_single_pt(directory):
    pts = list(Path(directory).glob("*.pt"))
    if not pts:
        raise FileNotFoundError(f"No .pt file found in {directory}")
    if len(pts) > 1:
        raise RuntimeError(f"Multiple .pt files in {directory}: {pts}")
    return str(pts[0])


def discover_datasets(root):
    return [
        d.name
        for d in sorted(Path(root).iterdir(), key=lambda x: x.name)
        if d.is_dir() and (d / "data").is_dir()
    ]


def discover_keys(root, dataset, seed, sae_mode, subdir="unreg"):
    if subdir:
        seed_dir = Path(root) / dataset / "SAE_params" / sae_mode / subdir / f"seed_{seed}"
    else:
        seed_dir = Path(root) / dataset / "SAE_params" / sae_mode / f"seed_{seed}"
    if not seed_dir.is_dir():
        print(f"  [discover_keys] Path does not exist: {seed_dir}")
        return []
    return sorted(k.name for k in seed_dir.iterdir() if k.is_dir())


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_flat_cube(pt_path):
    raw = torch.load(pt_path, map_location="cpu")
    cube = raw if torch.is_tensor(raw) else raw["cube"]
    a, b, c = cube.shape
    return cube.reshape(a * b, c).float()


def train_val_split(flat_data, seed=0, val_frac=0.05):
    """Reproducible 95/5 split identical to the SAE training script."""
    torch.manual_seed(seed)
    n = flat_data.shape[0]
    n_val = int(val_frac * n)
    perm = torch.randperm(n)
    return flat_data[perm[n_val:]], flat_data[perm[:n_val]], perm[n_val:], perm[:n_val]


# ---------------------------------------------------------------------------
# Corruption
# ---------------------------------------------------------------------------

def apply_corruption(data_np, corruption_type, k, rng=None):
    """Apply a corruption to (N, M) float64 array. Returns (corrupted, info)."""
    if rng is None:
        rng = np.random.default_rng()
    data_np = np.asarray(data_np, dtype=np.float64)
    M = data_np.shape[1]  # number of spectral bands

    if corruption_type == "drop_random":
        n_drop = max(1, int(round(k * M)))  # convert fraction -> integer band count
        corrupted, _ = drop_random_bands(data_np, n_drop, rng=rng)
        info = {"type": corruption_type, "k_frac": k, "n_bands_dropped": n_drop}
    elif corruption_type == "drop_contiguous":
        n_drop = max(1, int(round(k * M)))
        corrupted, _ = drop_contiguous_bands(data_np, n_drop, rng=rng)
        info = {"type": corruption_type, "k_frac": k, "n_bands_dropped": n_drop}
    elif corruption_type == "shift_global":
        corrupted, info = shift_mass_global(data_np, frac=k, rng=rng)
        info.update({"type": corruption_type, "k_frac": k})
    elif corruption_type == "shift_split":
        corrupted, info = shift_mass_random_split(data_np, frac=k, rng=rng)
        info.update({"type": corruption_type, "k_frac": k})
    elif corruption_type == "log_warp":
        corrupted = np.stack([pushforward_log_warp(data_np[i], a=k)
                              for i in range(data_np.shape[0])])
        info = {"type": corruption_type, "a": k}
    else:
        raise ValueError(f"Unknown corruption type '{corruption_type}'.")
    return corrupted, info


# ---------------------------------------------------------------------------
# OT map computation + caching
# ---------------------------------------------------------------------------

def _ot_cache_path(ds_dir, tag="clean"):
    cache_dir = Path(ds_dir) / ".ot_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{tag}.npy"


def compute_ot_maps(data_np, source_supp_size, print_prefix="OT"):
    N, D = data_np.shape
    source_grid = np.linspace(0.0, 1.0, source_supp_size)
    target_grid = np.linspace(0.0, 1.0, D)
    ot_maps = np.empty((N, source_supp_size), dtype=np.float64)
    print_every = max(1, N // 20)
    t0 = _time.time()
    for i in range(N):
        if i % print_every == 0 or i == N - 1:
            elapsed = _time.time() - t0
            eta = (N - i - 1) / max((i + 1) / max(elapsed, 1e-9), 1e-9)
            print(f"  {print_prefix}: {i+1}/{N} ({100*(i+1)/N:.1f}%) "
                  f"elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)
        ot_maps[i] = np.asarray(
            wass_map_1D(source_grid, None, target_grid, data_np[i])
        ).reshape(-1)
    print(f"  {print_prefix}: done ({N} maps in {_time.time()-t0:.1f}s)", flush=True)
    return ot_maps


def load_or_compute_ot(data_np, source_supp_size, cache_path=None,
                       print_prefix="OT", force=False):
    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and not force and cache_path.exists():
        ot = np.load(cache_path)
        if ot.shape == (data_np.shape[0], source_supp_size):
            print(f"  {print_prefix}: loaded from cache {cache_path}", flush=True)
            return ot
        print(f"  {print_prefix}: cache shape mismatch, recomputing...", flush=True)
    ot = compute_ot_maps(data_np, source_supp_size, print_prefix=print_prefix)
    if cache_path is not None:
        try:
            np.save(cache_path, ot)
            print(f"  {print_prefix}: cached to {cache_path}", flush=True)
        except Exception as e:
            print(f"  {print_prefix}: WARNING could not write cache: {e}", flush=True)
    return ot


# ---------------------------------------------------------------------------
# MSE
# ---------------------------------------------------------------------------

def mse_numpy(x_true, x_hat):
    diff = np.asarray(x_true, np.float64) - np.asarray(x_hat, np.float64)
    return float(np.mean(diff ** 2))


# ---------------------------------------------------------------------------
# SAE reconstruction
# ---------------------------------------------------------------------------

def sae_reconstruct(val_tensor, key, sae_root, device, batch_size=1024):
    """Encode + decode val_tensor. input_dim inferred from val_tensor.shape[1]."""
    cfg = SAE_PARAMETERS[key]
    model = load_model(
        cfg["architecture"],
        cfg["hidden_dim"],
        model_path_for(key, sae_root),
        top_k=cfg.get("top_K"),
    )
    model.eval().to(device)
    parts = []
    with torch.no_grad():
        for start in range(0, val_tensor.shape[0], batch_size):
            batch = val_tensor[start:start + batch_size].to(device).float()
            out = model(batch)
            parts.append((out[0] if isinstance(out, (tuple, list)) else out).cpu().float())
    x_hat  = torch.cat(parts, dim=0).numpy().astype(np.float64)
    x_true = val_tensor.numpy().astype(np.float64)
    return mse_numpy(x_true, x_hat)


# ---------------------------------------------------------------------------
# NMF reconstruction
# ---------------------------------------------------------------------------

def _row_normalize(x_np):
    """L1-normalize each row (spectrum). Rows summing to 0 are left as-is."""
    sums = x_np.sum(axis=1, keepdims=True)
    sums = np.where(sums == 0, 1.0, sums)
    return x_np / sums


def nmf_reconstruct_val(train_data, val_data, rank, H_dict=None):
    """Fit NMF on train (or reuse H_dict), project val.

    Inputs are L1-normalised per spectrum before fitting (matching the
    approach used in exploratory NMF notebooks).  Val projection uses
    model.transform which is faster and more numerically stable than
    non_negative_factorization with update_H=False.

    Returns (x_hat_val_unnorm, mse_against_unnorm_val, H_dict).
    MSE is computed in the original (unnormalised) space so it is
    comparable to SAE reconstruction MSE.
    """
    to_np = lambda t: (t.numpy() if torch.is_tensor(t) else np.asarray(t)).astype(np.float64)
    train_np = np.maximum(to_np(train_data), 0)
    val_np   = np.maximum(to_np(val_data),   0)

    # L1-normalise
    train_norm = _row_normalize(train_np)
    val_norm   = _row_normalize(val_np)

    if H_dict is None:
        m = NMF(n_components=rank, init="nndsvda", max_iter=1200, random_state=0)
        m.fit_transform(train_norm)
        H_dict = m.components_   # (rank, C) — fitted in normalised space
    else:
        # Wrap H_dict in a dummy NMF so we can call .transform
        m = NMF(n_components=rank, init="custom", max_iter=1, random_state=0)
        # Fit for 1 iter to initialise internal state, then overwrite components_
        m.fit_transform(train_norm, H=H_dict.astype(train_norm.dtype),
                        W=np.ones((train_norm.shape[0], rank), dtype=train_norm.dtype))
        m.components_ = H_dict.astype(train_norm.dtype)

    W_val = m.transform(val_norm)          # fast NNLS projection, no H update
    x_hat_val_norm = W_val @ H_dict

    # Rescale back to original units using val row sums
    val_sums = to_np(val_data).sum(axis=1, keepdims=True)
    val_sums = np.where(val_sums == 0, 1.0, val_sums)
    x_hat_val = x_hat_val_norm * val_sums

    val_np_orig = to_np(val_data)
    return x_hat_val, mse_numpy(val_np_orig, x_hat_val), H_dict


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_mse(mse_list):
    """mse_list: list of floats (one per model-seed × corruption-seed) -> summary."""
    vals = np.array(mse_list, dtype=np.float64)
    return {
        "mse_mean": float(np.mean(vals)),
        "mse_std":  float(np.std(vals)),
        "mse_var":  float(np.var(vals)),
        "n_trials": int(len(vals)),
    }


# ---------------------------------------------------------------------------
# Key resolution
# ---------------------------------------------------------------------------

def _resolve_valid_keys(root, dataset, seeds, sae_mode, requested_keys, subdir="unreg"):
    discovered = set()
    for s in seeds:
        discovered.update(discover_keys(root, dataset, s, sae_mode, subdir=subdir))
    if requested_keys is not None:
        allowed = set(k.strip() for k in requested_keys)
        valid   = sorted(allowed & discovered & set(SAE_PARAMETERS))
        skipped = sorted(allowed - discovered)
        if skipped:
            print(f"  [{sae_mode}] Keys not found on disk for '{dataset}': {skipped}")
            print(f"  [{sae_mode}] Keys discovered on disk: {sorted(discovered)}")
    else:
        valid = sorted(discovered & set(SAE_PARAMETERS))
    return valid


# ---------------------------------------------------------------------------
# Per-condition evaluation
# (returns raw per-trial MSE dicts, aggregation happens at condition level)
# ---------------------------------------------------------------------------

def eval_sae_on_val(dataset, ds_dir, sae_mode, valid_keys, seeds,
                    val_tensor, device, batch_size, subdir="unreg"):
    """Run SAE reconstruction for all (key, seed) combos on val_tensor.

    Returns {(dataset, label): [mse_float, ...]}  — one entry per model seed.
    """
    results = defaultdict(list)   # (dataset, label) -> [mse, ...]
    n_total = len(seeds) * len(valid_keys)
    n_done  = 0
    for seed in seeds:
        if subdir:
            sae_root = str(ds_dir / "SAE_params" / sae_mode / subdir / f"seed_{seed}")
        else:
            sae_root = str(ds_dir / "SAE_params" / sae_mode / f"seed_{seed}")
        for key in valid_keys:
            if not (Path(sae_root) / key).is_dir():
                n_done += 1
                continue
            n_done += 1
            print(f"  [{sae_mode}] SAE {n_done}/{n_total}: {key} seed={seed}", flush=True)
            mse = sae_reconstruct(val_tensor, key=key, sae_root=sae_root,
                                  device=device, batch_size=batch_size)
            print(f"  [{sae_mode}]   -> MSE = {mse:.6f}", flush=True)
            label = f"{sae_mode}_{key}"
            results[(dataset, label)].append(mse)
    return results


def eval_nmf_on_val(dataset, ds_dir, sae_mode, train_data, val_data,
                    unique_ranks, use_nmf_cache, clean_H_dicts):
    """Evaluate NMF on val_data using the fixed clean dictionary.

    Returns {(dataset, label): mse_float}
    """
    nmf_cache_dir = ds_dir / f".nmf_recon_cache_{sae_mode}" if use_nmf_cache else None
    if nmf_cache_dir is not None:
        nmf_cache_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for rank in unique_ranks:
        cache_file = nmf_cache_dir / f"H_dict_rank{rank}.npy" if nmf_cache_dir is not None else None

        H_dict = clean_H_dicts.get(rank, None)
        if H_dict is None:
            if use_nmf_cache and cache_file.exists():
                print(f"  [{sae_mode}] Loading cached NMF H_dict (rank={rank})")
                H_dict = np.load(cache_file)

        _, mse_val, H_fitted = nmf_reconstruct_val(train_data, val_data, rank, H_dict=H_dict)

        # Cache and store the clean dictionary on first fit
        if H_dict is None:
            if use_nmf_cache:
                try:
                    np.save(cache_file, H_fitted)
                    print(f"  [{sae_mode}] Cached NMF H_dict (rank={rank}) to {cache_file}")
                except Exception as e:
                    print(f"  [{sae_mode}] WARNING: could not cache NMF dict: {e}")
            clean_H_dicts[rank] = H_fitted

        label = f"{sae_mode}_rank{rank}"
        results[(dataset, label)] = mse_val
    return results


# ---------------------------------------------------------------------------
# PNG rendering
# ---------------------------------------------------------------------------

def _condition_key(ctype, k):
    """Canonical string key for a (corruption_type, strength) condition."""
    return f"{ctype}_k{k:.2g}"


def render_mse_table_png(
    model_labels,       # ordered list of row labels (model names)
    clean_agg,          # {model_label: {mse_mean, mse_std}}
    corrupt_agg,        # {(ctype, k): {model_label: {mse_mean, mse_std}}}
    strengths,          # list of float strengths
    output_path,
):
    """One subtable per corruption type, showing strengths as columns.

    Rows = models.  First column = clean MSE.  Remaining columns = strengths.
    Best (lowest) MSE per row is highlighted green.
    """
    n_types  = len(ALL_CORRUPTION_TYPES)
    n_cols   = 1 + len(strengths)           # clean + strengths
    col_hdrs = ["clean"] + [f"k={k:.2g}" for k in strengths]

    fig_h = max(6, 0.35 * len(model_labels) * n_types + 2 * n_types)
    fig, axes = plt.subplots(n_types, 1, figsize=(max(10, 2 * n_cols), fig_h))
    if n_types == 1:
        axes = [axes]

    for ax, ctype in zip(axes, ALL_CORRUPTION_TYPES):
        ax.axis("off")
        ax.set_title(f"Corruption: {ctype}", fontsize=11, pad=8)

        cell_text = []
        bold_mask = []

        for mlabel in model_labels:
            row_vals = []
            # Clean column
            clean_entry = clean_agg.get(mlabel, {})
            c_mean = clean_entry.get("mse_mean", float("nan"))
            c_std  = clean_entry.get("mse_std",  float("nan"))
            row_vals.append((c_mean, c_std))
            # Corruption columns
            for k in strengths:
                entry = corrupt_agg.get((ctype, k), {}).get(mlabel, {})
                row_vals.append((
                    entry.get("mse_mean", float("nan")),
                    entry.get("mse_std",  float("nan")),
                ))

            # Best (lowest mean) across columns for this row
            valid_means = [v[0] for v in row_vals if not np.isnan(v[0])]
            best = min(valid_means) if valid_means else float("nan")

            row_text, row_bold = [], []
            for mean, std in row_vals:
                if np.isnan(mean):
                    row_text.append("—")
                elif std == 0.0 or np.isnan(std):
                    row_text.append(f"{mean:.5f}")
                else:
                    row_text.append(f"{mean:.5f}\n±{std:.5f}")
                row_bold.append(not np.isnan(mean) and abs(mean - best) < 1e-12)

            cell_text.append(row_text)
            bold_mask.append(row_bold)

        table = ax.table(
            cellText=cell_text,
            rowLabels=model_labels,
            colLabels=col_hdrs,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 2.0)

        # Header styling
        for j in range(n_cols):
            c = table[0, j]
            c.set_text_props(fontweight="bold", fontsize=7)
            c.set_facecolor("#d9e2f3" if j > 0 else "#c6efce")

        # Row label + cell styling
        for i, mlabel in enumerate(model_labels):
            lc = table[i + 1, -1]
            lc.set_text_props(fontsize=6)
            lc.set_facecolor("#f2dcdb" if "NMF" in mlabel else "#e2efda")
            for j in range(n_cols):
                if bold_mask[i][j]:
                    c = table[i + 1, j]
                    c.set_text_props(fontweight="bold", color="darkgreen")
                    c.set_facecolor("#ffffcc")

    fig.suptitle("Reconstruction MSE — Pavia val split (mean ± std over model seeds × corruption seeds)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Table PNG saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep all corruption types × strengths on Pavia.\n"
            "MSE is averaged over both model seeds and corruption seeds."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="SAE training seeds (default: 0-4)")
    parser.add_argument(
        "--potential_keys", type=str, nargs="+", default=None, metavar="KEY",
        help="SAE keys in OT-potential space. Omit for all discovered.",
    )
    parser.add_argument(
        "--linear_keys", type=str, nargs="+", default=None, metavar="KEY",
        help="SAE keys on raw spectra. Omit for all discovered.",
    )
    parser.add_argument(
        "--potential_subdir", type=str, default="unreg", metavar="SUBDIR",
    )
    parser.add_argument(
        "--linear_subdir", type=str, default="unreg", metavar="SUBDIR",
    )
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device. Pass 'cpu' to force CPU.",
    )
    parser.add_argument(
        "--corruption_strengths", type=float, nargs="+",
        default=DEFAULT_STRENGTHS,
        metavar="K",
        help="Corruption strengths to sweep (default: 0.1 0.2 0.3 0.4 0.5).",
    )
    parser.add_argument(
        "--corruption_seeds", type=int, nargs="+", default=[0, 1, 2],
        metavar="SEED",
        help="RNG seeds for corruption, averaged over (default: 0 1 2).",
    )
    parser.add_argument("--use_ot_cache", action="store_true",
                        help="Persist OT caches under each dataset directory. Default is off.")
    parser.add_argument("--use_nmf_cache", action="store_true",
                        help="Persist NMF caches under each dataset directory. Default is off.")
    parser.add_argument("--no_ot_cache", action="store_true",
                        help="Deprecated alias for the default no-cache behavior.")
    parser.add_argument("--no_nmf_cache", action="store_true",
                        help="Deprecated alias for the default no-cache behavior.")
    parser.add_argument("--skip_nmf", action="store_true",
                        help="Skip NMF baselines entirely.")
    parser.add_argument("--output_json", default="recon_mse_sweep.json")
    parser.add_argument("--output_png",  default="recon_mse_sweep.png")
    args = parser.parse_args()
    use_ot_cache = args.use_ot_cache and not args.no_ot_cache
    use_nmf_cache = args.use_nmf_cache and not args.no_nmf_cache

    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Corruption types:    {ALL_CORRUPTION_TYPES}")
    print(f"Corruption strengths: {args.corruption_strengths}")
    print(f"Corruption seeds:     {args.corruption_seeds}")

    root = Path(args.root)
    all_datasets = discover_datasets(root)
    datasets = [d for d in all_datasets if d == "Pavia"]
    if not datasets:
        raise RuntimeError(f"'Pavia' not found under {root}. Available: {all_datasets}")
    print(f"Running on datasets: {datasets}")

    subdir_map = {
        "transport_maps": args.potential_subdir,
        "linear":     args.linear_subdir,
    }

    # Final aggregated outputs
    # clean_agg  : {model_label -> {mse_mean, mse_std, n_trials}}
    # corrupt_agg: {(ctype, k)  -> {model_label -> {mse_mean, mse_std, n_trials}}}
    clean_agg  = {}
    corrupt_agg = defaultdict(dict)   # (ctype, k) -> {model_label -> agg}

    # Raw results for JSON
    all_raw = []

    # -----------------------------------------------------------------------
    for dataset in datasets:
        ds_dir       = root / dataset
        og_pt        = find_single_pt(ds_dir / "data")
        potential_pt = find_single_pt(ds_dir / "transport_maps")

        print(f"\n{'=' * 70}")
        print(f"Dataset: {dataset}")
        print(f"{'=' * 70}")

        # Load raw cube; fix split indices at seed 0
        flat_og = load_flat_cube(og_pt)
        og_train, og_val, train_idx, val_idx = train_val_split(
            flat_og, seed=0, val_frac=args.val_frac
        )
        print(f"  Train: {og_train.shape}, Val: {og_val.shape}")

        # Load precomputed clean OT maps and apply same split
        flat_pot  = load_flat_cube(potential_pt)
        pot_train = flat_pot[train_idx]
        pot_val   = flat_pot[val_idx]

        # Resolve valid keys (same for all conditions)
        mode_valid_keys = {}
        for sae_mode, req_keys in [("transport_maps", args.potential_keys),
                                   ("linear",     args.linear_keys)]:
            vk = _resolve_valid_keys(
                root, dataset, args.seeds, sae_mode, req_keys,
                subdir=subdir_map[sae_mode],
            )
            mode_valid_keys[sae_mode] = vk
            print(f"  [{sae_mode}] Valid keys ({len(vk)}): {vk}")

        unique_ranks_by_mode = {
            sae_mode: sorted({SAE_PARAMETERS[k]["hidden_dim"] for k in vk})
            for sae_mode, vk in mode_valid_keys.items() if vk
        }

        # NMF dictionaries fitted once on clean data, reused for all conditions
        clean_H_dicts_by_mode = {"transport_maps": {}, "linear": {}}

        # ----------------------------------------------------------------
        # CLEAN evaluation
        # ----------------------------------------------------------------
        print(f"\n  {'─' * 30} CLEAN {'─' * 30}")

        # Accumulate MSE lists per model across model seeds
        clean_sae_lists  = defaultdict(list)   # model_label -> [mse, ...]
        clean_nmf_values = {}                  # model_label -> mse (deterministic)

        for sae_mode, (train_data, val_data) in [
            ("transport_maps", (pot_train, pot_val)),
            ("linear",     (og_train,  og_val)),
        ]:
            valid_keys   = mode_valid_keys.get(sae_mode, [])
            unique_ranks = unique_ranks_by_mode.get(sae_mode, [])

            if not valid_keys:
                continue

            # NMF (deterministic — no model seeds)
            if not args.skip_nmf:
                print(f"\n  Evaluating NMF [{sae_mode}] on clean val...")
                nmf_res = eval_nmf_on_val(
                    dataset, ds_dir, sae_mode,
                    train_data, val_data,
                    unique_ranks, use_nmf_cache,
                    clean_H_dicts_by_mode[sae_mode],
                )
                for (ds, lbl), mse in nmf_res.items():
                    clean_nmf_values[lbl] = mse
                    print(f"    NMF [{sae_mode}] rank={lbl.split('_rank')[1]} MSE={mse:.6f}")

            # SAE
            print(f"\n  Evaluating SAE [{sae_mode}] on clean val...")
            sae_res = eval_sae_on_val(
                dataset, ds_dir, sae_mode, valid_keys,
                args.seeds, val_data, device, args.batch_size,
                subdir=subdir_map[sae_mode],
            )
            for (ds, lbl), mse_list in sae_res.items():
                clean_sae_lists[lbl].extend(mse_list)

        # Aggregate clean results
        for lbl, mse_list in clean_sae_lists.items():
            agg = aggregate_mse(mse_list)
            clean_agg[lbl] = agg
            all_raw.append({"condition": "clean", "model": lbl,
                            "dataset": dataset, **agg})

        for lbl, mse in clean_nmf_values.items():
            agg = aggregate_mse([mse])
            clean_agg[lbl] = agg
            all_raw.append({"condition": "clean", "model": lbl,
                            "dataset": dataset, **agg})

        # ----------------------------------------------------------------
        # CORRUPTION sweep: all types × strengths × corruption_seeds
        # ----------------------------------------------------------------
        for ctype in ALL_CORRUPTION_TYPES:
            for k in args.corruption_strengths:
                print(f"\n  {'─' * 20} {ctype} k={k:.2g} {'─' * 20}")

                # Accumulate over corruption seeds
                # SAE: lists per (model_label, cseed, model_seed)
                # NMF: list per (model_label, cseed) — deterministic per cseed
                corrupt_sae_lists  = defaultdict(list)
                corrupt_nmf_values = defaultdict(list)

                for cseed in args.corruption_seeds:
                    rng = np.random.default_rng(cseed)

                    # Corrupt the FULL array, then re-split by stored indices
                    full_og_np = flat_og.numpy().astype(np.float64)
                    full_corrupted_np, _ = apply_corruption(
                        full_og_np, ctype, k, rng=rng
                    )
                    train_og_c = torch.from_numpy(
                        full_corrupted_np[train_idx.numpy()]).float()
                    val_og_c   = torch.from_numpy(
                        full_corrupted_np[val_idx.numpy()]).float()

                    # OT maps for corrupted transport-map mode
                    full_ot_c = load_or_compute_ot(
                        full_corrupted_np,
                        source_supp_size=full_og_np.shape[1],
                        cache_path=(
                            _ot_cache_path(ds_dir, tag=f"{ctype}_k{k:.4f}_cseed{cseed}")
                            if use_ot_cache else None
                        ),
                        print_prefix=f"OT|{ctype}|k={k:.2g}|cseed={cseed}",
                        force=args.no_ot_cache,
                    )
                    train_pot_c = torch.from_numpy(
                        full_ot_c[train_idx.numpy()]).float()
                    val_pot_c   = torch.from_numpy(
                        full_ot_c[val_idx.numpy()]).float()

                    mode_data = {
                        "transport_maps": (train_pot_c, val_pot_c),
                        "linear":     (train_og_c,  val_og_c),
                    }

                    for sae_mode, (train_data, val_data) in mode_data.items():
                        valid_keys   = mode_valid_keys.get(sae_mode, [])
                        unique_ranks = unique_ranks_by_mode.get(sae_mode, [])
                        if not valid_keys:
                            continue

                        # NMF — one value per cseed (H fixed from clean)
                        if not args.skip_nmf:
                            nmf_res = eval_nmf_on_val(
                                dataset, ds_dir, sae_mode,
                                train_data, val_data,
                                unique_ranks, use_nmf_cache,
                                clean_H_dicts_by_mode[sae_mode],
                            )
                            for (ds, lbl), mse in nmf_res.items():
                                corrupt_nmf_values[lbl].append(mse)

                        # SAE — one value per (cseed, model_seed)
                        sae_res = eval_sae_on_val(
                            dataset, ds_dir, sae_mode, valid_keys,
                            args.seeds, val_data, device, args.batch_size,
                            subdir=subdir_map[sae_mode],
                        )
                        for (ds, lbl), mse_list in sae_res.items():
                            corrupt_sae_lists[lbl].extend(mse_list)

                # Aggregate over corruption seeds (+ model seeds for SAE)
                cond_key = (ctype, k)
                for lbl, mse_list in corrupt_sae_lists.items():
                    agg = aggregate_mse(mse_list)
                    corrupt_agg[cond_key][lbl] = agg
                    all_raw.append({"condition": _condition_key(ctype, k),
                                   "model": lbl, "dataset": dataset, **agg})

                for lbl, mse_list in corrupt_nmf_values.items():
                    agg = aggregate_mse(mse_list)
                    corrupt_agg[cond_key][lbl] = agg
                    all_raw.append({"condition": _condition_key(ctype, k),
                                   "model": lbl, "dataset": dataset, **agg})

                # Print summary for this (ctype, k)
                print(f"\n  Summary [{ctype} k={k:.2g}]:")
                all_model_lbls = (list(corrupt_sae_lists.keys()) +
                                  list(corrupt_nmf_values.keys()))
                for lbl in sorted(set(all_model_lbls)):
                    agg = corrupt_agg[cond_key].get(lbl, {})
                    print(f"    {lbl}: {agg.get('mse_mean', float('nan')):.6f} "
                          f"± {agg.get('mse_std', float('nan')):.6f} "
                          f"({agg.get('n_trials', '?')} trials)")

    # -----------------------------------------------------------------------
    # Write JSON
    # -----------------------------------------------------------------------
    output = {
        "config": {
            "root": str(root),
            "seeds": args.seeds,
            "potential_keys": args.potential_keys,
            "linear_keys": args.linear_keys,
            "val_frac": args.val_frac,
            "device": str(device),
            "corruption_types": ALL_CORRUPTION_TYPES,
            "corruption_strengths": args.corruption_strengths,
            "corruption_seeds": args.corruption_seeds,
        },
        "clean": {lbl: agg for lbl, agg in clean_agg.items()},
        "corrupted": {
            _condition_key(ctype, k): {
                lbl: agg for lbl, agg in model_dict.items()
            }
            for (ctype, k), model_dict in corrupt_agg.items()
        },
        "raw_results": all_raw,
    }

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON results written to {args.output_json}")

    # -----------------------------------------------------------------------
    # Render PNG
    # -----------------------------------------------------------------------
    # Determine stable model label ordering: NMF rows last, SAE sorted
    all_model_labels = sorted(
        set(clean_agg.keys()) |
        {lbl for md in corrupt_agg.values() for lbl in md}
    )
    # Put NMF rows at the bottom
    sae_labels = [l for l in all_model_labels if "NMF" not in l]
    nmf_labels = [l for l in all_model_labels if "NMF" in l]
    ordered_labels = sae_labels + nmf_labels

    render_mse_table_png(
        model_labels=ordered_labels,
        clean_agg=clean_agg,
        corrupt_agg=corrupt_agg,
        strengths=args.corruption_strengths,
        output_path=args.output_png,
    )


if __name__ == "__main__":
    main()
