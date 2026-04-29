"""
Multi-trial hyperspectral clustering evaluation.

Iterates over datasets, seeds, and SAE keys found in the directory structure:

    root/
      <dataset>/
        data/          -> contains a single .pt file
        ground_truth/  -> directory passed as labels_path
        transport_maps/    -> contains a single .pt file
        SAE_params/transport_maps/unreg/
          seed_0/
            <key_A>/
            <key_B>/
          seed_1/
            ...

NMF rank always matches the SAE hidden_dim (# atoms) for each key.
NMF is computed once per unique (dataset, seed, hidden_dim) combo.
For each (dataset, seed), NMF and SAE trials alternate per key.

Device handling:
  - By default uses cuda:0 if available, else cpu.
  - When launched via corruption_sweep_wrapper.py, CUDA_VISIBLE_DEVICES is set
    so that cuda:0 maps to the correct physical GPU.
  - Can be overridden with --device cpu to force CPU execution.

Speed optimizations:
  - Baseline (clean) OT maps are reused in memory within a run.
    Persistent <root>/<dataset>/.ot_cache/ files are only written with
    --use_ot_cache.
  - The --corruption_seeds flag accepts multiple seeds; data is loaded once,
    corruption + OT + NMF + SAE are run for each seed in a single process,
    and per-seed results are emitted in the output JSON as separate entries.

EXAMPLES
nohup python -u clustering_eval.py \
    --root datasets/hsi_data \
    --seeds 0 1 2 3 4  \
    --keys \
        JUMPRELUAE_15_1e-1_mon \
        JUMPRELUAE_15_5e-1_mon \
        JUMPRELUAE_10_5e-1_mon \
        JUMPRELUAE_10_1e-2_mon \
        JUMPRELUAE_17_1e-5_mon \
        JUMPRELUAE_17_1e-3_mon \
        JUMPRELUAE_7_1e-5_mon \
        JUMPRELUAE_7_1e-3_mon \
        JUMPRELUAE_7_1e-7_mon \
        \
    --sae_mode transport_maps\
    --output_json multitrial_results_drop_contiguous.json \
    --output_png multitrial_table_drop_drop_contiguous.png \
>& multitrial_contiguous.log &


nohup python -u clustering_eval.py \
    --root datasets/hsi_data \
    --seeds 0  \
    --keys \
        JUMPRELUAE_15_1e-1_nonneg \
        JUMPRELUAE_15_5e-1_nonneg \
        JUMPRELUAE_15_1e-3_nonneg \
        JUMPRELUAE_15_1e-5_nonneg \
        JUMPRELUAE_10_5e-1_nonneg \
        JUMPRELUAE_10_1e-2_nonneg \
        JUMPRELUAE_10_1e-3_nonneg \
        JUMPRELUAE_10_1e-4_nonneg \
        JUMPRELUAE_17_1e-3_nonneg \
        JUMPRELUAE_17_1e-5_nonneg\
        JUMPRELUAE_17_1e-2_nonneg\
        JUMPRELUAE_17_1e-1_nonneg\
        JUMPRELUAE_7_1e-3_nonneg\
        JUMPRELUAE_7_1e-5_nonneg\
        JUMPRELUAE_7_1e-2_nonneg\
        JUMPRELUAE_7_1e-1_nonneg\
        \
    --sae_mode linear\
    --output_json linear_multitrial_results.json \
    --output_png linear_multitrial_table.png \
>& linear_multitrial_clean.log &
"""

 #   --corruption drop_random 0 \

import argparse
import hashlib
import json
import os
import time as _time
from collections import defaultdict
from multiprocessing import Pool, cpu_count
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

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.mixture import GaussianMixture
from brenier_embedding_functions import wass_map_1D

from hsi_utils import (
    ARCH_REGISTRY,
    INPUT_DIM,
    SAE_PARAMETERS,
    load_labels,
    load_model,
    model_path_for,
    load_splits,
)
from SAE_analysis_functions import sae_encode_potential_batch

CLUSTERING_METHODS = ["kmeans", "gmm", "spectral"]


def hungarian_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    cost = np.zeros((len(true_labels), len(pred_labels)), dtype=np.int64)
    for i, t in enumerate(true_labels):
        for j, p in enumerate(pred_labels):
            cost[i, j] = np.sum((y_true == t) & (y_pred == p))
    row_ind, col_ind = linear_sum_assignment(cost.max() - cost)
    return float(cost[row_ind, col_ind].sum() / len(y_true))


def run_clustering(codes, labels, n_clusters, method_name, embedding):
    codes = np.asarray(codes, dtype=np.float64)
    labels = np.asarray(labels)

    if method_name == "kmeans":
        pred = KMeans(n_clusters=n_clusters, random_state=0, n_init=20).fit_predict(codes)
    elif method_name == "gmm":
        pred = GaussianMixture(n_components=n_clusters, random_state=0).fit_predict(codes)
    elif method_name == "spectral":
        pred = SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            assign_labels="kmeans",
            random_state=0,
        ).fit_predict(codes)
    else:
        raise ValueError(f"Unknown clustering method: {method_name}")

    return {
        "method": method_name,
        "embedding": embedding,
        "accuracy": hungarian_accuracy(labels, pred),
        "ari": float(adjusted_rand_score(labels, pred)),
        "nmi": float(normalized_mutual_info_score(labels, pred)),
    }


def nmf_fit_and_codes(data, rank, normalize=True):
    X = data.detach().cpu().numpy() if torch.is_tensor(data) else np.asarray(data)
    X = np.maximum(X.astype(np.float64), 0.0)
    if normalize:
        denom = X.sum(axis=1, keepdims=True)
        X = np.divide(X, denom, out=np.zeros_like(X), where=denom > 0)

    model = NMF(n_components=int(rank), init="nndsvda", random_state=0, max_iter=500)
    codes = model.fit_transform(X)
    dictionary = model.components_
    reconstruction = model.inverse_transform(codes)
    info = {"reconstruction_err": float(model.reconstruction_err_), "n_iter": int(model.n_iter_)}
    return codes, dictionary, reconstruction, info


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def resolve_device(device_str=None):
    """Return a torch.device.

    When CUDA_VISIBLE_DEVICES is set by the sweep wrapper, cuda:0 already
    refers to the correct physical GPU.
    """
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
    """Return the path to the single .pt file inside *directory*."""
    pts = list(Path(directory).glob("*.pt"))
    if len(pts) == 0:
        raise FileNotFoundError(f"No .pt file found in {directory}")
    if len(pts) > 1:
        raise RuntimeError(f"Multiple .pt files in {directory}: {pts}")
    return str(pts[0])


def discover_datasets(root):
    """Return sorted list of dataset directory names under *root*."""
    root = Path(root)
    datasets = []
    for d in sorted(root.iterdir(), key=lambda x: x.name):
        if d.is_dir() and (d / "data").is_dir():
            datasets.append(d.name)
    return datasets


def discover_keys(root, dataset, seed, sae_mode="transport_maps"):
    """Return sorted list of SAE key names for a given dataset and seed."""
    seed_dir = Path(root) / dataset / "SAE_params" / sae_mode / "unreg" / f"seed_{seed}"
    if not seed_dir.is_dir():
        return []
    return sorted(k.name for k in seed_dir.iterdir() if k.is_dir())


# ---------------------------------------------------------------------------
# OT cache helpers
# ---------------------------------------------------------------------------

def _ot_cache_path(root, dataset, tag="clean"):
    """Return the path for a cached OT .npy file.

    Cache lives at <root>/<dataset>/.ot_cache/<tag>.npy when --use_ot_cache is set.
    """
    cache_dir = Path(root) / dataset / ".ot_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{tag}.npy"


def load_or_compute_ot(data_np, source_supp_size, cache_path=None, print_prefix="OT",
                       n_workers=None, force_recompute=False):
    """Load cached OT maps if requested, otherwise compute without persisting.

    Parameters
    ----------
    data_np : ndarray (N, D)
    source_supp_size : int
    cache_path : Path or str or None — where to read/write the .npy cache.
        If None, no persistent cache is read or written.
    print_prefix : str
    n_workers : ignored (kept for API compat)
    force_recompute : bool — skip cache even if file exists

    Returns
    -------
    ot_maps : ndarray (N, source_supp_size)
    """
    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and not force_recompute and cache_path.exists():
        print(f"  {print_prefix}: Loading cached OT maps from {cache_path}", flush=True)
        ot_maps = np.load(cache_path)
        if ot_maps.shape == (data_np.shape[0], source_supp_size):
            print(f"  {print_prefix}: Loaded {ot_maps.shape} from cache", flush=True)
            return ot_maps
        else:
            print(f"  {print_prefix}: Cache shape mismatch "
                  f"({ot_maps.shape} vs expected ({data_np.shape[0]}, {source_supp_size})), "
                  f"recomputing...", flush=True)

    result = compute_ot_from_data(
        data_np, source_supp_size,
        print_prefix=print_prefix, n_workers=n_workers,
    )
    ot_maps = result["ot_maps"]

    if cache_path is not None:
        try:
            np.save(cache_path, ot_maps)
            print(f"  {print_prefix}: Cached OT maps to {cache_path}", flush=True)
        except Exception as e:
            print(f"  {print_prefix}: WARNING — could not write cache: {e}", flush=True)

    return ot_maps


# ---------------------------------------------------------------------------
# NMF trial (once per dataset+rank, always uses seed=0)
# ---------------------------------------------------------------------------
def run_nmf_trial(og_train, labels, nmf_rank, nonzero_mask, H_dict=None):
    """Compute NMF codes and cluster them. Returns (result_list, H_dict).

    If *H_dict* is provided, project og_train onto that fixed basis
    (no fitting). Otherwise fit NMF from scratch.

    *nonzero_mask* is applied to filter out background pixels.
    """
    r = int(nmf_rank)

    if H_dict is not None:
        from sklearn.decomposition import non_negative_factorization
        og_np = og_train.numpy() if torch.is_tensor(og_train) else np.asarray(og_train)
        og_np = np.maximum(og_np, 0)
        # Ensure H_dict matches X dtype (sklearn requires this)
        H_dict = H_dict.astype(og_np.dtype)
        W, H_out, _ = non_negative_factorization(
            og_np, W=None, H=H_dict, n_components=r,
            init='random', update_H=False, max_iter=300,
        )
        nmf_codes_all = W
    else:
        nmf_codes_all, H_dict, Xhat, info = nmf_fit_and_codes(
            og_train, rank=r, normalize=True,
        )

    # Apply nonzero mask
    nmf_codes = nmf_codes_all[nonzero_mask]
    labels_filtered = labels[nonzero_mask]
    n_clusters = len(np.unique(labels_filtered))

    print(f"    NMF codes shape: {nmf_codes.shape} (after removing background)")

    all_results = []
    for method_name in CLUSTERING_METHODS:
        result = run_clustering(
            nmf_codes, labels_filtered, n_clusters, method_name, "NMF",
        )
        all_results.append(result)

    return all_results, H_dict


# ---------------------------------------------------------------------------
# SAE trial (once per dataset+key+seed)
# ---------------------------------------------------------------------------
def run_sae_trial(
    key,
    potential_pt,
    og_pt,
    sae_root,
    labels_path,
    seed,
    input_dim,
    batch_size=1024,
    precomputed_potentials=None,
    device=None,
):
    """Run SAE encoding + clustering for one (dataset, key, seed) combo.

    NO SPLITS: uses the full dataset each time.

    *input_dim* is the number of spectral bands for this dataset.
    *device* is the torch.device to use for SAE encoding.

    Returns a list of per-method result dicts (SAE only, no NMF).
    """
    if device is None:
        device = resolve_device()

    cfg = SAE_PARAMETERS[key]
    sae_hidden_dim = cfg["hidden_dim"]

    print(f"    SAE hidden_dim (# atoms) = {sae_hidden_dim}")
    print(f"    input_dim (spectral bands) = {input_dim}")
    print(f"    device = {device}")

    # 1. Load data (NO split)
    if precomputed_potentials is not None:
        ot_train = precomputed_potentials
    else:
        ot_train = torch.load(potential_pt, map_location="cpu")
        ot_train=ot_train['cube']
        a, b, c = ot_train.shape
        ot_train = ot_train.reshape(a * b, c)
    # 2. Labels (NO split)
    labels_flat = load_labels(labels_path)
    labels_all = np.asarray(labels_flat, dtype=np.int64).reshape(-1)
    # Sanity: OT and labels must align in length
    N = ot_train.shape[0]
    if labels_all.shape[0] != N:
        raise ValueError(
            f"Label/OT mismatch: labels has {labels_all.shape[0]} entries, OT has {N} rows. "
            f"(Check label flattening / dataset ordering.)"
        )

    # Always remove background (label 0)
    nonzero_mask = (labels_all != 0)
    labels = labels_all[nonzero_mask]
    n_clusters = len(np.unique(labels))

    # 3. SAE encodings
    ot_codes_all = sae_encode_potential_batch(
        ot_train,
        model_path_for(key, sae_root),
        cfg["architecture"],
        input_dim,
        cfg["hidden_dim"],
        cfg["top_K"],
        device=device,
        batch_size=batch_size,
    )
    if torch.is_tensor(ot_codes_all):
        ot_codes_all = ot_codes_all.detach().cpu().numpy().astype(np.float64)
    else:
        ot_codes_all = np.asarray(ot_codes_all, dtype=np.float64)

    # Apply nonzero mask
    ot_codes = ot_codes_all[nonzero_mask]
    print(f"    SAE codes shape: {ot_codes.shape} (after removing background)")

    # 4. Clustering (SAE only)
    all_results = []
    for method_name in CLUSTERING_METHODS:
        result = run_clustering(
            ot_codes, labels, n_clusters, method_name, "SAE",
        )
        all_results.append(result)

    return all_results



# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

METRICS = ["accuracy"]


def aggregate_over_seeds(per_seed_results):
    """Given {seed: [result_dicts]}, compute mean/var per (method, embedding, metric).

    Returns a list of aggregated dicts.
    """
    # Group by (method, embedding)
    groups = defaultdict(lambda: defaultdict(list))
    for seed, results in per_seed_results.items():
        for r in results:
            gkey = (r["method"], r["embedding"])
            for m in METRICS:
                groups[gkey][m].append(r[m])

    aggregated = []
    for (method, embedding), metric_vals in groups.items():
        entry = {
            "method": method,
            "embedding": embedding,
            "n_seeds": len(next(iter(metric_vals.values()))),
        }
        for m in METRICS:
            vals = np.array(metric_vals[m])
            entry[f"{m}_mean"] = float(np.mean(vals))
            entry[f"{m}_var"] = float(np.var(vals))
            entry[f"{m}_std"] = float(np.std(vals))
        aggregated.append(entry)

    return aggregated


# ---------------------------------------------------------------------------
# PNG table generation — accuracy only,
# NMF as its own row per dataset, best accuracy bolded per dataset group
# ---------------------------------------------------------------------------

def _build_table(ax, title, all_sae_agg, all_nmf_agg, datasets):
    """Build a single table on *ax*.

    Rows: for each dataset, one row per SAE key + one NMF row.
    Columns: one per clustering method (accuracy only).
    Best accuracy per dataset group is bolded.
    """
    # Collect clustering method names across all aggregated entries
    method_set = set()
    for agg_list in list(all_sae_agg.values()) + list(all_nmf_agg.values()):
        for a in agg_list:
            method_set.add(a["method"])
    methods = sorted(method_set)

    if not methods:
        ax.axis("off")
        ax.set_title(f"{title}\n(no results)", fontsize=11)
        return

    col_headers = [m for m in methods]
    n_cols = len(col_headers)

    # Build rows grouped by dataset
    row_labels = []
    row_data = []       # list of {method: {accuracy_mean, accuracy_std}}
    row_dataset = []    # dataset name for each row (for grouping)

    for dataset in datasets:
        # SAE key rows
        sae_keys_for_ds = sorted(
            k for (ds, k) in all_sae_agg if ds == dataset
        )
        for key in sae_keys_for_ds:
            row_labels.append(f"{dataset} / {key}")
            row_dataset.append(dataset)
            rd = {}
            for a in all_sae_agg[(dataset, key)]:
                rd[a["method"]] = a
            row_data.append(rd)

        # NMF rows for this dataset (one per unique hidden_dim/rank)
        nmf_hdims_for_ds = sorted(
            hdim for (ds, hdim) in all_nmf_agg if ds == dataset
        )
        for hdim in nmf_hdims_for_ds:
            row_labels.append(f"{dataset} / NMF (rank={hdim})")
            row_dataset.append(dataset)
            rd = {}
            for a in all_nmf_agg[(dataset, hdim)]:
                rd[a["method"]] = a
            row_data.append(rd)

    n_rows = len(row_labels)
    if n_rows == 0:
        ax.axis("off")
        ax.set_title(f"{title}\n(no results)", fontsize=11)
        return

    # Find best accuracy per (dataset, method) across all rows in that group
    best_per_group = defaultdict(lambda: -1.0)  # (dataset, method) -> best mean
    for ri in range(n_rows):
        ds = row_dataset[ri]
        for m in methods:
            if m in row_data[ri]:
                val = row_data[ri][m].get("accuracy_mean", -1)
                if val > best_per_group[(ds, m)]:
                    best_per_group[(ds, m)] = val

    # Build cell text and bold mask
    cell_text = []
    bold_mask = []
    for ri in range(n_rows):
        ds = row_dataset[ri]
        row_texts = []
        row_bolds = []
        for m in methods:
            if m in row_data[ri]:
                mean_val = row_data[ri][m].get("accuracy_mean", float("nan"))
                std_val = row_data[ri][m].get("accuracy_std", float("nan"))
                # For NMF (single seed), std will be 0 — show just mean
                if std_val == 0.0:
                    txt = f"{mean_val:.4f}"
                else:
                    txt = f"{mean_val:.4f} ± {std_val:.4f}"
                is_best = (
                    abs(mean_val - best_per_group[(ds, m)]) < 1e-9
                    and mean_val > 0
                )
            else:
                txt = "—"
                is_best = False
            row_texts.append(txt)
            row_bolds.append(is_best)
        cell_text.append(row_texts)
        bold_mask.append(row_bolds)

    # Draw table
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=10)

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Style header row
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_text_props(fontweight="bold", fontsize=9)
        cell.set_facecolor("#d9e2f3")

    # Style row labels + shade NMF rows
    for i in range(n_rows):
        label_cell = table[i + 1, -1]
        label_cell.set_text_props(fontweight="bold", fontsize=8)
        if "/ NMF" in row_labels[i]:
            label_cell.set_facecolor("#f2dcdb")
            for j in range(n_cols):
                table[i + 1, j].set_facecolor("#fdf2f2")
        else:
            label_cell.set_facecolor("#e2efda")

    # Bold the best cells
    for i in range(n_rows):
        for j in range(n_cols):
            if bold_mask[i][j]:
                cell = table[i + 1, j]
                cell.set_text_props(fontweight="bold", color="darkgreen")
                cell.set_facecolor("#ffffcc")

    return table

def render_table_png(all_sae_agg, all_nmf_agg, datasets, output_path,
                     corruption_type=None, corruption_k=None):
    """Render table in one PNG, accuracy only."""
    if corruption_type is not None:
        title = (f"Clustering Accuracy — Nonzero Labels (mean ± std over seeds)\n"
                 f"Corruption: {corruption_type}, k={corruption_k}")
    else:
        title = "Clustering Accuracy — Nonzero Labels (mean ± std over seeds)"

    fig, ax = plt.subplots(1, 1, figsize=(14, max(4, 1.2 * 10)))

    _build_table(
        ax,
        title,
        all_sae_agg,
        all_nmf_agg,
        datasets,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nTable saved to {output_path}")

from hsi_utils import (
    drop_random_bands,
    drop_contiguous_bands,
    shift_mass_global,
    shift_mass_random_split,
    pushforward_log_warp
)


CORRUPTION_TYPES = {
    "drop_random",
    "drop_contiguous",
    "shift_global",
    "shift_split",
    "log_warp",
}


def apply_corruption(data, corruption_type, k, rng=None):
    """Apply a corruption to an (N, M) array.

    Parameters
    ----------
    data : np.ndarray, shape (N, M)
        Input spectra / codes.
    corruption_type : str
        One of: "drop_random", "drop_contiguous", "shift_global", "shift_split".
    k : float
        Fraction in (0, 1].
        - For drop corruptions: fraction of mass to remove per spectrum.
        - For shift corruptions: used directly as ``frac``.
    rng : np.random.Generator, optional

    Returns
    -------
    corrupted : np.ndarray, same shape as *data*
    info : dict
        Metadata about the corruption applied.
    """
    if corruption_type not in CORRUPTION_TYPES:
        raise ValueError(
            f"Unknown corruption type '{corruption_type}'. "
            f"Must be one of {sorted(CORRUPTION_TYPES)}"
        )
    if corruption_type == "log_warp":
        if k <= 0:
            raise ValueError(f"log_warp requires a > 0, got {k}")
    elif not (0.0 < k <= 1.0):
        raise ValueError(f"k must be in (0, 1], got {k}")

    if rng is None:
        rng = np.random.default_rng()

    data = np.asarray(data, dtype=np.float64)
    N, M = data.shape

    if corruption_type == "drop_random":
        corrupted, mask = drop_random_bands(data, k, rng=rng)
        info = {"type": corruption_type, "k_frac": k}

    elif corruption_type == "drop_contiguous":
        corrupted, mask = drop_contiguous_bands(data, k, rng=rng)
        info = {"type": corruption_type, "k_frac": k}

    elif corruption_type == "shift_global":
        corrupted, info = shift_mass_global(data, frac=k, rng=rng)
        info["type"] = corruption_type
        info["k_frac"] = k

    elif corruption_type == "shift_split":
        corrupted, info = shift_mass_random_split(data, frac=k, rng=rng)
        info["type"] = corruption_type
        info["k_frac"] = k

    elif corruption_type == "log_warp":
        # k is reinterpreted as the warp strength parameter 'a'
        corrupted = np.empty_like(data)
        for i in range(N):
            corrupted[i] = pushforward_log_warp(data[i], a=k)
        info = {"type": corruption_type, "a": k}


    return corrupted, info


# ---------------------------------------------------------------------------
# Parallel OT computation
# ---------------------------------------------------------------------------

def compute_ot_from_data(
    data,
    source_supp_size,
    *,
    print_prefix="OT",
    n_workers=None,  # kept for signature compatibility, ignored
):
    data_np = data.detach().cpu().numpy() if torch.is_tensor(data) else np.asarray(data)
    N, D = data_np.shape

    source_grid = np.linspace(0.0, 1.0, source_supp_size)
    target_grid = np.linspace(0.0, 1.0, D)

    ot_maps = np.empty((N, source_supp_size), dtype=np.float64)
    print_every = max(1, N // 20)
    t0 = _time.time()

    for i in range(N):
        if i % print_every == 0 or i == N - 1:
            elapsed = _time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-9)
            eta = (N - i - 1) / max(rate, 1e-9)
            print(f"  {print_prefix} progress: {i+1}/{N} "
                  f"({100*(i+1)/N:.1f}%) "
                  f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                  flush=True)

        target_masses = data_np[i].reshape(-1)
        result = wass_map_1D(source_grid, None, target_grid, target_masses)
        if torch.is_tensor(result):
            result = result.detach().cpu().numpy()
        else:
            result = np.asarray(result)
        ot_maps[i] = result.reshape(-1)

    elapsed = _time.time() - t0
    print(f"  {print_prefix}: Done ({N} OT maps in {elapsed:.2f}s)", flush=True)

    return {"ot_maps": ot_maps}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-trial clustering comparison across datasets, seeds, and SAE keys"
    )
    parser.add_argument("--root", required=True,
                        help="Root directory containing dataset subdirectories")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Seed indices to iterate over (default: 0 1 2 3 4)")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--output_json", default="multitrial_results.json",
                        help="Path to write aggregated results JSON")
    parser.add_argument("--output_png", default="multitrial_table.png",
                        help="Path to write summary table PNG")
    parser.add_argument("--sae_mode", type=str, choices=["linear", "transport_maps", "potentials"], default="transport_maps",
                    help="Subdirectory under SAE_params (default: transport_maps; potentials is a deprecated alias)")

    parser.add_argument("--keys", type=str, nargs="+", default=None,
                    help="Specific SAE keys to use (default: all discovered keys)")
    parser.add_argument(
        "--corruption", type=str, nargs=2, default=None, metavar=("TYPE", "K"),
        help="Corruption type and fraction k. Types: drop_random, drop_contiguous, shift_global, shift_split"
    )
    parser.add_argument("--corruption_seeds", type=int, nargs="+", default=[0],
                    help="RNG seed(s) for corruption (default: [0]). "
                         "Multiple seeds are processed in a single invocation, "
                         "sharing data loading and clean OT computation.")
    # Keep old name as alias for backward compat
    parser.add_argument("--corruption_seed", type=int, default=None,
                    help="(Deprecated) Single corruption seed. Use --corruption_seeds instead.")
    parser.add_argument("--device", type=str, default=None,
                    help="Torch device (default: cuda:0 if available, else cpu). "
                         "When launched by the sweep wrapper, CUDA_VISIBLE_DEVICES "
                         "is set so cuda:0 maps to the assigned physical GPU.")
    parser.add_argument("--ot_workers", type=int, default=None,
                    help="Number of parallel workers for OT computation "
                         "(default: min(cpu_count, 16))")
    parser.add_argument("--use_ot_cache", action="store_true",
                    help="Persist OT/NMF caches under each dataset directory. "
                         "Default is off to avoid large .ot_cache directories.")
    parser.add_argument("--no_ot_cache", action="store_true",
                    help="Deprecated alias for the default no-cache behavior.")

    args = parser.parse_args()
    if args.sae_mode == "potentials":
        print("Note: --sae_mode potentials is deprecated for HSI; using transport_maps.")
        args.sae_mode = "transport_maps"
    use_persistent_cache = args.use_ot_cache and not args.no_ot_cache

    # Handle backward-compat: --corruption_seed N  ->  --corruption_seeds [N]
    if args.corruption_seed is not None:
        corruption_seeds = [args.corruption_seed]
    else:
        corruption_seeds = args.corruption_seeds

    # Resolve device once, use everywhere
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    corruption_type = None
    corruption_k = None
    if args.corruption is not None:
        corruption_type = args.corruption[0]
        corruption_k = float(args.corruption[1])
        if corruption_type not in CORRUPTION_TYPES:
            raise ValueError(f"Unknown corruption type '{corruption_type}'.")
        print(f"Corruption: {corruption_type}, k={corruption_k}, "
              f"corruption_seeds={corruption_seeds}")

    root = Path(args.root)
    datasets = discover_datasets(root)
    print(f"Discovered datasets: {datasets}")

    # ---- Per-corruption-seed collectors ----
    # We collect results across ALL corruption seeds in this single process.
    # The output JSON will contain one entry per corruption seed in
    # "per_corruption_seed_outputs", each structured like the old single-seed
    # output (with aggregated_sae, aggregated_nmf, etc.).
    per_cseed_outputs = {}  # cseed -> {aggregated_sae, aggregated_nmf, per_seed_results, ...}

    for dataset in datasets:
        ds_dir = root / dataset
        potential_pt = find_single_pt(ds_dir / "transport_maps")
        og_pt = find_single_pt(ds_dir / "data")
        labels_path = str(ds_dir / "ground_truth")

        # Compute n_classes from ground-truth labels for this dataset (excluding 0)
        labels_flat_ds = load_labels(labels_path)
        nonzero_labels = labels_flat_ds[labels_flat_ds != 0]
        n_classes = len(np.unique(nonzero_labels))
        print(f"\nDataset '{dataset}': {n_classes} classes (excluding background)")

        # Use supplied keys as allowlist, intersected with what exists on disk
        if args.keys is not None:
            allowed = set(args.keys)
            discovered = set()
            for seed in args.seeds:
                discovered.update(discover_keys(root, dataset, seed, args.sae_mode))
            all_keys = sorted(allowed & discovered)
            skipped = sorted(allowed - discovered)
            if skipped:
                print(f"  Keys not found in any seed dir for '{dataset}': {skipped}")
        else:
            all_keys = set()
            for seed in args.seeds:
                all_keys.update(discover_keys(root, dataset, seed, args.sae_mode))
            all_keys = sorted(all_keys)

        print(f"\n{'#' * 70}")
        print(f"Dataset: {dataset}  |  Keys: {all_keys}  |  n_classes: {n_classes}")
        print(f"{'#' * 70}")

        # ------------------------------------------------------------------
        # Infer input_dim (spectral bands) from the data
        # ------------------------------------------------------------------
        _tmp = torch.load(og_pt, map_location="cpu")
        if torch.is_tensor(_tmp):
            _cube = _tmp
        elif isinstance(_tmp, dict) and "cube" in _tmp:
            _cube = _tmp["cube"]
        else:
            raise ValueError(f"Unexpected data format in {og_pt}")
        input_dim = _cube.shape[2]  # (rows, cols, bands)
        del _tmp, _cube
        print(f"  input_dim (spectral bands) = {input_dim}")

        # Filter to valid keys
        valid_keys = [k for k in all_keys if k in SAE_PARAMETERS]
        for k in all_keys:
            if k not in SAE_PARAMETERS:
                print(f"  WARNING: key '{k}' not in SAE_PARAMETERS, skipping.")

        unique_hidden_dims = sorted(set(
            SAE_PARAMETERS[k]["hidden_dim"] for k in valid_keys
        ))

        # ------------------------------------------------------------------
        # Load raw data ONCE for this dataset
        # ------------------------------------------------------------------
        og_train_seed0 = torch.load(og_pt, map_location="cpu")  # (a,b,c)
        a, b, c = og_train_seed0.shape
        og_train_seed0 = og_train_seed0.reshape(a*b, c)
        og_train_np = og_train_seed0.numpy()        # (a*b,c)

        labels_all_seed0 = np.asarray(load_labels(labels_path)).reshape(-1)
        nonzero_mask_seed0 = labels_all_seed0 != 0
# ------------------------------------------------------------------
        # Cache / load clean OT maps (shared across all corruption seeds)
        # ------------------------------------------------------------------
        clean_potentials = None
        if args.sae_mode == "transport_maps":
            cache_path = _ot_cache_path(root, dataset, tag="clean") if use_persistent_cache else None
            clean_potentials = load_or_compute_ot(
                og_train_np,
                source_supp_size=og_train_np.shape[1],
                cache_path=cache_path,
                print_prefix=f"OT|{dataset}|clean",
                n_workers=args.ot_workers,
                force_recompute=args.no_ot_cache,
            )
        # ------------------------------------------------------------------
        # Fit clean NMF dictionaries ONCE per dataset (before corruption loop)
        # Cache H_dict to disk so it persists across subprocess invocations.
        # ------------------------------------------------------------------
        nmf_cache_dir = ds_dir / ".nmf_cache" if use_persistent_cache else None
        if nmf_cache_dir is not None:
            nmf_cache_dir.mkdir(parents=True, exist_ok=True)

        clean_nmf_dicts = {}    # hidden_dim -> H_dict
        clean_nmf_results = {}  # hidden_dim -> results list
        for hidden_dim in unique_hidden_dims:
            cache_file = nmf_cache_dir / f"H_dict_rank{hidden_dim}.npy" if nmf_cache_dir is not None else None
            print(f"\n  --- NMF (clean fit) | {dataset} | rank={hidden_dim} ---")

            if use_persistent_cache and cache_file.exists():
                print(f"    Loading cached H_dict from {cache_file}", flush=True)
                H_dict = np.load(cache_file)
                # Project clean data onto cached dictionary to get clean results
                results, _ = run_nmf_trial(
                    og_train_seed0, labels_all_seed0, hidden_dim, nonzero_mask_seed0,
                    H_dict=H_dict,
                )
            else:
                results, H_dict = run_nmf_trial(
                    og_train_seed0, labels_all_seed0, hidden_dim, nonzero_mask_seed0,
                    H_dict=None,
                )
                if use_persistent_cache:
                    try:
                        np.save(cache_file, H_dict)
                        print(f"    Cached H_dict to {cache_file}", flush=True)
                    except Exception as e:
                        print(f"    WARNING — could not write NMF cache: {e}", flush=True)

            clean_nmf_dicts[hidden_dim] = H_dict
            clean_nmf_results[hidden_dim] = results


        # ------------------------------------------------------------------
        # Loop over corruption seeds
        # ------------------------------------------------------------------
        for cseed in corruption_seeds:
            cseed_key = cseed if corruption_type is not None else None

            # Initialize per-cseed storage if needed
            if cseed_key not in per_cseed_outputs:
                per_cseed_outputs[cseed_key] = {
                    "sae_per_seed": defaultdict(dict),
                    "nmf_per_seed": defaultdict(dict),
                    "full_results": [],
                    "all_sae_agg": {},
                    "all_nmf_agg": {},
                }
            out = per_cseed_outputs[cseed_key]

            print(f"\n  --- Corruption seed: {cseed} ---" if corruption_type else "")

            # ----------------------------------------------------------
            # Corrupt (or not)
            # ----------------------------------------------------------
            corrupted_potentials = None
            og_train_nmf = og_train_seed0  # torch, no corruption
            og_train_corrupt_np = None

            if corruption_type is not None:
                og_train_corrupt_np, corr_info = apply_corruption(
                    og_train_np,
                    corruption_type,
                    corruption_k,
                    rng=np.random.default_rng(cseed),
                )
                og_train_nmf = torch.from_numpy(og_train_corrupt_np).float()

                print(f"  Applied {corruption_type} (k={corruption_k}, cseed={cseed}) "
                      f"to TRAIN input (shared for NMF + OT)")

                if args.sae_mode == "transport_maps":
                    print(f"\n  --- Computing OT maps from corrupted TRAIN | {dataset} | cseed={cseed} ---")
                    corrupted_potentials = load_or_compute_ot(
                        og_train_corrupt_np,
                        source_supp_size=og_train_np.shape[1],
                        cache_path=(
                            _ot_cache_path(
                                root, dataset,
                                tag=f"{corruption_type}_k{corruption_k}_cseed{cseed}",
                            )
                            if use_persistent_cache else None
                        ),
                        print_prefix=f"OT|{dataset}|cseed{cseed}",
                        n_workers=args.ot_workers,
                        force_recompute=args.no_ot_cache,
                    )
            else:
                # No corruption — use clean transport maps
                corrupted_potentials = clean_potentials
# ----------------------------------------------------------
            # NMF for each hidden_dim — project corrupted data onto
            # the clean-fitted dictionary (H fixed)
            # ----------------------------------------------------------
            for hidden_dim in unique_hidden_dims:
                print(f"\n  --- NMF | {dataset} | cseed={cseed_key} | rank={hidden_dim} ---")
                try:
                    if corruption_type is None:
                        # No corruption — reuse the clean fit results directly
                        nmf_results = clean_nmf_results[hidden_dim]
                    else:
                        # Project corrupted data onto clean-fitted dictionary
                        nmf_results, _ = run_nmf_trial(
                            og_train_nmf, labels_all_seed0, hidden_dim, nonzero_mask_seed0,
                            H_dict=clean_nmf_dicts[hidden_dim],
                        )
                    out["nmf_per_seed"][(dataset, hidden_dim)][0] = nmf_results
                    out["full_results"].append({
                        "dataset": dataset,
                        "key": f"NMF_rank{hidden_dim}",
                        "seed": 0,
                        "corruption_seed": cseed_key,
                        "results": nmf_results,
                    })
                except Exception as e:
                    print(f"  ERROR on NMF {dataset}/cseed_{cseed}/rank_{hidden_dim}: {e}")
                    import traceback
                    traceback.print_exc()
            # ----------------------------------------------------------
            # SAE across training seeds
            # ----------------------------------------------------------
            for seed in args.seeds:
                for key in valid_keys:
                    cfg = SAE_PARAMETERS[key]
                    hidden_dim = cfg["hidden_dim"]

                    sae_root = str(
                        ds_dir / "SAE_params" / args.sae_mode / "unreg" / f"seed_{seed}"
                    )
                    key_dir = Path(sae_root) / key
                    if not key_dir.is_dir():
                        print(f"  Seed {seed}: {key_dir} not found, skipping.")
                        continue

                    # Choose the right input for the SAE
                    if args.sae_mode == "linear":
                        if og_train_corrupt_np is not None:
                            precomputed = og_train_corrupt_np
                        else:
                            precomputed = og_train_np
                    else:
                        precomputed = corrupted_potentials

                    print(f"  === SAE | {dataset} | {key} | seed={seed} | cseed={cseed_key} ===")
                    try:
                        results = run_sae_trial(
                            key=key,
                            potential_pt=potential_pt,
                            og_pt=og_pt,
                            sae_root=sae_root,
                            labels_path=labels_path,
                            seed=seed,
                            input_dim=input_dim,
                            batch_size=args.batch_size,
                            precomputed_potentials=precomputed,
                            device=device,
                        )

                        out["sae_per_seed"][(dataset, key)][seed] = results
                        out["full_results"].append({
                            "dataset": dataset,
                            "key": key,
                            "seed": seed,
                            "corruption_seed": cseed_key,
                            "results": results,
                        })
                    except Exception as e:
                        print(f"  ERROR on {dataset}/{key}/seed_{seed}/cseed_{cseed}: {e}")
                        import traceback
                        traceback.print_exc()

            # Aggregate after all seeds for this dataset + cseed
            for (ds, key), seed_dict in out["sae_per_seed"].items():
                if ds == dataset and seed_dict:
                    out["all_sae_agg"][(ds, key)] = aggregate_over_seeds(seed_dict)
            for (ds, hdim), seed_dict in out["nmf_per_seed"].items():
                if ds == dataset and seed_dict:
                    out["all_nmf_agg"][(ds, hdim)] = aggregate_over_seeds(seed_dict)

    # ------------------------------------------------------------------
    # Build output JSON
    # ------------------------------------------------------------------
    # If only one corruption seed (the common case when called by the old
    # wrapper), emit the same top-level structure for backward compat.
    # If multiple seeds, emit per_corruption_seed_outputs.
    if len(per_cseed_outputs) == 1:
        cseed_key = next(iter(per_cseed_outputs))
        out = per_cseed_outputs[cseed_key]
        output = {
            "config": {
                "root": str(root),
                "seeds": args.seeds,
                "nmf_seed": 0,
                "nmf_rank_policy": "matched_to_sae_hidden_dim",
                "n_clusters_policy": "from_ground_truth_labels_excluding_background",
                "corruption_seed": cseed_key if cseed_key is not None else 0,
                "device": str(device),
            },
            "per_seed_results": out["full_results"],
            "aggregated_sae": {
                f"{ds}|{k}": agg for (ds, k), agg in out["all_sae_agg"].items()
            },
            "aggregated_nmf": {
                f"{ds}|rank{hdim}": agg for (ds, hdim), agg in out["all_nmf_agg"].items()
            },
        }
    else:
        # Multi-cseed mode: emit one block per corruption seed
        all_cseed_blocks = {}
        for cseed_key, out in per_cseed_outputs.items():
            all_cseed_blocks[str(cseed_key)] = {
                "per_seed_results": out["full_results"],
                "aggregated_sae": {
                    f"{ds}|{k}": agg for (ds, k), agg in out["all_sae_agg"].items()
                },
                "aggregated_nmf": {
                    f"{ds}|rank{hdim}": agg for (ds, hdim), agg in out["all_nmf_agg"].items()
                },
            }
        output = {
            "config": {
                "root": str(root),
                "seeds": args.seeds,
                "nmf_seed": 0,
                "nmf_rank_policy": "matched_to_sae_hidden_dim",
                "n_clusters_policy": "from_ground_truth_labels_excluding_background",
                "corruption_seeds": corruption_seeds,
                "device": str(device),
            },
            "per_corruption_seed_outputs": all_cseed_blocks,
        }

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON results written to {args.output_json}")

    # ------------------------------------------------------------------
    # Render PNG table (use first cseed for the table)
    # ------------------------------------------------------------------
    first_out = next(iter(per_cseed_outputs.values()))
    render_table_png(
        first_out["all_sae_agg"], first_out["all_nmf_agg"],
        datasets, args.output_png,
        corruption_type=corruption_type, corruption_k=corruption_k,
    )


if __name__ == "__main__":
    main()
