"""
Utility functions for hyperspectral SAE dictionary learning experiments.

Requires: SAE, SAE_analysis_functions, brenier_embedding_functions
          (must be on PYTHONPATH or in working directory)
"""

import numpy as np
import torch
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wasserstein_distance
from sklearn.decomposition import NMF
from pathlib import Path
import time
import json
import sys
import ot

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import SAE
from SAE_analysis_functions import *
from brenier_embedding_functions import *

# ── Architecture registry ──
ARCH_REGISTRY = {
    "TopKAE":              TopKAE,
    "topk":                TopKAE,
    "JumpReLU":            JumpReLUAE,
    "GatedSAE":            GatedSAE,
    "ReLUAE":              ReLUAE,
    "TopKAE_monotone":     TopKAE_monotone,
    "JumpReLUAE_monotone": JumpReLUAE_monotone,
    "JumpReLU_monotone": JumpReLUAE_monotone,
    "ReLUAE_monotone":     ReLUAE_monotone,
    "JumpReLUAE_nonneg": JumpReLUAE_nonneg,
    "JumpReLU_nonneg":JumpReLUAE_nonneg
}

INPUT_DIM = 102
DEVICE = "cpu"


def _sae_param(architecture, hidden_dim, l1, lr=1e-5, top_k=0):
    return {
        "architecture": architecture,
        "l1": l1,
        "lr": lr,
        "hidden_dim": hidden_dim,
        "top_K": top_k,
    }


SAE_PARAMETERS = {}
for _hidden_dim in (7, 10, 15, 17, 20, 40, 128, 512, 1024, 2048):
    for _l1 in ("1e-8", "1e-7", "1e-6", "1e-5", "1e-4", "1e-3", "1e-2", "1e-1", "5e-1", "1"):
        _compact_l1 = _l1.replace("-", "")
        SAE_PARAMETERS[f"JUMPRELUAE_{_hidden_dim}_{_compact_l1}_mon"] = _sae_param(
            "JumpReLU_monotone", _hidden_dim, _l1,
        )
        SAE_PARAMETERS[f"JUMPRELUAE_{_hidden_dim}_{_l1}_mon"] = _sae_param(
            "JumpReLU_monotone", _hidden_dim, _l1,
        )
        SAE_PARAMETERS[f"JUMPRELUAE_{_hidden_dim}_{_compact_l1}_nonneg"] = _sae_param(
            "JumpReLU_nonneg", _hidden_dim, _l1, lr=1e-4,
        )
        SAE_PARAMETERS[f"JUMPRELUAE_{_hidden_dim}_{_l1}_nonneg"] = _sae_param(
            "JumpReLU_nonneg", _hidden_dim, _l1, lr=1e-4,
        )
SAE_PARAMETERS.update({
    "TOPKAE_128_mon": _sae_param("TopKAE_monotone", 128, "0", top_k=8),
    "TOPKAE_512_mon": _sae_param("TopKAE_monotone", 512, "0", top_k=12),
    "TOPKAE_1024_mon": _sae_param("TopKAE_monotone", 1024, "0", top_k=32),
    "TOPKAE_2048_mon": _sae_param("TopKAE_monotone", 2048, "0", top_k=32),
})


def load_labels(labels_path):
    labels_path = Path(labels_path)
    if labels_path.is_dir():
        pts = sorted(labels_path.glob("*.pt"))
        if len(pts) != 1:
            raise RuntimeError(f"Expected one .pt label file in {labels_path}, found {pts}")
        labels_path = pts[0]

    labels = torch.load(labels_path, map_location="cpu")
    if isinstance(labels, dict):
        labels = labels.get("gt", labels.get("labels", labels.get("cube")))
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    return np.asarray(labels).reshape(-1)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(architecture, hidden_dim, model_path, top_k=None, device=DEVICE,
               input_dim=None):
    """Instantiate an SAE, load weights, return in eval mode.

    If *input_dim* is None, falls back to the module-level INPUT_DIM (102).
    Pass the actual spectral band count when working with non-Pavia datasets.
    """
    dim = input_dim if input_dim is not None else INPUT_DIM
    cls = ARCH_REGISTRY[architecture]
    model = (
        cls(dim, hidden_dim, top_k=top_k)
        if cls in (TopKAE, TopKAE_monotone)
        else cls(dim, hidden_dim)
    )
    model = model.to(device)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def model_path_for(key, sae_root):
    """Return the checkpoint path for a given SAE_PARAMETERS key."""
    return Path(sae_root) / key / "sparse_ae.pt"


def get_atoms(model):
    """Extract atom matrix (K, m) as numpy float64."""
    with torch.no_grad():
        return model.atoms().detach().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_splits(data_dir, seed=0):
    """
    Load hyperspectral data and split into train/val sets.
    Uses the same seed as the original training script for identical split.

    Args:
        data_dir: path to .pt file containing data
        seed: Random seed (default: 0, same as training script)

    Returns:
        train_data: Training tensor  (N_train, D)
        val_data:   Validation tensor (N_val, D)
    """
    torch.manual_seed(seed)

    data = torch.load(data_dir)
    if torch.is_tensor(data):
        cube = data
    elif isinstance(data, dict) and "cube" in data:
        cube = data["cube"]

    a, b, c = cube.shape
    flat_cube = cube.reshape(a * b, c)

    n_samples = flat_cube.shape[0]
    #HACKY FIX
    n_val = 1
    
    #int(0.0 * n_samples)

    perm = torch.randperm(n_samples)
    val_indices = perm[:n_val]
    train_indices = perm[n_val:]

    train_data = flat_cube[train_indices]
    val_data = flat_cube[val_indices]

    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    return train_data, val_data


# ---------------------------------------------------------------------------
# Inverse-CDF helpers
# ---------------------------------------------------------------------------

def normalize_inv_cdf(T, eps=1e-12):
    """Shift to start at 0, scale to end at 1, enforce monotonicity."""
    T = T - T[0]
    T = T / (T[-1] if abs(T[-1]) > eps else eps)
    return np.maximum.accumulate(T)


def fd_density_from_inv_cdf(inv_cdf, eps=1e-12):
    """Finite-difference density from a monotone inverse-CDF."""
    T = np.maximum.accumulate(np.asarray(inv_cdf, dtype=np.float64))
    m = len(T)
    dp = 1.0 / (m - 1)

    dT = np.empty(m)
    dT[1:-1] = (T[2:] - T[:-2]) / 2.0
    dT[0] = T[1] - T[0]
    dT[-1] = T[-1] - T[-2]
    dT = np.maximum(dT, eps)

    pdf = dp / dT
    area = np.trapz(pdf, T)
    if area > 0:
        pdf /= area
    return T, pdf


def fast_spectrum_from_inv_cdf(inv_cdf, D, *, n_samples=20000, sigma_bins=3,
                               x_grid=None):
    """Fast histogram + Gaussian-smoothed density estimate from an inverse CDF.

    Much faster than full KDE: draws samples via inverse-CDF interpolation,
    bins them into a histogram, and smooths with ``gaussian_filter1d``.

    Args:
        inv_cdf:     1-D array, a (possibly non-normalized) inverse CDF.
        D:           number of output bins (typically the spectral dimension).
        n_samples:   number of uniform samples to draw.
        sigma_bins:  std-dev of Gaussian smoothing kernel (in bin units).
        x_grid:      evaluation grid of length D (auto if None).

    Returns:
        x_grid:   evaluation grid of shape (D,).
        density:  smoothed density on *x_grid*, normalized to integrate to 1.
    """
    inv_cdf = np.asarray(inv_cdf, dtype=np.float64)

    # 1) sample via inverse-CDF interpolation
    p_grid = np.linspace(0.0, 1.0, inv_cdf.size)
    u = np.random.rand(n_samples)
    samples = np.interp(u, p_grid, inv_cdf)

    # 2) choose grid
    if x_grid is None:
        x_grid = np.linspace(inv_cdf.min(), inv_cdf.max(), D)

    # 3) histogram onto grid bins
    edges = np.linspace(x_grid[0], x_grid[-1], D + 1)
    counts, _ = np.histogram(samples, bins=edges, density=False)

    dx = edges[1] - edges[0]
    density = counts.astype(np.float64) / (n_samples * dx)

    # 4) smooth in bin-space (approximate KDE)
    density = gaussian_filter1d(density, sigma=sigma_bins, mode="nearest")

    # 5) normalize to integrate to 1
    density /= np.trapz(density, x_grid)
    return x_grid, density




def fast_spectrum_batch(inv_cdf_array, D, *, n_samples=20000, sigma_bins=1.5,
                        x_grid=None):
    """Compute fast histogram densities for an array of inverse-CDF vectors.

    Args:
        inv_cdf_array: 2-D array of shape (N, m), each row an inverse CDF.
        D:             number of output bins per spectrum.
        n_samples:     number of samples per spectrum.
        sigma_bins:    Gaussian smoothing width in bin units.
        x_grid:        shared evaluation grid of length D (auto if None).

    Returns:
        x_grid:    shared evaluation grid of shape (D,).
        densities: array of shape (N, D).
    """
    inv_cdf_array = np.asarray(inv_cdf_array, dtype=np.float64)
    if inv_cdf_array.ndim == 1:
        inv_cdf_array = inv_cdf_array[np.newaxis, :]

    if x_grid is None:
        lo = inv_cdf_array.min()
        hi = inv_cdf_array.max()
        x_grid = np.linspace(lo, hi, D)

    densities = np.empty((len(inv_cdf_array), D))
    for i, row in enumerate(inv_cdf_array):
        _, dens = fast_spectrum_from_inv_cdf(
            row, D, n_samples=n_samples, sigma_bins=sigma_bins, x_grid=x_grid
        )
        densities[i] = dens

    return x_grid, densities


def fd_spectrum_from_inv_cdf(inv_cdf, sigma_bins=0.0, eps=1e-12):
    """Analytic density from an inverse CDF via finite differences.

    No sampling — pure array math.  Optionally smooth the result with a
    Gaussian kernel for parity with the histogram-based approach.

    Args:
        inv_cdf:     1-D array, a (possibly non-normalized) inverse CDF.
        sigma_bins:  if > 0, apply ``gaussian_filter1d`` after the FD step.
        eps:         floor for dT to avoid division by zero.

    Returns:
        x_grid:  the support values (length m, same as input).
        density: density values on *x_grid*, normalized to integrate to 1.
    """
    T = np.maximum.accumulate(np.asarray(inv_cdf, dtype=np.float64))
    m = len(T)
    dp = 1.0 / (m - 1)

    dT = np.empty(m)
    dT[1:-1] = (T[2:] - T[:-2]) / 2.0
    dT[0] = T[1] - T[0]
    dT[-1] = T[-1] - T[-2]
    dT = np.maximum(dT, eps)

    pdf = dp / dT

    if sigma_bins > 0:
        pdf = gaussian_filter1d(pdf, sigma=sigma_bins, mode="nearest")

    area = np.trapz(pdf, T)
    if area > 0:
        pdf /= area
    return T, pdf


def fd_spectrum_batch(inv_cdf_array, sigma_bins=0.0, eps=1e-12):
    """Fully vectorized FD density for an (N, m) array of inverse CDFs.

    No Python loop — operates on the entire array at once using
    broadcasting.  Orders of magnitude faster than per-row approaches.

    Args:
        inv_cdf_array: 2-D array of shape (N, m).
        sigma_bins:    if > 0, smooth each row with ``gaussian_filter1d``.
        eps:           floor for dT.

    Returns:
        x_grids:    array of shape (N, m) — the support per row.
        densities:  array of shape (N, m) — normalized densities.
    """
    arr = np.asarray(inv_cdf_array, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    N, m = arr.shape

    # Enforce monotonicity row-wise
    T = np.maximum.accumulate(arr, axis=1)

    dp = 1.0 / (m - 1)

    # Central differences (interior), one-sided (boundaries)
    dT = np.empty_like(T)
    dT[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / 2.0
    dT[:, 0] = T[:, 1] - T[:, 0]
    dT[:, -1] = T[:, -1] - T[:, -2]
    dT = np.maximum(dT, eps)

    pdf = dp / dT

    # Optional per-row smoothing (the only non-vectorizable part)
    if sigma_bins > 0:
        for i in range(N):
            pdf[i] = gaussian_filter1d(pdf[i], sigma=sigma_bins, mode="nearest")

    # Normalize each row to integrate to 1 via trapezoidal rule
    # trapz along axis=1: use midpoint weights
    dx = np.diff(T, axis=1)                        # (N, m-1)
    mid = (pdf[:, :-1] + pdf[:, 1:]) / 2.0         # (N, m-1)
    areas = (dx * mid).sum(axis=1, keepdims=True)   # (N, 1)
    areas = np.where(areas == 0, 1.0, areas)
    pdf = pdf / areas

    return T, pdf


# ---------------------------------------------------------------------------
# Synthesis / reconstruction
# ---------------------------------------------------------------------------

def synthesis_monotone(x, z, model, *, start_at_zero=True):
    """
    Reconstruct x from codes z using a pre-loaded monotone model.

    Args:
        x: original data tensor.
        z: sparse codes tensor.
        model: loaded SAE model (eval mode).
        start_at_zero: if True, shift reconstructions so they start at 0.

    Returns:
        xhat: reconstructed tensor (on CPU).
        mse:  per-sample MSE tensor (on CPU).
    """
    with torch.no_grad():
        A = model.atoms()
        x = x.to(A.device, dtype=A.dtype)
        z = z.to(A.device, dtype=A.dtype)
        xhat = z @ A
        if start_at_zero:
            xhat = xhat - xhat[..., :1]
        mse = ((xhat - x) ** 2).mean(dim=-1)
    return xhat.cpu(), mse.cpu()


def compute_mu_sigma(data_tensor, eps=1e-6):
    """Mean & std across dim=0, shape [1, D]."""
    data_tensor = data_tensor.float()
    mu = data_tensor.mean(dim=0, keepdim=True)
    sigma = data_tensor.std(dim=0, keepdim=True).clamp_min(eps)
    return mu, sigma


# ---------------------------------------------------------------------------
# MSE for normalized hyperspectral data
# ---------------------------------------------------------------------------

def normalized_hyperspectral_mse(data1, data2):
    """Compute MSE between two datasets of normalized hyperspectral data.

    Each spectrum is normalized to sum to 1 before comparison.

    Args:
        data1: array-like of shape (N, D).
        data2: array-like of shape (N, D).

    Returns:
        per_sample_mse: array of shape (N,), MSE per spectrum.
        mean_mse:       scalar, average over all spectra.
    """
    if torch.is_tensor(data1):
        data1 = data1.detach().cpu().numpy()
    if torch.is_tensor(data2):
        data2 = data2.detach().cpu().numpy()
    data1 = np.asarray(data1, dtype=np.float64)
    data2 = np.asarray(data2, dtype=np.float64)

    sums1 = data1.sum(axis=1, keepdims=True)
    sums2 = data2.sum(axis=1, keepdims=True)
    sums1 = np.where(sums1 == 0, 1.0, sums1)
    sums2 = np.where(sums2 == 0, 1.0, sums2)

    norm1 = data1 / sums1
    norm2 = data2 / sums2

    per_sample_mse = np.mean((norm1 - norm2) ** 2, axis=1)
    mean_mse = per_sample_mse.mean()
    return per_sample_mse, mean_mse


# ---------------------------------------------------------------------------
# 1-D Wasserstein distance for normalized hyperspectral data
# ---------------------------------------------------------------------------

def normalized_hyperspectral_wasserstein(data1, data2, *, grid=None, return_squared=False):
    """
    Compute the 2-Wasserstein distance (W2) between pairs of normalized spectra,
    using POT (Python Optimal Transport).

    Each spectrum is treated as a discrete 1-D distribution supported on a
    uniform grid in [0, 1] (or a user-supplied grid), with masses given by the
    spectrum values normalized to sum to 1.

    Args:
        data1: array-like (N, D)
        data2: array-like (N, D)
        grid:  optional array-like (D,), support locations. If None, uses linspace(0,1,D).
        return_squared: if True, return W2^2 instead of W2

    Returns:
        per_sample: (N,) array of W2 (or W2^2 if return_squared=True)
        mean_val:  scalar mean over samples
    """
    if torch.is_tensor(data1):
        data1 = data1.detach().cpu().numpy()
    if torch.is_tensor(data2):
        data2 = data2.detach().cpu().numpy()
    data1 = np.asarray(data1, dtype=np.float64)
    data2 = np.asarray(data2, dtype=np.float64)

    # Normalize each spectrum to sum to 1 (avoid division by 0)
    sums1 = data1.sum(axis=1, keepdims=True)
    sums2 = data2.sum(axis=1, keepdims=True)
    sums1 = np.where(sums1 == 0, 1.0, sums1)
    sums2 = np.where(sums2 == 0, 1.0, sums2)
    a = data1 / sums1
    b = data2 / sums2

    N, D = a.shape
    if grid is None:
        x = np.linspace(0.0, 1.0, D).astype(np.float64)
    else:
        x = np.asarray(grid, dtype=np.float64)
        if x.shape != (D,):
            raise ValueError(f"grid must have shape ({D},), got {x.shape}")

    per = np.empty(N, dtype=np.float64)

    # POT's wasserstein_1d returns W_p^p (cost) in many versions.
    # We compute cost with p=2, then sqrt unless return_squared=True.
    for i in range(N):
        cost = ot.wasserstein_1d(x, x, a[i], b[i], p=2)
        per[i] = cost if return_squared else np.sqrt(max(cost, 0.0))

    return per, float(per.mean())


# ---------------------------------------------------------------------------
# NMF save / load helpers
# ---------------------------------------------------------------------------

def save_nmf_dict(path, H, metadata=None):
    """Save an NMF dictionary (components matrix) to an .npz file.

    Args:
        path:     file path (will get .npz extension if not present).
        H:        NMF components matrix of shape (rank, D).
        metadata: optional dict of extra info (rank, max_iter, etc.).
    """
    save_dict = {"H": np.asarray(H)}
    if metadata is not None:
        # Store JSON-serialisable metadata as a 0-d object array
        save_dict["metadata"] = np.array(json.dumps(metadata))
    np.savez(str(path), **save_dict)


def load_nmf_dict(path):
    """Load an NMF dictionary from a file saved by ``save_nmf_dict``.

    Args:
        path: .npz file produced by save_nmf_dict.

    Returns:
        H:        components matrix of shape (rank, D).
        metadata: dict (or None if not saved).
    """
    data = np.load(str(path), allow_pickle=False)
    H = data["H"]
    metadata = None
    if "metadata" in data:
        metadata = json.loads(str(data["metadata"]))
    return H, metadata


# ---------------------------------------------------------------------------
# NMF experiment
# ---------------------------------------------------------------------------

def run_nmf_experiment(
    X_train,
    X_val,
    ranks,
    max_iter=1200,
    random_state=42,
    normalize=True,
    verbose=True,
    load_path=None,
    save_dir=None,
):
    """Run NMF at several ranks and report train/val MSE on normalized spectra.

    If *load_path* is given for a rank, the dictionary is loaded from disk
    instead of being fit from scratch.  If *save_dir* is given, every fitted
    (or loaded) dictionary is saved there as ``nmf_rank_{r}.npz``.

    Args:
        X_train:      training data, shape (N_train, D).
        X_val:        validation data, shape (N_val, D).
        ranks:        list of integer ranks to try.
        max_iter:     maximum NMF iterations.
        random_state: seed for reproducibility.
        normalize:    if True, normalize each spectrum to sum to 1.
        verbose:      print progress.
        load_path:    None, a single path/str, or a dict {rank: path}.
                      * None  → fit every rank from scratch.
                      * str/Path → directory; looks for ``nmf_rank_{r}.npz``
                        inside it for each rank.  Ranks without a file are
                        fit from scratch.
                      * dict  → explicit mapping from rank to .npz file.
        save_dir:     if not None, save each dictionary to this directory
                      as ``nmf_rank_{r}.npz`` via ``save_nmf_dict``.

    Returns:
        results: dict  rank -> {train_mse, val_mse, train_rmse, val_rmse,
                                 train_relative_error, val_relative_error,
                                 fit_time_s, transform_time_s, n_iter,
                                 W_train, W_val, H}
    """
    if torch.is_tensor(X_train):
        X_train = X_train.detach().cpu().numpy()
    if torch.is_tensor(X_val):
        X_val = X_val.detach().cpu().numpy()
    X_train = np.asarray(X_train, dtype=np.float64)
    X_val = np.asarray(X_val, dtype=np.float64)

    if normalize:
        train_sums = X_train.sum(axis=1, keepdims=True)
        val_sums = X_val.sum(axis=1, keepdims=True)
        train_sums = np.where(train_sums == 0, 1.0, train_sums)
        val_sums = np.where(val_sums == 0, 1.0, val_sums)
        X_train = X_train / train_sums
        X_val = X_val / val_sums

    X_train = np.clip(X_train, 0, None)
    X_val = np.clip(X_val, 0, None)

    # ── resolve load_path into a dict {rank: filepath_or_None} ──
    load_map = {}
    if load_path is not None:
        if isinstance(load_path, dict):
            load_map = {int(k): Path(v) for k, v in load_path.items()}
        else:
            ld = Path(load_path)
            for r in ranks:
                candidate = ld / f"nmf_rank_{r}.npz"
                if candidate.exists():
                    load_map[r] = candidate

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    print_every = 50
    results = {}

    for r in ranks:
        # ── try loading a pre-fitted dictionary ──
        if r in load_map:
            if verbose:
                print(f"\nLoading NMF rank {r} from {load_map[r]}")
            H, _meta = load_nmf_dict(load_map[r])
            assert H.shape[0] == r, (
                f"Loaded H has {H.shape[0]} components but rank={r} requested"
            )
            t0 = time.time()
            # Project train & val onto the loaded dictionary
            nmf_proj = NMF(
                n_components=r, init="custom", max_iter=1,
                random_state=random_state, verbose=0,
            )
            # fix H, solve for W only via transform
            nmf_proj.components_ = H
            nmf_proj.n_components = r
            W = nmf_proj.transform(X_train)
            W_val = nmf_proj.transform(X_val)
            fit_time = time.time() - t0
            total_iters = 0
        else:
            # ── fit from scratch ──
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Fitting NMF with rank r={r}")
                print(f"{'=' * 60}")

            t0 = time.time()
            W, H = None, None
            total_iters = 0

            for chunk_start in range(0, max_iter, print_every):
                chunk_size = min(print_every, max_iter - chunk_start)

                if W is None:
                    model = NMF(
                        n_components=r,
                        init="nndsvda",
                        max_iter=chunk_size,
                        random_state=random_state,
                        verbose=0,
                    )
                    W = model.fit_transform(X_train)
                else:
                    model = NMF(
                        n_components=r,
                        init="custom",
                        max_iter=chunk_size,
                        random_state=random_state,
                        verbose=0,
                    )
                    W = model.fit_transform(X_train, W=W, H=H)

                H = model.components_
                total_iters += model.n_iter_

                if verbose:
                    recon_err = np.sqrt(np.mean((X_train - W @ H) ** 2))
                    print(
                        f"  Iter {total_iters:>5}/{max_iter} | "
                        f"Train RMSE: {recon_err:.6f}"
                    )

                if model.n_iter_ < chunk_size:
                    if verbose:
                        print(f"  Converged at iteration {total_iters}")
                    break

            fit_time = time.time() - t0

            t1 = time.time()
            # Re-project val using the last sklearn model object
            W_val = model.transform(X_val)
            transform_time_extra = time.time() - t1
            fit_time += transform_time_extra  # include in reported time

        # ── evaluate ──
        X_train_recon = W @ H
        train_mse = float(np.mean((X_train - X_train_recon) ** 2))
        train_rmse = float(np.sqrt(train_mse))
        train_rel_err = float(
            np.linalg.norm(X_train - X_train_recon) / np.linalg.norm(X_train)
        )

        X_val_recon = W_val @ H
        val_mse = float(np.mean((X_val - X_val_recon) ** 2))
        val_rmse = float(np.sqrt(val_mse))
        val_rel_err = float(
            np.linalg.norm(X_val - X_val_recon) / np.linalg.norm(X_val)
        )

        _, norm_train_mse = normalized_hyperspectral_mse(X_train, X_train_recon)
        _, norm_val_mse = normalized_hyperspectral_mse(X_val, X_val_recon)

        results[r] = {
            "train_mse": train_mse,
            "train_rmse": train_rmse,
            "train_relative_error": train_rel_err,
            "val_mse": val_mse,
            "val_rmse": val_rmse,
            "val_relative_error": val_rel_err,
            "norm_train_mse": float(norm_train_mse),
            "norm_val_mse": float(norm_val_mse),
            "fit_time_s": float(fit_time),
            "n_iter": int(total_iters),
            "W_train": W,
            "W_val": W_val,
            "H": H,
        }

        # ── save dictionary ──
        if save_dir is not None:
            out_path = save_dir / f"nmf_rank_{r}.npz"
            meta = {
                "rank": r,
                "max_iter": max_iter,
                "random_state": random_state,
                "n_iter": total_iters,
                "train_mse": train_mse,
                "val_mse": val_mse,
            }
            save_nmf_dict(out_path, H, metadata=meta)
            if verbose:
                print(f"  Saved dictionary → {out_path}")

        if verbose:
            print(f"Fit time: {fit_time:.1f}s, total iterations: {total_iters}")
            print(
                f"Train MSE: {train_mse:.6f} | RMSE: {train_rmse:.6f} | "
                f"Rel Error: {train_rel_err:.4f}"
            )
            print(
                f"Val   MSE: {val_mse:.6f} | RMSE: {val_rmse:.6f} | "
                f"Rel Error: {val_rel_err:.4f}"
            )

    if verbose:
        print(f"\n{'=' * 60}")
        print("Summary")
        print(f"{'=' * 60}")
        print(
            f"{'Rank':>6} | {'Train RMSE':>12} | {'Val RMSE':>12} | "
            f"{'Train RelErr':>12} | {'Val RelErr':>12} | {'Time (s)':>8}"
        )
        print("-" * 75)
        for r in ranks:
            res = results[r]
            print(
                f"{r:>6} | {res['train_rmse']:>12.6f} | {res['val_rmse']:>12.6f} | "
                f"{res['train_relative_error']:>12.4f} | "
                f"{res['val_relative_error']:>12.4f} | {res['fit_time_s']:>8.1f}"
            )

    return results

'''
def drop_random_bands(spectra, k, rng=None):
    """Zero out randomly chosen bands until at least fraction k of each
    spectrum's total mass has been removed.

    Parameters
    ----------
    spectra : ndarray, shape (N, M)
    k : float in (0, 1]
        Target fraction of mass to remove per spectrum.
    rng : np.random.Generator, optional

    Returns
    -------
    corrupted : ndarray, same shape
    masks : bool ndarray, True where bands were zeroed
    """
    if rng is None:
        rng = np.random.default_rng()
    N, M = spectra.shape

    # Random permutation per row: shuffle column indices
    # argsort of random values gives a random permutation per row
    noise = rng.random((N, M))
    order = np.argsort(noise, axis=1)

    # Gather values in permuted order
    gathered = np.take_along_axis(spectra, order, axis=1)

    # Cumulative mass in removal order
    cumsum = np.cumsum(gathered, axis=1)
    totals = spectra.sum(axis=1, keepdims=True)

    # Mask: remove bands up to and including the one that crosses the threshold
    # For zero-mass rows, totals=0 so threshold=0, nothing gets removed
    threshold = k * totals
    remove_in_order = cumsum <= threshold
    # Include the band that crosses the threshold
    # (shift right: if cumsum[j-1] < threshold, band j should be removed)
    crosses = (np.roll(cumsum, 1, axis=1) < threshold)
    crosses[:, 0] = True  # first band is always a candidate
    remove_in_order = remove_in_order | (crosses & ~remove_in_order)

    # Scatter back to original band positions
    masks = np.zeros((N, M), dtype=bool)
    np.put_along_axis(masks, order, remove_in_order, axis=1)

    # Zero-mass rows: don't corrupt
    zero_rows = totals.ravel() <= 1e-12
    masks[zero_rows] = False

    corrupted = spectra.copy()
    corrupted[masks] = 0.0
    return corrupted, masks
'''


def pushforward_log_warp(x, a):
    """
    Pushforward warp on λ ∈ [0,1] for the log-type monotone map

        phi(t) = log(1 + a t) / log(1 + a),    a > 0

    Density pushforward formula on the uniform grid λ:
        y(λ) = x(phi^{-1}(λ)) * d/dλ phi^{-1}(λ)

    Args:
        x: (B,) nonnegative spectrum sampled on uniform λ-grid in [0,1]
        a: positive warp strength parameter

    Returns:
        y: (B,) warped spectrum on the same λ-grid
    """
    a = float(a)
    if a <= 0:
        raise ValueError("log warp requires a > 0")

    B = x.size
    lam = np.linspace(0.0, 1.0, B)

    denom = np.log1p(a)

    # phi^{-1}(u) = (exp(u * log(1+a)) - 1) / a
    t_inv = np.expm1(lam * denom) / a

    # (phi^{-1})'(u) = (log(1+a) * exp(u * log(1+a))) / a
    jac = (denom * np.exp(lam * denom)) / a

    # interpolate x at locations t_inv
    x_interp = np.interp(t_inv, lam, x)

    y = np.clip(x_interp * jac, 0.0, None)

    # optional: renormalize to preserve total mass (useful if x is treated as a density)
    s0 = x.sum()
    s1 = y.sum()
    if s1 > 0:
        y *= (s0 / s1)

    return y

def drop_contiguous_bands(spectra, k, rng=None):
    """Zero out a contiguous block of bands (wrapping) that captures at least
    fraction k of each spectrum's total mass.

    Parameters
    ----------
    spectra : ndarray, shape (N, M)
    k : float in (0, 1]
        Target fraction of mass to remove per spectrum.
    rng : np.random.Generator, optional

    Returns
    -------
    corrupted : ndarray, same shape
    masks : bool ndarray, True where bands were zeroed
    """
    if rng is None:
        rng = np.random.default_rng()
    N, M = spectra.shape

    # Tile spectra to handle wrapping: [band0..bandM-1, band0..bandM-1]
    tiled = np.concatenate([spectra, spectra], axis=1)  # (N, 2M)

    # Random start per row
    starts = rng.integers(0, M, size=N)

    # Cumulative sums from each row's start using the tiled array
    # Shift each row so position 0 = start band
    col_idx = (starts[:, None] + np.arange(M)[None, :]) % (2 * M)  # (N, M)
    shifted = np.take_along_axis(tiled, col_idx, axis=1)  # (N, M)

    cumsum = np.cumsum(shifted, axis=1)
    totals = spectra.sum(axis=1, keepdims=True)
    threshold = k * totals

    # Number of bands to remove per row: first index where cumsum >= threshold
    reached = cumsum >= threshold
    # argmax on bool gives first True; if none True, returns 0
    first_reached = np.argmax(reached, axis=1)
    # Handle rows that never reach threshold (remove all M bands)
    never_reached = ~reached.any(axis=1)
    first_reached[never_reached] = M - 1
    n_remove = first_reached + 1  # (N,)

    # Build masks
    band_positions = np.arange(M)[None, :]  # (1, M)
    remove_in_shifted = band_positions < n_remove[:, None]  # (N, M)

    # Map back to original band indices
    orig_idx = col_idx % M
    masks = np.zeros((N, M), dtype=bool)
    np.put_along_axis(masks, orig_idx, remove_in_shifted, axis=1)

    # Zero-mass rows: don't corrupt
    zero_rows = totals.ravel() <= 1e-12
    masks[zero_rows] = False

    corrupted = spectra.copy()
    corrupted[masks] = 0.0
    return corrupted, masks



def drop_random_bands(spectra, k, rng=None):
    """Zero out k randomly chosen bands in each spectrum."""
    if rng is None:
        rng = np.random.default_rng()
    N, M = spectra.shape
    assert k <= M, f"k={k} exceeds number of bands M={M}"
    corrupted = spectra.copy()
    masks = np.zeros((N, M), dtype=bool)
    for i in range(N):
        idx = rng.choice(M, size=k, replace=False)
        corrupted[i, idx] = 0.0
        masks[i, idx] = True
    return corrupted, masks

'''

def drop_contiguous_bands(spectra, k, rng=None):
    """Zero out a contiguous block of k bands in each spectrum."""
    if rng is None:
        rng = np.random.default_rng()
    N, M = spectra.shape
    assert k <= M, f"k={k} exceeds number of bands M={M}"
    corrupted = spectra.copy()
    masks = np.zeros((N, M), dtype=bool)
    for i in range(N):
        start = rng.integers(0, M - k + 1)
        corrupted[i, start:start + k] = 0.0
        masks[i, start:start + k] = True
    return corrupted, masks
'''


# ---------------------------------------------------------------------------
# Shift corruption helpers
# ---------------------------------------------------------------------------


def _renorm_rows(X, eps=1e-12):
    return X / np.maximum(X.sum(axis=1, keepdims=True), eps)

def _as_2d(spectra):
    X = np.asarray(spectra, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2:
        raise ValueError("spectra must be 1-D or 2-D")
    return X

def _apply_mask_shift(X, frac, direction, mask, shift=1):
    """
    Apply a mass shift of `shift` bins only on columns where `mask` is True.

    For each selected band i:
      - keep (1-frac)*mass at i
      - move frac*mass to i+shift (right) or i-shift (left) if that target
        band is also selected by mask; otherwise that moved mass is dropped.
    """
    if shift < 0:
        raise ValueError("shift must be >= 0")
    if shift == 0 or frac == 0.0:
        return X.copy()

    Y = X.copy()
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return Y

    # If shift is so big you can't land within the masked set, you just scale down.
    if len(idx) <= shift:
        Y[:, idx] = (1 - frac) * X[:, idx]
        return Y

    sub = X[:, idx]                 # (N, K)
    shifted = (1 - frac) * sub      # mass that stays

    k = int(shift)
    if direction == "right":
        # mass from positions 0..K-k-1 moves to k..K-1
        shifted[:, k:] += frac * sub[:, :-k]
    elif direction == "left":
        # mass from positions k..K-1 moves to 0..K-k-1
        shifted[:, :-k] += frac * sub[:, k:]
    else:
        raise ValueError("direction must be 'left' or 'right'")

    Y[:, idx] = shifted
    return Y

def shift_mass_global(spectra, frac=0.2, shift=1, direction=None, renorm=True, rng=None):
    rng = rng or np.random.default_rng()
    X = _as_2d(spectra)
    direction = direction or rng.choice(["left", "right"])
    mask = np.ones(X.shape[1], dtype=bool)
    Y = _apply_mask_shift(X, frac, direction, mask, shift=shift)
    Y = np.clip(Y, 0.0, None)
    if renorm:
        Y = _renorm_rows(Y)
    return Y, {"direction": direction, "frac": float(frac), "shift": int(shift)}

def shift_mass_halfspace(spectra, frac=0.2, shift=1, direction=None, split_idx=51, renorm=True, rng=None):
    rng = rng or np.random.default_rng()
    X = _as_2d(spectra)
    M = X.shape[1]
    direction = direction or rng.choice(["left", "right"])
    mask = np.zeros(M, dtype=bool)
    if direction == "right":
        mask[split_idx + 1:] = True
    else:
        mask[:split_idx] = True
    Y = _apply_mask_shift(X, frac, direction, mask, shift=shift)
    Y = np.clip(Y, 0.0, None)
    if renorm:
        Y = _renorm_rows(Y)
    return Y, {"direction": direction, "frac": float(frac), "shift": int(shift), "split_idx": int(split_idx)}

def shift_mass_random_split(spectra, frac=0.2, shift=1, direction=None,
                            split_low=0, split_high=None, renorm=True, rng=None):
    rng = rng or np.random.default_rng()
    X = _as_2d(spectra)
    N, M = X.shape
    split_high = split_high if split_high is not None else M - 1

    splits = rng.integers(split_low, split_high + 1, size=N)
    if direction is None:
        dirs = rng.choice(["left", "right"], size=N)
    else:
        dirs = np.full(N, direction)

    Y = X.copy()
    for d in ("left", "right"):
        d_mask = (dirs == d)
        if not np.any(d_mask):
            continue
        for s in np.unique(splits[d_mask]):
            rows = d_mask & (splits == s)
            band_mask = np.zeros(M, dtype=bool)
            if d == "right":
                band_mask[s + 1:] = True
            else:
                band_mask[:s] = True
            Y[rows] = _apply_mask_shift(X[rows], frac, d, band_mask, shift=shift)

    Y = np.clip(Y, 0.0, None)
    if renorm:
        Y = _renorm_rows(Y)
    return Y, {
        "frac": float(frac),
        "shift": int(shift),
        "split_idx_per_row": splits,
        "direction_per_row": dirs
    }
