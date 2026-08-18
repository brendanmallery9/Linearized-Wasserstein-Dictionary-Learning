"""
Baselines and shared evaluation metrics for the minimal point-cloud ablation
(LWDL-EOT vs. PCA vs. conventional sparse coding).

Everything here operates on the *same* flattened displacement representation so
the three methods are directly comparable:

    displacement field   V_i = T_i - X          (per sample, shape (n, d))
    flattened            v_i = V_i.reshape(n*d)  (shape (n*d,))

PCA and sparse coding are fit on the training split only and then evaluated on
train/test; the LWDL evaluator (evaluate_pointcloud_lwdl_for_ablation.py) reuses
the metric helpers here so every row of the final table is computed identically.

Native reconstruction quality is reported in three spaces:
  * map-space L2         -- matches the LWDL training loss convention
  * Wasserstein          -- OT distance between true / reconstructed clouds
  * Chamfer (optional)   -- symmetric nearest-neighbour distance

Nothing here mutates any existing result directory.
"""

import time

import numpy as np
import torch


# ============================================================
# (Un)flatten displacement maps
# ============================================================

def flatten_displacement_maps(X, maps):
    """
    Convert stacked transport maps to flattened displacement vectors.

    Args:
        X:    (n, d) base measure support.
        maps: (N, n, d) transport maps.

    Returns:
        (N, n*d) flattened displacement fields  V = maps - X.
    """
    N = maps.shape[0]
    return (maps - X[None]).reshape(N, -1)


def unflatten_displacement_maps(X, flat):
    """
    Inverse of flatten_displacement_maps: rebuild transport maps.

    Args:
        X:    (n, d) base measure support.
        flat: (N, n*d) flattened displacement fields.

    Returns:
        (N, n, d) reconstructed transport maps  X + V.
    """
    n, d = X.shape
    N = flat.shape[0]
    return X[None] + flat.reshape(N, n, d)


# ============================================================
# Small conversion helpers
# ============================================================

def _to_torch(x, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ============================================================
# Reconstruction losses
# ============================================================

def map_l2_loss(true_maps, recon_maps):
    """
    Mean map-space reconstruction loss, matching the LWDL convention:

        L = mean_i [ 0.5 * ||T_i - T_hat_i||^2 / n ]

    where the squared norm sums over both the support (n) and coordinate (d)
    axes.  Accepts torch tensors or numpy arrays.
    """
    T = _to_torch(true_maps)
    That = _to_torch(recon_maps)
    n = T.shape[1]
    diff_sq = ((T - That) ** 2).sum(dim=(1, 2))
    per_sample = 0.5 * diff_sq / n
    return float(per_sample.mean().item())


def chamfer_loss(true_clouds, recon_clouds, max_samples=None):
    """
    Mean symmetric Chamfer distance between corresponding point clouds.

    For each pair (A, B) with A, B shaped (n, d):
        cd = mean_a min_b ||a - b|| + mean_b min_a ||a - b||
    using Euclidean distances (torch.cdist).  Returned value is the mean over
    samples.

    Args:
        true_clouds:  (N, n, d)
        recon_clouds: (N, n, d)
        max_samples:  optionally cap the number of pairs evaluated.
    """
    A = _to_torch(true_clouds)
    B = _to_torch(recon_clouds)
    N = A.shape[0]
    if max_samples is not None and max_samples > 0:
        N = min(N, int(max_samples))
    total = 0.0
    with torch.no_grad():
        for i in range(N):
            D = torch.cdist(A[i], B[i])           # (n, m) Euclidean
            forward = D.min(dim=1).values.mean()
            backward = D.min(dim=0).values.mean()
            total += float((forward + backward).item())
    return total / max(N, 1)


def wasserstein_loss(true_clouds, recon_clouds, metric="sqeuclidean",
                     max_samples=None):
    """
    Mean Wasserstein / EMD reconstruction distance between corresponding clouds.

    Uses POT (`ot.emd2`) with uniform masses on both supports.  The `metric`
    argument selects the ground cost and is recorded by the caller:
        "sqeuclidean" -> emd2 returns W_2^2 (squared 2-Wasserstein)
        "euclidean"   -> emd2 returns W_1   (1-Wasserstein)

    Args:
        true_clouds:  (N, n, d)
        recon_clouds: (N, n, d)
        metric:       "sqeuclidean" or "euclidean".
        max_samples:  optionally cap the number of pairs evaluated.

    Returns:
        (mean_distance, n_evaluated)
    """
    import ot

    if metric not in ("sqeuclidean", "euclidean"):
        raise ValueError(f"Unknown wasserstein metric: {metric!r}")

    A = _to_numpy(true_clouds).astype(np.float64)
    B = _to_numpy(recon_clouds).astype(np.float64)
    N = A.shape[0]
    if max_samples is not None and max_samples > 0:
        N = min(N, int(max_samples))

    total = 0.0
    for i in range(N):
        a_pts, b_pts = A[i], B[i]
        wa = np.full(a_pts.shape[0], 1.0 / a_pts.shape[0])
        wb = np.full(b_pts.shape[0], 1.0 / b_pts.shape[0])
        M = ot.dist(a_pts, b_pts, metric=metric)
        total += float(ot.emd2(wa, wb, M))
    return total / max(N, 1), N


# ============================================================
# Sparsity statistics
# ============================================================

def sparsity_stats(codes, threshold=1e-8):
    """
    Summarize the sparsity of a code matrix.

    Args:
        codes:     (N, m) coefficient matrix.
        threshold: magnitude below which a coefficient counts as zero.

    Returns:
        dict with mean_l0, mean_l1, frac_nonzero.
    """
    C = _to_numpy(codes)
    nz = np.abs(C) > threshold
    mean_l0 = float(nz.sum(axis=1).mean())
    mean_l1 = float(np.abs(C).sum(axis=1).mean())
    frac_nonzero = float(nz.mean())
    return {
        "mean_l0": mean_l0,
        "mean_l1": mean_l1,
        "frac_nonzero": frac_nonzero,
    }


# ============================================================
# Linear probe (representation quality)
# ============================================================

def linear_probe(train_codes, train_labels, test_codes, test_labels,
                 seed=42, max_iter=2000):
    """
    Fit a standardized logistic-regression probe on the codes and score it.

    Returns:
        dict with accuracy and macro_f1 (on the test split).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Xtr = _to_numpy(train_codes)
    Xte = _to_numpy(test_codes)
    ytr = _to_numpy(train_labels).astype(int).ravel()
    yte = _to_numpy(test_labels).astype(int).ravel()

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter, random_state=seed),
    )
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    return {
        "accuracy": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro")),
    }


# ============================================================
# Baseline 1: PCA on flattened displacement maps
# ============================================================

def run_pca_baseline(train_flat, test_flat, m, seed=42):
    """
    Dense linear PCA baseline.

    Fits sklearn PCA with `m` components on the train split only, projects both
    splits into codes, and reconstructs via the inverse transform.

    Returns dict with:
        train_codes, test_codes        -- (N, m)
        train_recon_flat, test_recon_flat -- (N, n*d)
        explained_variance_ratio_sum
        runtime_seconds
        n_components
    """
    from sklearn.decomposition import PCA

    Xtr = _to_numpy(train_flat).astype(np.float64)
    Xte = _to_numpy(test_flat).astype(np.float64)

    m = int(min(m, Xtr.shape[0], Xtr.shape[1]))
    t0 = time.time()
    pca = PCA(n_components=m, random_state=seed)
    train_codes = pca.fit_transform(Xtr)
    test_codes = pca.transform(Xte)
    train_recon = pca.inverse_transform(train_codes)
    test_recon = pca.inverse_transform(test_codes)
    runtime = time.time() - t0

    return {
        "train_codes": train_codes,
        "test_codes": test_codes,
        "train_recon_flat": train_recon,
        "test_recon_flat": test_recon,
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "runtime_seconds": runtime,
        "n_components": m,
    }


# ============================================================
# Baseline 2: conventional sparse coding / dictionary learning
# ============================================================

DEFAULT_SPARSE_ALPHAS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]


def run_sparse_coding_baseline(train_flat, test_flat, m, target_l0, seed=42,
                               alphas=None, max_iter=200, batch_size=64):
    """
    Unconstrained sparse dictionary-learning baseline.

    Uses sklearn MiniBatchDictionaryLearning with a width-`m` dictionary fit on
    the train split only.  The sparsity penalty `alpha` is swept over a small
    log grid; for each alpha the dictionary is fit, both splits transformed
    (lasso_lars sparse codes), and the resulting train mean L0 recorded.  The
    alpha whose train mean L0 is closest to `target_l0` is selected.

    Reconstruction uses  recon = codes @ components_.

    Returns dict with the same reconstruction/code fields as run_pca_baseline
    plus the chosen alpha and per-split mean L0.
    """
    from sklearn.decomposition import MiniBatchDictionaryLearning

    Xtr = _to_numpy(train_flat).astype(np.float64)
    Xte = _to_numpy(test_flat).astype(np.float64)
    if alphas is None:
        alphas = DEFAULT_SPARSE_ALPHAS
    m = int(min(m, Xtr.shape[1]))

    t0 = time.time()
    trials = []
    for alpha in alphas:
        dl = MiniBatchDictionaryLearning(
            n_components=m,
            alpha=float(alpha),
            max_iter=int(max_iter),
            batch_size=int(batch_size),
            fit_algorithm="lars",
            transform_algorithm="lasso_lars",
            transform_alpha=float(alpha),
            random_state=seed,
        )
        dl.fit(Xtr)
        train_codes = dl.transform(Xtr)
        mean_l0 = float((np.abs(train_codes) > 1e-8).sum(axis=1).mean())
        trials.append({
            "alpha": float(alpha),
            "train_mean_l0": mean_l0,
            "dl": dl,
            "train_codes": train_codes,
        })
        print(f"    [sparse-coding] alpha={alpha:g}  train_mean_L0={mean_l0:.3f}",
              flush=True)

    best = min(trials, key=lambda tr: abs(tr["train_mean_l0"] - float(target_l0)))
    dl = best["dl"]
    train_codes = best["train_codes"]
    test_codes = dl.transform(Xte)
    components = dl.components_                       # (m, n*d)
    train_recon = train_codes @ components
    test_recon = test_codes @ components
    runtime = time.time() - t0

    train_mean_l0 = best["train_mean_l0"]
    test_mean_l0 = float((np.abs(test_codes) > 1e-8).sum(axis=1).mean())

    return {
        "train_codes": train_codes,
        "test_codes": test_codes,
        "train_recon_flat": train_recon,
        "test_recon_flat": test_recon,
        "chosen_alpha": best["alpha"],
        "target_l0": float(target_l0),
        "train_mean_l0": train_mean_l0,
        "test_mean_l0": test_mean_l0,
        "alpha_grid": [float(a) for a in alphas],
        "runtime_seconds": runtime,
        "n_components": m,
    }
