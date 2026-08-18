"""
3D analog of MNIST_WDL_run_experiments.py for point clouds.

Trains TransportMapSAE / DisplacementFieldSAE models on transport maps from a
uniform base measure on [0,1]^3 to point-cloud measures (one per shape sample).

Key differences from the 2D MNIST version:
  - Maps live in [0,1]^3 instead of [0,1]^2.
  - The atom target grid Y is sampled from a *mixture of point clouds* in the
    dataset by default (sample_grid_from_cloud_mixture), not from a uniform
    cube grid -- a uniform 3D grid blows up fast (grid_side=16 -> 4096 points).
    You can still pick "uniform" if you want a regular cube grid for comparison.

Reuses mnist_sae_models.py wholesale: GibbsAtoms / SinkhornAtoms infer dim from
the base measure, all einsums are dimension-agnostic.

Usage:
    python wdl_repo/pointcloud/pipeline/pointcloud_run_experiments.py \
        --data_dir wdl_repo/datasets/geomshapes_ot \
        --output_dir wdl_repo/pointcloud/results/geomshapes
"""

import json
import math
import sys
import time
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MNIST_PIPELINE = REPO_ROOT / "mnist" / "pipeline"
for p in (str(REPO_ROOT), str(MNIST_PIPELINE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

from mnist_sae_models import (
    CenteredDisplacementFieldSAE,
    DisplacementFieldSAE,
    PCRemovedCenteredDisplacementFieldSAE,
    TransportMapSAE,
    WhitenedCenteredDisplacementFieldSAE,
    make_grid_nd,
)
from pointcloud.pipeline.pointcloud_data import (
    load_transport_maps,
    make_dataloaders,
)


# ============================================================
# Device helpers (identical structure to the MNIST runner)
# ============================================================

def resolve_device(device_str):
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_str == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    if device_str == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("MPS not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


def make_gpu_list(device_str, gpu_ids):
    if gpu_ids and device_str in ("cuda", "auto") and torch.cuda.is_available():
        return [torch.device(f"cuda:{gid}") for gid in gpu_ids]
    return [resolve_device(device_str)]


# ============================================================
# Atom-grid construction
# ============================================================

def sample_grid_from_cloud_mixture(data_dir, n_clouds=10, support_size=3000,
                                   classes=None, seed=42):
    """
    Build the atom target grid Y by drawing points from a random subset of
    the dataset's point clouds.

    1. Load the per-class mappings (each map is shaped (n_base, 3) and lives
       in [0,1]^3 because we normalized clouds to the unit cube during prep).
    2. Pick `n_clouds` cloud samples uniformly at random across all classes.
    3. Concatenate them into a big point set, then uniformly subsample
       `support_size` points without replacement.

    Returns:
        Y: torch.Tensor (support_size, 3) -- atom target grid points.
    """
    rng = np.random.RandomState(seed)
    _, maps = load_transport_maps(data_dir, classes=classes)
    N, n, d = maps.shape

    n_clouds = min(n_clouds, N)
    chosen = rng.choice(N, size=n_clouds, replace=False)
    pool = maps[chosen].reshape(-1, d).numpy()  # (n_clouds * n, 3)

    if support_size > pool.shape[0]:
        support_size = pool.shape[0]
    pick = rng.choice(pool.shape[0], size=support_size, replace=False)
    Y = pool[pick].astype("float32")
    print(f"Atom grid: sampled {support_size} points from {n_clouds} "
          f"cloud(s) (pool size={pool.shape[0]})")
    return torch.tensor(Y)


# ============================================================
# Default config (mirrors MNIST_WDL_run_experiments DEFAULT_CONFIG)
# ============================================================

DEFAULT_CONFIG = dict(
    # Data (resolved to absolute paths below so cwd doesn't matter)
    data_dir=str(REPO_ROOT / "datasets" / "geomshapes_ot"),
    classes=None,            # None = all classes found on disk
    test_fraction=0.1,
    seed=42,
    model_seed=None,

    # Model / atoms
    m=20,
    grid_side=10,            # only used if grid_mode="uniform" (10^3 = 1000 pts)
    grid_mode="cloud_mixture",   # "uniform" or "cloud_mixture"
    grid_n_clouds=10,
    grid_support_size=3000,
    lista_steps=20,
    activation_type="relu",  # "relu" | "jumprelu" | "final_jumprelu" | "topk" | "topk_simplex" | "softtopk" | "final_softtopk" | "final_softtopk_simplex"
    topk_k=3,                # k for topk / topk_simplex / softtopk
    softtopk_tau=0.1,
    softtopk_tau_start=None,
    softtopk_tau_end=None,
    softtopk_tau_anneal_epochs=None,

    # Training
    batch_size=64,
    epochs=2000,
    lr=1e-3,
    optimizer="adamw",
    weight_decay=0.1,
    scheduler="none",
    lr_min=0.0,
    grad_clip_norm=None,
    sparsity_warmup=0,
    init_checkpoint=None,

    # Sweep
    epsilons=[0.025],
    sparsity_coeffs=[1e-4],
    methods=["displacement"],
    displacement_center_mode="generator_mean",
    displacement_pc_index=1,
    displacement_pc_count=1,
    displacement_pc_mode="dataset_pca",
    whitening_mode="dataset_zca",
    whitening_eps=1e-4,
    whitening_max_samples=None,

    # Output
    output_dir=str(REPO_ROOT / "pointcloud" / "results" / "geomshapes"),

    # Device
    device="auto",
    gpu_ids=None,
)


# ============================================================
# Loss / training (same structure as the 2D version)
# ============================================================

def compute_losses(model, loader, sparsity_coeff, device):
    model.eval()
    n = model.n
    total_recon = 0.0
    total_l1 = 0.0
    total_active = 0.0
    total_samples = 0

    with torch.no_grad():
        for T_batch in loader:
            T_batch, aux_batch = unpack_transport_batch(T_batch, device)
            B = T_batch.shape[0]
            T_hat, lam = (
                model(T_batch, aux_batch)
                if aux_batch is not None else model(T_batch)
            )

            diff_sq = ((T_batch - T_hat) ** 2).sum(dim=(1, 2))
            recon_per_sample = 0.5 * diff_sq / n
            l1_per_sample = lam.abs().sum(dim=1)
            active_per_sample = (lam > 0).float().sum(dim=1)

            total_recon += recon_per_sample.sum().item()
            total_l1 += l1_per_sample.sum().item()
            total_active += active_per_sample.sum().item()
            total_samples += B

    mean_recon = total_recon / total_samples
    mean_l1 = total_l1 / total_samples
    mean_active = total_active / total_samples
    total_loss = mean_recon + sparsity_coeff * mean_l1
    model.train()
    return {
        "recon_loss": mean_recon,
        "total_loss": total_loss,
        "mean_l1": mean_l1,
        "mean_active": mean_active,
    }


def clone_model_state(model):
    """Copy a state_dict onto CPU so later optimizer steps cannot mutate it."""
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def load_displacement_center(data_dir, X, maps, mode="generator_mean"):
    """Fixed residual baseline for `displacement_centered` runs."""
    data_dir = Path(data_dir)
    if mode in (None, "none"):
        return torch.zeros_like(X)
    if mode == "generator_mean":
        base_maps_path = data_dir / "base_maps.pt"
        if base_maps_path.exists():
            base_maps = torch.load(base_maps_path, map_location="cpu").float()
            return base_maps.mean(dim=0) - X
        print("  Warning: base_maps.pt not found; using dataset mean displacement center")
        return maps.mean(dim=0) - X
    if mode == "dataset_mean":
        return maps.mean(dim=0) - X
    raise ValueError(f"Unknown displacement_center_mode: {mode}")


def load_displacement_whitening(data_dir, X, maps, displacement_center,
                                mode="dataset_zca", eps=1e-4,
                                max_samples=None, seed=42):
    """Build a fixed ZCA whitening / unwhitening pair for centered residuals."""
    D = X.numel()
    if mode in (None, "none"):
        eye = torch.eye(D, dtype=X.dtype)
        return eye, eye

    data_dir = Path(data_dir)
    if mode == "dataset_zca":
        source = maps
    elif mode == "generator_zca":
        base_maps_path = data_dir / "base_maps.pt"
        if not base_maps_path.exists():
            raise FileNotFoundError(
                f"whitening_mode=generator_zca requires {base_maps_path}"
            )
        source = torch.load(base_maps_path, map_location="cpu").float()
    else:
        raise ValueError(f"Unknown whitening_mode: {mode}")

    residuals = source.float() - X.unsqueeze(0) - displacement_center.unsqueeze(0)
    if max_samples is not None and max_samples > 0 and residuals.shape[0] > max_samples:
        gen = torch.Generator().manual_seed(int(seed))
        idx = torch.randperm(residuals.shape[0], generator=gen)[: int(max_samples)]
        residuals = residuals[idx]

    R = residuals.reshape(residuals.shape[0], -1).double()
    second_moment = (R.T @ R) / max(R.shape[0], 1)
    evals, evecs = torch.linalg.eigh(second_moment)
    evals = evals.clamp(min=0.0)
    eps = float(eps)
    inv_sqrt = torch.rsqrt(evals + eps)
    sqrt = torch.sqrt(evals + eps)
    W = (evecs * inv_sqrt.unsqueeze(0)) @ evecs.T
    Winv = (evecs * sqrt.unsqueeze(0)) @ evecs.T
    return W.to(dtype=X.dtype), Winv.to(dtype=X.dtype)


def load_displacement_pc_components(data_dir, X, maps, displacement_center,
                                    mode="dataset_pca", pc_index=1, pc_count=1,
                                   max_samples=None, seed=42):
    """Return unit principal components of centered residual fields."""
    data_dir = Path(data_dir)
    if mode == "dataset_pca":
        source = maps
    elif mode == "generator_pca":
        base_maps_path = data_dir / "base_maps.pt"
        if not base_maps_path.exists():
            raise FileNotFoundError(
                f"displacement_pc_mode=generator_pca requires {base_maps_path}"
            )
        source = torch.load(base_maps_path, map_location="cpu").float()
    else:
        raise ValueError(f"Unknown displacement_pc_mode: {mode}")

    residuals = source.float() - X.unsqueeze(0) - displacement_center.unsqueeze(0)
    if max_samples is not None and max_samples > 0 and residuals.shape[0] > max_samples:
        gen = torch.Generator().manual_seed(int(seed))
        idx = torch.randperm(residuals.shape[0], generator=gen)[: int(max_samples)]
        residuals = residuals[idx]

    R = residuals.reshape(residuals.shape[0], -1).double()
    second_moment = (R.T @ R) / max(R.shape[0], 1)
    evals, evecs = torch.linalg.eigh(second_moment)
    order = torch.argsort(evals, descending=True)
    pc_index = int(pc_index)
    pc_count = int(pc_count)
    if pc_count <= 0:
        raise ValueError(f"displacement_pc_count must be positive, got {pc_count}")
    if pc_index < 0 or pc_index + pc_count > order.numel():
        raise ValueError(
            f"PC range [{pc_index}, {pc_index + pc_count}) is out of range for "
            f"{order.numel()} residual dimensions"
        )
    chosen = order[pc_index: pc_index + pc_count]
    pcs = evecs[:, chosen].T.to(dtype=X.dtype).reshape(pc_count, *X.shape)
    pcs = pcs / pcs.reshape(pc_count, -1).norm(dim=1).clamp(min=1e-8).view(pc_count, 1, 1)
    variances = [float(evals[idx].item()) for idx in chosen]
    return pcs, variances


def precompute_whitened_residuals(maps, X, displacement_center,
                                  whitening_matrix, batch_size=4096):
    """Cache W(T - Id - center) once for a fixed dataset."""
    center_flat = (X + displacement_center).reshape(1, -1)
    Wt = whitening_matrix.T
    out = torch.empty_like(maps)
    for start in range(0, maps.shape[0], batch_size):
        batch = maps[start:start + batch_size]
        residual_flat = batch.reshape(batch.shape[0], -1) - center_flat
        whitened = residual_flat @ Wt
        out[start:start + batch_size] = whitened.reshape_as(batch)
    return out


def unpack_transport_batch(batch, device):
    """Return raw maps plus optional cached whitened residuals."""
    if isinstance(batch, (tuple, list)):
        T_batch = batch[0].to(device).float()
        aux_batch = batch[1].to(device).float() if len(batch) > 1 else None
        return T_batch, aux_batch
    return batch.to(device).float(), None


def train_one_model(model, train_loader, test_loader, config, device):
    c = config["sparsity_coeff"]
    n = model.n
    grad_clip_norm = config.get("grad_clip_norm")
    sparsity_warmup = int(config.get("sparsity_warmup", 0) or 0)

    if config["optimizer"] == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                weight_decay=config["weight_decay"])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=config["lr"])

    scheduler = None
    if config.get("scheduler", "none") == "cosine":
        eta_min = config.get("lr_min") or 0.0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=config["epochs"], eta_min=eta_min,
        )

    if grad_clip_norm is not None and grad_clip_norm > 0:
        print(f"  Gradient clipping: max_norm={grad_clip_norm:g}", flush=True)

    tau_start = config.get("softtopk_tau_start")
    tau_end = config.get("softtopk_tau_end")
    use_tau_anneal = tau_start is not None and tau_end is not None
    if use_tau_anneal:
        tau_start = float(tau_start)
        tau_end = float(tau_end)
        tau_anneal_epochs = config.get("softtopk_tau_anneal_epochs")
        if tau_anneal_epochs is None:
            tau_anneal_epochs = config["epochs"]
        tau_anneal_epochs = max(1, int(tau_anneal_epochs))
        print(f"  SoftTopK tau anneal: {tau_start:g} -> {tau_end:g} "
              f"over {tau_anneal_epochs} epoch(s)", flush=True)

    best_epoch = 0
    best_epoch_loss = float("inf")
    best_state = clone_model_state(model)

    model.train()
    for epoch in range(config["epochs"]):
        cur_eps = model.atoms_module.eps
        cur_tau = None
        if use_tau_anneal:
            denom = max(tau_anneal_epochs - 1, 1)
            frac = min(epoch, tau_anneal_epochs - 1) / denom
            cur_tau = tau_start + frac * (tau_end - tau_start)
            model.encoder.softtopk_tau = cur_tau
        epoch_loss = 0.0
        epoch_steps = 0

        for T_batch in train_loader:
            T_batch, aux_batch = unpack_transport_batch(T_batch, device)
            T_hat, lam = (
                model(T_batch, aux_batch)
                if aux_batch is not None else model(T_batch)
            )

            diff_sq = ((T_batch - T_hat) ** 2).sum(dim=(1, 2))
            recon = 0.5 * diff_sq.mean() / n
            if sparsity_warmup > 0 and epoch < sparsity_warmup:
                c_eff = c * (epoch + 1) / sparsity_warmup
            else:
                c_eff = c
            sparsity = c_eff * lam.abs().sum(dim=1).mean()
            loss = recon + sparsity

            opt.zero_grad()
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip_norm,
                )
            opt.step()

            epoch_loss += loss.item()
            epoch_steps += 1

        if scheduler is not None:
            scheduler.step()

        avg_loss = epoch_loss / epoch_steps
        if epoch == 0 or avg_loss < best_epoch_loss:
            best_epoch_loss = avg_loss
            best_epoch = epoch + 1
            best_state = clone_model_state(model)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr_now = opt.param_groups[0]["lr"]
            tag = config.get("name", "")
            prefix = f"[{tag}] " if tag else "  "
            print(f"{prefix}Epoch {epoch+1:4d}/{config['epochs']}  loss={avg_loss:.6f}  "
                  f"best={best_epoch_loss:.6f}@{best_epoch}  "
                  f"lr={lr_now:.2e}  eps={cur_eps:.4g}  c={c_eff:.4g}"
                  + (f"  tau={cur_tau:.4g}" if cur_tau is not None else ""),
                  flush=True)

    final_state = clone_model_state(model)
    train_metrics = compute_losses(model, train_loader, c, device)
    test_metrics = compute_losses(model, test_loader, c, device)
    final_metrics = {
        "train_recon_loss": train_metrics["recon_loss"],
        "train_total_loss": train_metrics["total_loss"],
        "train_mean_l1": train_metrics["mean_l1"],
        "train_mean_active": train_metrics["mean_active"],
        "test_recon_loss": test_metrics["recon_loss"],
        "test_total_loss": test_metrics["total_loss"],
        "test_mean_l1": test_metrics["mean_l1"],
        "test_mean_active": test_metrics["mean_active"],
    }

    model.load_state_dict(best_state)
    best_train_metrics = compute_losses(model, train_loader, c, device)
    best_test_metrics = compute_losses(model, test_loader, c, device)
    best_metrics = {
        "train_recon_loss": best_train_metrics["recon_loss"],
        "train_total_loss": best_train_metrics["total_loss"],
        "train_mean_l1": best_train_metrics["mean_l1"],
        "train_mean_active": best_train_metrics["mean_active"],
        "test_recon_loss": best_test_metrics["recon_loss"],
        "test_total_loss": best_test_metrics["total_loss"],
        "test_mean_l1": best_test_metrics["mean_l1"],
        "test_mean_active": best_test_metrics["mean_active"],
    }
    model.load_state_dict(final_state)

    return {
        "final_state": final_state,
        "final_metrics": final_metrics,
        "best_state": best_state,
        "best_metrics": best_metrics,
        "best_epoch": best_epoch,
        "best_epoch_loss": best_epoch_loss,
    }


# ============================================================
# Experiment runner
# ============================================================

def run_all_experiments(config=None):
    if config is None:
        config = dict(DEFAULT_CONFIG)

    devices = make_gpu_list(config["device"], config.get("gpu_ids"))
    print(f"Using device(s): {[str(d) for d in devices]}")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    X, maps = load_transport_maps(config["data_dir"], classes=config["classes"])
    X = X.float()
    maps = maps.float()
    dim = X.shape[1]
    displacement_center = None
    displacement_pc_component = None
    displacement_pc_variance = None
    whitening_matrix = None
    unwhitening_matrix = None

    # Build atom grid Y
    grid_mode = config.get("grid_mode", "cloud_mixture")
    if grid_mode == "cloud_mixture":
        grid_points = sample_grid_from_cloud_mixture(
            config["data_dir"],
            n_clouds=config.get("grid_n_clouds", 10),
            support_size=config.get("grid_support_size", 3000),
            classes=config["classes"],
            seed=config["seed"],
        )
    elif grid_mode == "uniform":
        grid_points = make_grid_nd(config["grid_side"], dim=dim)
        print(f"Atom grid: uniform {config['grid_side']}^{dim} = {grid_points.shape[0]} pts")
    else:
        raise ValueError(f"Unknown grid_mode: {grid_mode}")

    train_loader, test_loader = make_dataloaders(
        maps, batch_size=config["batch_size"],
        test_fraction=config["test_fraction"], seed=config["seed"],
    )
    raw_train_loader, raw_test_loader = train_loader, test_loader
    whitened_train_loader = None
    whitened_test_loader = None

    all_results = []
    experiments = list(product(config["methods"], config["epsilons"],
                                config["sparsity_coeffs"]))
    print(f"\n{len(experiments)} total runs "
          f"({len(config['methods'])} methods x {len(config['epsilons'])} epsilons "
          f"x {len(config['sparsity_coeffs'])} sparsity coeffs)")

    for run_idx, (method, eps, c) in enumerate(experiments):
        device = devices[run_idx % len(devices)]
        run_name = f"{method}_eps{eps}_c{c}"
        print(f"\n{'='*60}")
        print(f"Run {run_idx+1}/{len(experiments)}: method={method}, eps={eps}, "
              f"c={c}  [device={device}]")
        print(f"{'='*60}")

        if method == "raw_map":
            model_cls = TransportMapSAE
        elif method == "displacement":
            model_cls = DisplacementFieldSAE
        elif method == "displacement_centered":
            model_cls = CenteredDisplacementFieldSAE
            if displacement_center is None:
                displacement_center = load_displacement_center(
                    config["data_dir"], X, maps,
                    mode=config.get("displacement_center_mode", "generator_mean"),
                ).float()
                center_norm = float(displacement_center.pow(2).sum().sqrt())
                print(
                    f"  Displacement center: mode="
                    f"{config.get('displacement_center_mode', 'generator_mean')} "
                    f"norm={center_norm:.4f}",
                    flush=True,
                )
        elif method == "displacement_centered_pc_removed":
            model_cls = PCRemovedCenteredDisplacementFieldSAE
            if displacement_center is None:
                displacement_center = load_displacement_center(
                    config["data_dir"], X, maps,
                    mode=config.get("displacement_center_mode", "generator_mean"),
                ).float()
                center_norm = float(displacement_center.pow(2).sum().sqrt())
                print(
                    f"  Displacement center: mode="
                    f"{config.get('displacement_center_mode', 'generator_mean')} "
                    f"norm={center_norm:.4f}",
                    flush=True,
                )
            if displacement_pc_component is None:
                displacement_pc_component, displacement_pc_variance = (
                    load_displacement_pc_components(
                        config["data_dir"], X, maps, displacement_center,
                        mode=config.get("displacement_pc_mode", "dataset_pca"),
                        pc_index=config.get("displacement_pc_index", 1),
                        pc_count=config.get("displacement_pc_count", 1),
                        max_samples=config.get("whitening_max_samples"),
                        seed=config["seed"],
                    )
                )
                print(
                    f"  Removed displacement PC: mode="
                    f"{config.get('displacement_pc_mode', 'dataset_pca')} "
                    f"start={config.get('displacement_pc_index', 1)} "
                    f"count={config.get('displacement_pc_count', 1)} "
                    f"variances={displacement_pc_variance}",
                    flush=True,
                )
        elif method == "displacement_centered_whitened":
            model_cls = WhitenedCenteredDisplacementFieldSAE
            if displacement_center is None:
                displacement_center = load_displacement_center(
                    config["data_dir"], X, maps,
                    mode=config.get("displacement_center_mode", "generator_mean"),
                ).float()
                center_norm = float(displacement_center.pow(2).sum().sqrt())
                print(
                    f"  Displacement center: mode="
                    f"{config.get('displacement_center_mode', 'generator_mean')} "
                    f"norm={center_norm:.4f}",
                    flush=True,
                )
            if whitening_matrix is None or unwhitening_matrix is None:
                whitening_matrix, unwhitening_matrix = load_displacement_whitening(
                    config["data_dir"], X, maps, displacement_center,
                    mode=config.get("whitening_mode", "dataset_zca"),
                    eps=config.get("whitening_eps", 1e-4),
                    max_samples=config.get("whitening_max_samples"),
                    seed=config["seed"],
                )
                cond_proxy = float(torch.linalg.cond(unwhitening_matrix.double()))
                print(
                    f"  Whitening: mode={config.get('whitening_mode', 'dataset_zca')} "
                    f"eps={config.get('whitening_eps', 1e-4):g} "
                    f"max_samples={config.get('whitening_max_samples')} "
                    f"unwhiten_cond~{cond_proxy:.3g}",
                    flush=True,
                )
            if whitened_train_loader is None or whitened_test_loader is None:
                print("  Precomputing whitened dataset residuals...", flush=True)
                whitened_maps = precompute_whitened_residuals(
                    maps, X, displacement_center, whitening_matrix,
                    batch_size=max(int(config["batch_size"]), 1024),
                )
                whitened_train_loader, whitened_test_loader = make_dataloaders(
                    maps, batch_size=config["batch_size"],
                    test_fraction=config["test_fraction"], seed=config["seed"],
                    aux_maps=whitened_maps,
                )
        else:
            raise ValueError(f"Unknown method: {method}")

        model_kwargs = dict(
            m=config["m"], eps=eps,
            grid_side=config["grid_side"],
            lista_steps=config["lista_steps"],
            grid_points=grid_points,
            normalize_atoms=True,
            per_atom_gain=True,
            lateral_init="damped_identity",
            activation_type=config["activation_type"],
            topk_k=config["topk_k"],
            softtopk_tau=(config["softtopk_tau_start"]
                          if config.get("softtopk_tau_start") is not None
                          else config["softtopk_tau"]),
        )
        model_seed = config.get("model_seed")
        if model_seed is not None:
            model_seed = int(model_seed)
            print(f"  Model init seed: {model_seed}", flush=True)
            torch.manual_seed(model_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(model_seed)
        if method == "displacement_centered":
            model = model_cls(X, displacement_center, **model_kwargs).to(device)
        elif method == "displacement_centered_pc_removed":
            model = model_cls(
                X, displacement_center, displacement_pc_component,
                **model_kwargs,
            ).to(device)
        elif method == "displacement_centered_whitened":
            model = model_cls(
                X, displacement_center, whitening_matrix, unwhitening_matrix,
                **model_kwargs,
            ).to(device)
        else:
            model = model_cls(X, **model_kwargs).to(device)

        init_checkpoint = config.get("init_checkpoint")
        if init_checkpoint:
            init_checkpoint = Path(init_checkpoint)
            ckpt = torch.load(init_checkpoint, map_location=device)
            state = ckpt["model_state"] if "model_state" in ckpt else ckpt
            model.load_state_dict(state)
            print(f"  Loaded init checkpoint: {init_checkpoint}", flush=True)

        run_config = dict(config, sparsity_coeff=c, name=run_name)
        if method == "displacement_centered_whitened":
            active_train_loader = whitened_train_loader
            active_test_loader = whitened_test_loader
        else:
            active_train_loader = raw_train_loader
            active_test_loader = raw_test_loader

        t0 = time.time()
        train_result = train_one_model(model, active_train_loader, active_test_loader,
                                       run_config, device)
        elapsed = time.time() - t0
        metrics = train_result["final_metrics"]
        best_metrics = train_result["best_metrics"]

        result = {
            "method": method,
            "epsilon": eps,
            "sparsity_coeff": c,
            "device": str(device),
            "m": config["m"],
            "epochs": config["epochs"],
            "lr": config["lr"],
            "grad_clip_norm": config.get("grad_clip_norm"),
            "sparsity_warmup": config.get("sparsity_warmup"),
            "model_seed": config.get("model_seed"),
            "init_checkpoint": config.get("init_checkpoint"),
            "softtopk_tau_start": config.get("softtopk_tau_start"),
            "softtopk_tau_end": config.get("softtopk_tau_end"),
            "softtopk_tau_anneal_epochs": config.get("softtopk_tau_anneal_epochs"),
            "displacement_center_mode": config.get("displacement_center_mode"),
            "displacement_pc_mode": config.get("displacement_pc_mode"),
            "displacement_pc_index": config.get("displacement_pc_index"),
            "displacement_pc_count": config.get("displacement_pc_count"),
            "displacement_pc_variance": displacement_pc_variance,
            "whitening_mode": config.get("whitening_mode"),
            "whitening_eps": config.get("whitening_eps"),
            "whitening_max_samples": config.get("whitening_max_samples"),
            "elapsed_seconds": round(elapsed, 1),
            "best_epoch": train_result["best_epoch"],
            "best_epoch_loss": train_result["best_epoch_loss"],
            "best_train_recon_loss": best_metrics["train_recon_loss"],
            "best_train_total_loss": best_metrics["train_total_loss"],
            "best_train_mean_l1": best_metrics["train_mean_l1"],
            "best_train_mean_active": best_metrics["train_mean_active"],
            "best_test_recon_loss": best_metrics["test_recon_loss"],
            "best_test_total_loss": best_metrics["test_total_loss"],
            "best_test_mean_l1": best_metrics["test_mean_l1"],
            "best_test_mean_active": best_metrics["test_mean_active"],
            **metrics,
        }
        all_results.append(result)

        print(f"\n  Results for {run_name}:")
        print(f"    Last train recon={metrics['train_recon_loss']:.6f}  "
              f"L1={metrics['train_mean_l1']:.4f}  "
              f"active={metrics['train_mean_active']:.2f}")
        print(f"    Last test  recon={metrics['test_recon_loss']:.6f}  "
              f"L1={metrics['test_mean_l1']:.4f}  "
              f"active={metrics['test_mean_active']:.2f}")
        print(f"    Best epoch={train_result['best_epoch']}  "
              f"epoch_loss={train_result['best_epoch_loss']:.6f}  "
              f"test_recon={best_metrics['test_recon_loss']:.6f}")

        ckpt_path = output_dir / f"{run_name}.pt"
        best_ckpt_path = output_dir / f"{run_name}_best.pt"

        common_ckpt = {
            "method": method,
            "eps": eps,
            "sparsity_coeff": c,
            "m": config["m"],
            "lista_steps": config["lista_steps"],
            "grid_points": grid_points,
            "X": X,
        }
        if method in (
            "displacement_centered",
            "displacement_centered_pc_removed",
            "displacement_centered_whitened",
        ):
            common_ckpt["displacement_center"] = displacement_center
            common_ckpt["displacement_center_mode"] = config.get(
                "displacement_center_mode", "generator_mean",
            )
        if method == "displacement_centered_pc_removed":
            common_ckpt["displacement_pc_component"] = displacement_pc_component
            common_ckpt["displacement_pc_mode"] = config.get(
                "displacement_pc_mode", "dataset_pca",
            )
            common_ckpt["displacement_pc_index"] = config.get("displacement_pc_index", 1)
            common_ckpt["displacement_pc_count"] = config.get("displacement_pc_count", 1)
            common_ckpt["displacement_pc_variance"] = displacement_pc_variance
        if method == "displacement_centered_whitened":
            common_ckpt["whitening_matrix"] = whitening_matrix
            common_ckpt["unwhitening_matrix"] = unwhitening_matrix
            common_ckpt["whitening_mode"] = config.get("whitening_mode", "dataset_zca")
            common_ckpt["whitening_eps"] = config.get("whitening_eps", 1e-4)
            common_ckpt["whitening_max_samples"] = config.get("whitening_max_samples")
        torch.save({
            **common_ckpt,
            "model_state": train_result["final_state"],
            "checkpoint_role": "last",
            "epoch": config["epochs"],
            "metrics": metrics,
        }, ckpt_path)
        torch.save({
            **common_ckpt,
            "model_state": train_result["best_state"],
            "checkpoint_role": "best",
            "epoch": train_result["best_epoch"],
            "epoch_loss": train_result["best_epoch_loss"],
            "metrics": best_metrics,
        }, best_ckpt_path)
        print(f"    Saved last checkpoint: {ckpt_path}")
        print(f"    Saved best checkpoint: {best_ckpt_path}")

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll metrics saved to {metrics_path}")

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    return all_results


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="3D point-cloud transport-map SAE experiments")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_CONFIG["data_dir"])
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--m", type=int, default=DEFAULT_CONFIG["m"])
    parser.add_argument("--lista_steps", type=int, default=DEFAULT_CONFIG["lista_steps"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--sparsity_coeffs", type=float, nargs="+",
                        default=DEFAULT_CONFIG["sparsity_coeffs"])
    parser.add_argument("--epsilons", type=float, nargs="+",
                        default=DEFAULT_CONFIG["epsilons"])
    parser.add_argument("--methods", type=str, nargs="+",
                        default=DEFAULT_CONFIG["methods"],
                        choices=["raw_map", "displacement", "displacement_centered",
                                 "displacement_centered_pc_removed",
                                 "displacement_centered_whitened"])
    parser.add_argument("--displacement_center_mode", type=str,
                        default=DEFAULT_CONFIG["displacement_center_mode"],
                        choices=["generator_mean", "dataset_mean", "none"],
                        help="Fixed residual baseline for displacement_centered runs.")
    parser.add_argument("--displacement_pc_index", type=int,
                        default=DEFAULT_CONFIG["displacement_pc_index"],
                        help="Zero-based PCA component index to remove after centering.")
    parser.add_argument("--displacement_pc_count", type=int,
                        default=DEFAULT_CONFIG["displacement_pc_count"],
                        help="Number of consecutive PCA components to remove.")
    parser.add_argument("--displacement_pc_mode", type=str,
                        default=DEFAULT_CONFIG["displacement_pc_mode"],
                        choices=["dataset_pca", "generator_pca"],
                        help="Residual source used to estimate removed PC.")
    parser.add_argument("--whitening_mode", type=str,
                        default=DEFAULT_CONFIG["whitening_mode"],
                        choices=["dataset_zca", "generator_zca", "none"],
                        help="Whitening source for displacement_centered_whitened runs.")
    parser.add_argument("--whitening_eps", type=float,
                        default=DEFAULT_CONFIG["whitening_eps"],
                        help="Diagonal regularizer added before inverse sqrt whitening.")
    parser.add_argument("--whitening_max_samples", type=int,
                        default=DEFAULT_CONFIG["whitening_max_samples"],
                        help="Optional cap on residual samples used to estimate whitening.")
    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG["device"],
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--gpu_ids", type=int, nargs="*", default=None)
    parser.add_argument("--grid_mode", type=str, default=DEFAULT_CONFIG["grid_mode"],
                        choices=["uniform", "cloud_mixture"])
    parser.add_argument("--grid_side", type=int, default=DEFAULT_CONFIG["grid_side"])
    parser.add_argument("--grid_n_clouds", type=int, default=DEFAULT_CONFIG["grid_n_clouds"])
    parser.add_argument("--grid_support_size", type=int,
                        default=DEFAULT_CONFIG["grid_support_size"])
    parser.add_argument("--scheduler", type=str, default=DEFAULT_CONFIG["scheduler"],
                        choices=["none", "cosine"])
    parser.add_argument("--lr_min", type=float, default=DEFAULT_CONFIG["lr_min"])
    parser.add_argument("--grad_clip_norm", type=float,
                        default=DEFAULT_CONFIG["grad_clip_norm"],
                        help="If >0, clip gradient norm to this value after backward.")
    parser.add_argument("--sparsity_warmup", type=int,
                        default=DEFAULT_CONFIG["sparsity_warmup"],
                        help="Linearly ramp L1 coefficient over this many epochs.")
    parser.add_argument("--init_checkpoint", type=str,
                        default=DEFAULT_CONFIG["init_checkpoint"],
                        help="Optional checkpoint whose model_state initializes each run.")
    parser.add_argument("--model_seed", type=int,
                        default=DEFAULT_CONFIG["model_seed"],
                        help="Optional torch seed applied immediately before model init.")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--classes", type=str, nargs="*", default=None)
    parser.add_argument("--test_fraction", type=float,
                        default=DEFAULT_CONFIG["test_fraction"])
    parser.add_argument("--weight_decay", type=float,
                        default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--optimizer", type=str, default=DEFAULT_CONFIG["optimizer"],
                        choices=["adamw", "adam"])
    parser.add_argument("--activation_type", type=str,
                        default=DEFAULT_CONFIG["activation_type"],
                        choices=["relu", "jumprelu", "final_jumprelu", "topk",
                                 "topk_simplex", "softtopk", "final_softtopk",
                                 "final_softtopk_simplex"])
    parser.add_argument("--topk_k", type=int, default=DEFAULT_CONFIG["topk_k"])
    parser.add_argument("--softtopk_tau", type=float,
                        default=DEFAULT_CONFIG["softtopk_tau"])
    parser.add_argument("--softtopk_tau_start", type=float,
                        default=DEFAULT_CONFIG["softtopk_tau_start"])
    parser.add_argument("--softtopk_tau_end", type=float,
                        default=DEFAULT_CONFIG["softtopk_tau_end"])
    parser.add_argument("--softtopk_tau_anneal_epochs", type=int,
                        default=DEFAULT_CONFIG["softtopk_tau_anneal_epochs"])

    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config.update(vars(args))
    run_all_experiments(config)
