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
    DisplacementFieldSAE,
    TransportMapSAE,
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

    # Model / atoms
    m=30,
    grid_side=10,            # only used if grid_mode="uniform" (10^3 = 1000 pts)
    grid_mode="cloud_mixture",   # "uniform" or "cloud_mixture"
    grid_n_clouds=10,
    grid_support_size=3000,
    lista_steps=20,
    activation_type="relu",  # "relu" | "jumprelu" | "topk" | "topk_simplex"
    topk_k=3,                # k for topk / topk_simplex

    # Training
    batch_size=64,
    epochs=2000,
    lr=1e-3,
    optimizer="adamw",
    weight_decay=0.1,
    scheduler="none",
    lr_min=0.0,
    grad_clip_norm=None,

    # Sweep
    epsilons=[0.025],
    sparsity_coeffs=[1e-4],
    methods=["displacement"],

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
            T_batch = T_batch.to(device).float()
            B = T_batch.shape[0]
            T_hat, lam = model(T_batch)

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


def train_one_model(model, train_loader, test_loader, config, device):
    c = config["sparsity_coeff"]
    n = model.n
    grad_clip_norm = config.get("grad_clip_norm")

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

    best_epoch = 0
    best_epoch_loss = float("inf")
    best_state = clone_model_state(model)

    model.train()
    for epoch in range(config["epochs"]):
        cur_eps = model.atoms_module.eps
        epoch_loss = 0.0
        epoch_steps = 0

        for T_batch in train_loader:
            T_batch = T_batch.to(device).float()
            T_hat, lam = model(T_batch)

            diff_sq = ((T_batch - T_hat) ** 2).sum(dim=(1, 2))
            recon = 0.5 * diff_sq.mean() / n
            sparsity = c * lam.abs().sum(dim=1).mean()
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
                  f"lr={lr_now:.2e}  eps={cur_eps:.4g}  c={c:.4g}", flush=True)

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
        else:
            raise ValueError(f"Unknown method: {method}")

        model = model_cls(
            X, m=config["m"], eps=eps,
            grid_side=config["grid_side"],
            lista_steps=config["lista_steps"],
            grid_points=grid_points,
            normalize_atoms=True,
            per_atom_gain=True,
            lateral_init="damped_identity",
            activation_type=config["activation_type"],
            topk_k=config["topk_k"],
        ).to(device)

        run_config = dict(config, sparsity_coeff=c, name=run_name)

        t0 = time.time()
        train_result = train_one_model(model, train_loader, test_loader,
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
                        choices=["raw_map", "displacement"])
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
                        choices=["relu", "jumprelu", "topk", "topk_simplex"])
    parser.add_argument("--topk_k", type=int, default=DEFAULT_CONFIG["topk_k"])

    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config.update(vars(args))
    run_all_experiments(config)
