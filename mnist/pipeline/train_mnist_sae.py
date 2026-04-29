"""
Training and experiment runner for displacement-field sparse autoencoders.

Trains a DisplacementFieldSAE model for each (epsilon, sparsity_coeff) pair.

Uses data produced by prepare_mnist_ot.py.

Device handling:
  --device auto   (default) uses CUDA when available, otherwise CPU.
  --device cuda   uses CUDA; falls back to CPU if unavailable.
  --device mps    uses Apple Silicon MPS; falls back to CPU if unavailable.
  --device cpu    forces CPU.

  On a multi-GPU machine, set --gpu_ids to select GPUs. When multiple GPUs
  are specified, the 10 experiment runs are distributed across them in a
  round-robin fashion (one model per GPU -- the models are small enough
  that data-parallel sharding within a single run is unnecessary).

Usage:
    # MacBook/default CPU fallback:
    python mnist/pipeline/train_mnist_sae.py --data_dir datasets/mnist_ot

    # Multi-GPU server, use GPUs 0-3:
    python mnist/pipeline/train_mnist_sae.py --data_dir datasets/mnist_ot --device cuda --gpu_ids 0 1 2 3

How the code uses prepare_mnist_ot.py:
    That script produces base_measure.pt and digit_*/mappings.pt files.
    We load them via mnist_ot_data.load_transport_maps(), which concatenates all
    digits and returns X (n,2) and maps (N,n,2).  To swap in entropic maps
    later, just replace the contents of those .pt files -- nothing else changes.
"""

import json
import math
import time
from pathlib import Path
from itertools import product
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

from mnist_ot_data import load_transport_maps, make_dataloaders
from mnist_sae_models import DisplacementFieldSAE


# ============================================================
# Device helpers
# ============================================================

def resolve_device(device_str):
    """
    Resolve a device string to a torch.device.

    "auto" -> cuda if available, else cpu
    "cuda" -> cuda if available, else cpu
    "mps"  -> mps if available, else cpu
    "cpu"  -> cpu
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    elif device_str == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    elif device_str == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("MPS not available, falling back to CPU")
        return torch.device("cpu")
    else:
        return torch.device(device_str)


def make_gpu_list(device_str, gpu_ids):
    """
    Build a list of devices for round-robin assignment across experiments.

    If gpu_ids is provided and device is cuda, returns one torch.device per GPU.
    Otherwise returns a single-element list with the resolved device.
    """
    if gpu_ids and device_str in ("cuda", "auto") and torch.cuda.is_available():
        return [torch.device(f"cuda:{gid}") for gid in gpu_ids]
    return [resolve_device(device_str)]


# ============================================================
# Grid generation from data mixture
# ============================================================

def sample_grid_from_data_mixture(n_images=500, support_size=1024, seed=42):
    """
    Build a set of grid points by sampling from a mixture of MNIST digit images.

    1. Download MNIST (cached after first call).
    2. Pick `n_images` random images (blind, all classes).
    3. Treat each image as a probability measure on [0,1]^2.
    4. Form an equal-weight mixture of these measures.
    5. Sample `support_size` points from the mixture.

    Returns:
        grid_points: torch.Tensor of shape (support_size, 2)
    """
    mnist = datasets.MNIST(root='./mnist_raw', train=True, download=True)
    rng = np.random.RandomState(seed)

    chosen = rng.choice(len(mnist), size=n_images, replace=False)

    # Build per-image coordinate grids and probabilities once (all 28x28)
    h, w = 28, 28
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = np.stack([cols.ravel() / w, rows.ravel() / h], axis=1)  # (784, 2), x then y

    # Collect unnormalized weights across all chosen images
    # We'll concatenate (coords, weight) per image, then sample globally.
    all_coords = np.tile(coords, (n_images, 1))  # (n_images*784, 2)
    all_weights = np.empty(n_images * 784, dtype='float64')

    for j, idx in enumerate(chosen):
        img = np.array(mnist[idx][0], dtype='float64')  # (28,28)
        img = np.maximum(img, 0)
        total = img.sum()
        if total == 0:
            img = np.ones_like(img)
            total = img.sum()
        # Each image has equal mixture weight (1/n_images), and within-image
        # weights are pixel intensities normalized to sum to 1.
        all_weights[j * 784:(j + 1) * 784] = img.ravel() / total / n_images

    # Normalize to a proper distribution (should already sum to ~1, but be safe)
    all_weights /= all_weights.sum()

    # Sample support_size points from the mixture
    indices = rng.choice(len(all_weights), size=support_size, replace=True, p=all_weights)
    grid_points = all_coords[indices].astype('float32')

    print(f"Grid: sampled {support_size} points from mixture of {n_images} MNIST images")
    return torch.tensor(grid_points)


# ============================================================
# Config -- all defaults collected here for easy modification
# ============================================================

DEFAULT_CONFIG = dict(
    # Data
    data_dir="datasets/mnist_ot",
    digits=None,          # None = all digits
    test_fraction=0.1,
    seed=42,

    # Model
    m=30,                 # number of dictionary atoms
    grid_side=64,         # target grid side (K = grid_side^2 = 1024)
    grid_mode="uniform",  # "uniform" or "data_mixture"
    grid_n_images=500,    # number of MNIST images for data_mixture grid
    grid_support_size=None,  # points in data_mixture grid; None = grid_side^2
    lista_steps=20,        # number of LISTA iterations (1 = single-step encoder)

    # Training
    batch_size=256,
    epochs=3000,
    lr=1e-3,
    optimizer="adamw",      # "adam" or "adamw"
    weight_decay=0.1,
    scheduler="none",       # "none" or "cosine"
    lr_min=0.0,             # minimum lr for cosine scheduler

    # Experiment grid
    epsilons=[0.025],
    sparsity_coeffs=[0.0001],

    # Output
    output_dir="2D_WDL_results",

    # Device: "auto" | "cuda" | "mps" | "cpu"
    device="auto",
    gpu_ids=None,         # e.g. [0,1,2,3] for multi-GPU round-robin
)


# ============================================================
# Loss computation
# ============================================================

def compute_losses(model, loader, sparsity_coeff, device):
    """
    Evaluate model on a full DataLoader.

    Returns dict with:
        recon_loss: (1/2) ||T - T_hat||^2_{L^2(rho)}, averaged over dataset
        total_loss: recon_loss + c * mean |lambda|_1
        mean_l1:    average L1 norm of codes (sum_j |lambda_j|) per sample
        mean_active: average number of active (nonzero) atoms per sample
    """
    model.eval()
    n = model.n

    total_recon = 0.0
    total_l1 = 0.0
    total_active = 0.0
    total_samples = 0

    with torch.no_grad():
        for T_batch in loader:
            T_batch = T_batch.to(device).float()  # (B, n, 2)
            B = T_batch.shape[0]

            T_hat, lam = model(T_batch)  # (B, n, 2), (B, m)

            # L^2(rho) reconstruction: (1/2n) sum_l ||T - T_hat||^2, per sample
            diff_sq = ((T_batch - T_hat) ** 2).sum(dim=(1, 2))  # (B,)
            recon_per_sample = 0.5 * diff_sq / n                # (B,)

            l1_per_sample = lam.abs().sum(dim=1)                # (B,)
            active_per_sample = (lam > 0).float().sum(dim=1)    # (B,)

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


# ============================================================
# Training loop
# ============================================================

def _anneal_eps(epoch, epochs, eps_start, eps_end, fraction):
    """
    Geometric (log-linear) anneal of eps from eps_start to eps_end over the
    first `fraction` of training; constant eps_end thereafter.
    """
    anneal_epochs = max(int(epochs * fraction), 1)
    progress = min(epoch / anneal_epochs, 1.0)
    log_eps = (1.0 - progress) * math.log(eps_start) + progress * math.log(eps_end)
    return math.exp(log_eps)


def train_one_model(model, train_loader, test_loader, config, device):
    """
    Train a single model. Returns dict of final metrics.

    If config["eps_anneal"] is True, the atoms module's eps is updated at the
    start of each epoch via _anneal_eps(eps_start -> eps_end) over the first
    eps_anneal_fraction of training.
    """
    c = config["sparsity_coeff"]
    n = model.n

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

    anneal = bool(config.get("eps_anneal", False))
    eps_start = config.get("eps_start", None)
    eps_end = config.get("eps_end", None)
    anneal_fraction = config.get("eps_anneal_fraction", 0.5)
    sparsity_warmup = config.get("sparsity_warmup", 0)

    model.train()
    for epoch in range(config["epochs"]):
        # Epsilon annealing
        if anneal and eps_start is not None and eps_end is not None:
            cur_eps = _anneal_eps(epoch, config["epochs"], eps_start, eps_end,
                                  anneal_fraction)
            model.atoms_module.set_eps(cur_eps)
        else:
            cur_eps = model.atoms_module.eps

        # Optional sparsity warmup: c ramps linearly from 0 to config c over
        # `sparsity_warmup` epochs.  Useful for simplex / topk runs where early
        # training needs to find useful atoms before the sparsity term kicks in.
        if sparsity_warmup > 0 and epoch < sparsity_warmup:
            c_eff = c * (epoch + 1) / sparsity_warmup
        else:
            c_eff = c

        epoch_loss = 0.0
        epoch_steps = 0

        for T_batch in train_loader:
            T_batch = T_batch.to(device).float()
            B = T_batch.shape[0]

            T_hat, lam = model(T_batch)

            # (1/2) ||T - T_hat||^2_{L^2(rho)} averaged over batch
            diff_sq = ((T_batch - T_hat) ** 2).sum(dim=(1, 2))  # (B,)
            recon = 0.5 * diff_sq.mean() / n

            # Sparsity: c * mean |lambda|_1  (skipped if c_eff == 0)
            sparsity = c_eff * lam.abs().sum(dim=1).mean()

            loss = recon + sparsity

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            epoch_steps += 1

        if scheduler is not None:
            scheduler.step()

        avg_loss = epoch_loss / epoch_steps
        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr_now = opt.param_groups[0]['lr']
            tag = config.get("name", "")
            prefix = f"[{tag}] " if tag else "  "
            print(f"{prefix}Epoch {epoch+1:4d}/{config['epochs']}  loss={avg_loss:.6f}  "
                  f"lr={lr_now:.2e}  eps={cur_eps:.4g}  c={c_eff:.4g}", flush=True)

    # Final evaluation
    train_metrics = compute_losses(model, train_loader, c, device)
    test_metrics = compute_losses(model, test_loader, c, device)

    return {
        "train_recon_loss": train_metrics["recon_loss"],
        "train_total_loss": train_metrics["total_loss"],
        "train_mean_l1": train_metrics["mean_l1"],
        "train_mean_active": train_metrics["mean_active"],
        "test_recon_loss": test_metrics["recon_loss"],
        "test_total_loss": test_metrics["total_loss"],
        "test_mean_l1": test_metrics["mean_l1"],
        "test_mean_active": test_metrics["mean_active"],
    }


# ============================================================
# Experiment runner
# ============================================================

def run_all_experiments(config=None):
    """
    Train a DisplacementFieldSAE for every (epsilon, sparsity_coeff) pair.
    Saves metrics to JSON and model checkpoints to output_dir.

    Multi-GPU: when gpu_ids is set (e.g. [0,1,2,3]), experiments are assigned
    to GPUs round-robin. Each model is small enough that one GPU per run is
    the right granularity -- no need for DataParallel within a single run.
    """
    if config is None:
        config = dict(DEFAULT_CONFIG)

    devices = make_gpu_list(config["device"], config.get("gpu_ids"))
    print(f"Using device(s): {[str(d) for d in devices]}")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data once
    print("Loading data...")
    X, maps = load_transport_maps(config["data_dir"], digits=config["digits"])
    X = X.float()
    maps = maps.float()

    # Build grid points (None = use uniform grid inside the model)
    grid_points = None
    if config.get("grid_mode", "uniform") == "data_mixture":
        gs = config.get("grid_support_size") or config["grid_side"] ** 2
        grid_points = sample_grid_from_data_mixture(
            n_images=config.get("grid_n_images", 500),
            support_size=gs,
            seed=config["seed"],
        )

    train_loader, test_loader = make_dataloaders(
        maps,
        batch_size=config["batch_size"],
        test_fraction=config["test_fraction"],
        seed=config["seed"],
    )

    all_results = []
    experiments = list(product(config["epsilons"], config["sparsity_coeffs"]))
    print(f"\n{len(experiments)} total runs "
          f"({len(config['epsilons'])} epsilons x {len(config['sparsity_coeffs'])} sparsity coeffs)")

    for run_idx, (eps, c) in enumerate(experiments):
        # Round-robin GPU assignment
        device = devices[run_idx % len(devices)]

        run_name = f"displacement_eps{eps}_c{c}"
        print(f"\n{'='*60}")
        print(f"Run {run_idx+1}/{len(experiments)}: eps={eps}, c={c}  [device={device}]")
        print(f"{'='*60}")

        model = DisplacementFieldSAE(
            X, m=config["m"], eps=eps, grid_side=config["grid_side"],
            lista_steps=config["lista_steps"],
            grid_points=grid_points,
            normalize_atoms=True,
            per_atom_gain=True,
            lateral_init="damped_identity",
        )

        model = model.to(device)

        # Override sparsity_coeff for this run
        run_config = dict(config, sparsity_coeff=c)

        t0 = time.time()
        metrics = train_one_model(model, train_loader, test_loader, run_config, device)
        elapsed = time.time() - t0

        # Record results
        result = {
            "method": "displacement",
            "epsilon": eps,
            "sparsity_coeff": c,
            "device": str(device),
            "m": config["m"],
            "epochs": config["epochs"],
            "lr": config["lr"],
            "elapsed_seconds": round(elapsed, 1),
            **metrics,
        }
        all_results.append(result)

        # Print summary
        print(f"\n  Results for {run_name}:")
        print(f"    Train recon={metrics['train_recon_loss']:.6f}  "
              f"total={metrics['train_total_loss']:.6f}  "
              f"L1={metrics['train_mean_l1']:.4f}  "
              f"active={metrics['train_mean_active']:.2f}")
        print(f"    Test  recon={metrics['test_recon_loss']:.6f}  "
              f"total={metrics['test_total_loss']:.6f}  "
              f"L1={metrics['test_mean_l1']:.4f}  "
              f"active={metrics['test_mean_active']:.2f}")

        # Save model checkpoint:  results/<run_name>.pt
        ckpt_path = output_dir / f"{run_name}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"    Saved checkpoint: {ckpt_path}")

    # Save all metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll metrics saved to {metrics_path}")

    # Also save config for reproducibility
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    return all_results


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transport map SAE experiments")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_CONFIG["data_dir"])
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--m", type=int, default=DEFAULT_CONFIG["m"],
                        help="Number of dictionary atoms")
    parser.add_argument("--lista_steps", type=int, default=DEFAULT_CONFIG["lista_steps"],
                        help="Number of LISTA iterations (1 = single-step encoder)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--sparsity_coeffs", type=float, nargs="+",
                        default=DEFAULT_CONFIG["sparsity_coeffs"],
                        help="Sparsity coefficients to sweep (e.g. --sparsity_coeffs 0.01 0.1)")
    parser.add_argument("--epsilons", type=float, nargs="+",
                        default=DEFAULT_CONFIG["epsilons"],
                        help="Gibbs epsilons to sweep (e.g. --epsilons 0.1 0.05 0.01)")
    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG["device"],
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Device: auto uses CUDA if available, otherwise CPU")
    parser.add_argument("--gpu_ids", type=int, nargs="*", default=None,
                        help="GPU IDs for multi-GPU round-robin (e.g. --gpu_ids 0 1 2 3)")
    parser.add_argument("--grid_mode", type=str, default=DEFAULT_CONFIG["grid_mode"],
                        choices=["uniform", "data_mixture"],
                        help="'uniform': regular grid; 'data_mixture': sample from MNIST images")
    parser.add_argument("--grid_n_images", type=int, default=DEFAULT_CONFIG["grid_n_images"],
                        help="Number of MNIST images for data_mixture grid")
    parser.add_argument("--grid_support_size", type=int, default=DEFAULT_CONFIG["grid_support_size"],
                        help="Number of points in data_mixture grid (default: grid_side^2)")
    parser.add_argument("--scheduler", type=str, default=DEFAULT_CONFIG["scheduler"],
                        choices=["none", "cosine"],
                        help="LR scheduler: 'none' or 'cosine'")
    parser.add_argument("--lr_min", type=float, default=DEFAULT_CONFIG["lr_min"],
                        help="Minimum LR for cosine scheduler")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])

    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config.update(vars(args))

    run_all_experiments(config)
