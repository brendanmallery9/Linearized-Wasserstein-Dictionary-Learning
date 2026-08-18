"""
Evaluate an already-trained LWDL-EOT (transport-map SAE) model on a fixed
train/test split, producing LWDL codes and reconstructions for the minimal
point-cloud ablation.

This does NOT train anything and does NOT touch the source result directory.
It rehydrates the model from a checkpoint (which stores X, grid_points, m, eps,
lista_steps, method, and the displacement center for centered runs), re-encodes
the maps of both splits into LWDL coefficients, reconstructs the maps, and (as a
CLI) writes a metrics row compatible with the PCA / sparse-coding rows produced
by run_pointcloud_minimal_ablation.py.

Supported methods: "displacement_centered" (the main ModelNet run),
"displacement", and "raw_map".

Usage (standalone):
    python pointcloud/pipeline/evaluate_pointcloud_lwdl_for_ablation.py \
        --data_dir datasets/modelnet10_6cls_ot \
        --results_dir pointcloud/results/modelnet10_6cls_uniform \
        --split_path pointcloud/results/minimal_ablation/split_indices.json \
        --output_dir pointcloud/results/minimal_ablation/lwdl
"""

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MNIST_PIPELINE = REPO_ROOT / "mnist" / "pipeline"
for p in (str(REPO_ROOT), str(MNIST_PIPELINE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from mnist_sae_models import DisplacementFieldSAE, TransportMapSAE
from pointcloud.pipeline.pointcloud_centered_sae import CenteredDisplacementFieldSAE
from pointcloud.pipeline.pointcloud_data import (
    load_with_labels,
    make_split_indices,
)


# ============================================================
# Device
# ============================================================

def resolve_device(device_str):
    if device_str in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    if device_str == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


# ============================================================
# Checkpoint discovery / model rehydration
# ============================================================

def find_checkpoint(results_dir, checkpoint=None):
    """Locate the LWDL checkpoint to evaluate."""
    if checkpoint:
        ckpt = Path(checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(f"--checkpoint not found: {ckpt}")
        return ckpt
    rd = Path(results_dir)
    for name in ("sparse_ae_best.pt", "sparse_ae.pt"):
        if (rd / name).exists():
            return rd / name
    best = sorted(rd.glob("*_best.pt"))
    if best:
        return best[0]
    pts = sorted(p for p in rd.glob("*.pt"))
    if len(pts) == 1:
        return pts[0]
    if len(pts) > 1:
        raise ValueError(
            f"Multiple checkpoints in {rd}; pass --checkpoint to disambiguate: "
            f"{[p.name for p in pts]}"
        )
    raise FileNotFoundError(f"No .pt checkpoint found in {rd}")


def load_config(results_dir):
    cfg_path = Path(results_dir) / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return {}


def build_model_from_checkpoint(ckpt, config, device):
    """
    Reconstruct the trained SAE architecture and load its weights.

    Structural hyperparameters (X, grid_points, m, eps, lista_steps, method,
    and displacement_center when centered) come from the checkpoint; activation
    details come from config.json with the same defaults the training runner
    uses.  normalize_atoms / per_atom_gain / lateral_init are fixed to the values
    hardcoded in
    pointcloud_run_experiments.py.
    """
    method = ckpt.get("method", "displacement")
    X = ckpt["X"].float()
    grid_points = ckpt["grid_points"]
    m = int(ckpt["m"])
    eps = float(ckpt["eps"])
    lista_steps = int(ckpt.get("lista_steps", 20))

    activation_type = config.get("activation_type") or "relu"
    topk_k = int(config.get("topk_k", 3) or 3)
    softtopk_tau_start = config.get("softtopk_tau_start")
    softtopk_tau = (float(softtopk_tau_start)
                    if softtopk_tau_start is not None
                    else float(config.get("softtopk_tau", 0.1) or 0.1))

    model_kwargs = dict(
        m=m, eps=eps,
        grid_side=int(config.get("grid_side", 10) or 10),
        lista_steps=lista_steps,
        grid_points=grid_points,
        normalize_atoms=True,
        per_atom_gain=True,
        lateral_init="damped_identity",
        activation_type=activation_type,
        topk_k=topk_k,
        softtopk_tau=softtopk_tau,
    )

    if method == "displacement":
        model_cls = DisplacementFieldSAE
        model_args = (X,)
    elif method in ("displacement_centered", "centered_displacement"):
        model_cls = CenteredDisplacementFieldSAE
        if "displacement_center" not in ckpt:
            raise KeyError(
                "Centered checkpoint is missing 'displacement_center'. "
                "Retrain with the updated pointcloud_run_experiments.py."
            )
        model_args = (X, ckpt["displacement_center"].float())
    elif method == "raw_map":
        model_cls = TransportMapSAE
        model_args = (X,)
    else:
        raise NotImplementedError(
            f"LWDL ablation evaluator supports methods 'displacement_centered', "
            f"'displacement', and 'raw_map'; got {method!r}."
        )

    # Only pass kwargs the installed model actually accepts -- the SAE classes
    # have gained optional args over time (e.g. softtopk_tau), so filtering by
    # the constructor signature keeps this evaluator compatible with both older
    # and newer mnist_sae_models.py without crashing on unknown keywords.
    valid = set(inspect.signature(model_cls.__init__).parameters)
    dropped = [k for k in model_kwargs if k not in valid]
    if dropped:
        print(f"  Note: model {model_cls.__name__} does not accept "
              f"{dropped}; using its defaults for those.")
    model_kwargs = {k: v for k, v in model_kwargs.items() if k in valid}
    model = model_cls(*model_args, **model_kwargs)

    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        print(f"  Warning: state_dict mismatch. missing={list(missing)} "
              f"unexpected={list(unexpected)}")
    model.to(device).eval()
    return model, method, X


# ============================================================
# Encode / reconstruct
# ============================================================

@torch.no_grad()
def encode_reconstruct(model, maps_subset, device, batch_size=128):
    """
    Run the SAE forward pass over a subset of maps.

    Returns:
        codes: (Ns, m) LWDL coefficients (cpu)
        recon: (Ns, n, d) reconstructed maps (cpu)
    """
    codes_chunks = []
    recon_chunks = []
    N = maps_subset.shape[0]
    for start in range(0, N, batch_size):
        batch = maps_subset[start:start + batch_size].to(device).float()
        T_hat, lam = model(batch)
        codes_chunks.append(lam.detach().cpu())
        recon_chunks.append(T_hat.detach().cpu())
    codes = torch.cat(codes_chunks, dim=0)
    recon = torch.cat(recon_chunks, dim=0)
    return codes, recon


# ============================================================
# Programmatic entry point (used by the orchestrator)
# ============================================================

def evaluate_lwdl(data_dir, results_dir, split_path, checkpoint=None,
                  batch_size=128, device="auto", classes=None,
                  test_fraction=0.1, seed=42):
    """
    Rehydrate the trained LWDL model and encode/reconstruct both splits.

    Returns a dict with train/test codes, train/test reconstructed maps,
    train/test true maps, labels, the base support X, class names, the LWDL
    method, and runtime.  All tensors are on CPU.
    """
    dev = resolve_device(device)
    ckpt_path = find_checkpoint(results_dir, checkpoint)
    print(f"LWDL checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = load_config(results_dir)

    model, method, X = build_model_from_checkpoint(ckpt, config, dev)

    if classes is None:
        classes = config.get("classes")
    X_data, maps, labels, class_names = load_with_labels(data_dir, classes=classes)
    maps = maps.float()

    n_model = X.shape[0]
    n_data = maps.shape[1]
    if n_model != n_data:
        raise ValueError(
            f"Support-size mismatch: checkpoint X has n={n_model} points but "
            f"maps in {data_dir} have n={n_data}. The LWDL results dir and the "
            f"data dir must share the same base support. Pick a matching pair "
            f"(e.g. results/modelnet10_6cls_uniform with datasets/"
            f"modelnet10_6cls_ot)."
        )

    train_idx, test_idx = make_split_indices(
        labels, test_fraction=test_fraction, seed=seed,
        split_path=split_path, stratified=True,
    )
    train_idx_t = torch.as_tensor(train_idx, dtype=torch.long)
    test_idx_t = torch.as_tensor(test_idx, dtype=torch.long)

    t0 = time.time()
    train_codes, train_recon = encode_reconstruct(
        model, maps[train_idx_t], dev, batch_size)
    test_codes, test_recon = encode_reconstruct(
        model, maps[test_idx_t], dev, batch_size)
    runtime = time.time() - t0

    return {
        "method": method,
        "checkpoint": str(ckpt_path),
        "X": X,
        "classes": class_names,
        "train_maps": maps[train_idx_t],
        "test_maps": maps[test_idx_t],
        "train_recon_maps": train_recon,
        "test_recon_maps": test_recon,
        "train_codes": train_codes,
        "test_codes": test_codes,
        "train_labels": labels[train_idx_t],
        "test_labels": labels[test_idx_t],
        "train_indices": train_idx,
        "test_indices": test_idx,
        "runtime_seconds": runtime,
    }


# ============================================================
# CLI: also compute the shared metric row and save
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wasserstein_metric", type=str, default="sqeuclidean",
                        choices=["sqeuclidean", "euclidean"])
    parser.add_argument("--max_wasserstein_samples", type=int, default=None)
    parser.add_argument("--save_reconstructions", action="store_true",
                        help="Also dump train/test reconstructed maps (large).")
    args = parser.parse_args()

    from pointcloud.pipeline.pointcloud_ablation_utils import (
        chamfer_loss,
        linear_probe,
        map_l2_loss,
        sparsity_stats,
        wasserstein_loss,
    )

    out = evaluate_lwdl(
        data_dir=args.data_dir, results_dir=args.results_dir,
        split_path=args.split_path, checkpoint=args.checkpoint,
        batch_size=args.batch_size, device=args.device,
        test_fraction=args.test_fraction, seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_map_l2 = map_l2_loss(out["train_maps"], out["train_recon_maps"])
    test_map_l2 = map_l2_loss(out["test_maps"], out["test_recon_maps"])
    wass, n_wass = wasserstein_loss(
        out["test_maps"], out["test_recon_maps"],
        metric=args.wasserstein_metric, max_samples=args.max_wasserstein_samples)
    chamfer = chamfer_loss(
        out["test_maps"], out["test_recon_maps"],
        max_samples=args.max_wasserstein_samples)
    spars = sparsity_stats(out["test_codes"])
    probe = linear_probe(out["train_codes"], out["train_labels"],
                         out["test_codes"], out["test_labels"], seed=args.seed)

    row = {
        "method": "LWDL-EOT",
        "lwdl_method": out["method"],
        "checkpoint": out["checkpoint"],
        "train_map_l2": train_map_l2,
        "test_map_l2": test_map_l2,
        "test_wasserstein": wass,
        "wasserstein_metric": args.wasserstein_metric,
        "n_wasserstein_samples": n_wass,
        "test_chamfer": chamfer,
        "mean_l0": spars["mean_l0"],
        "mean_l1": spars["mean_l1"],
        "frac_nonzero": spars["frac_nonzero"],
        "dense_code": False,
        "accuracy": probe["accuracy"],
        "macro_f1": probe["macro_f1"],
        "runtime_seconds": out["runtime_seconds"],
    }

    with open(output_dir / "lwdl_row.json", "w") as f:
        json.dump(row, f, indent=2)
    torch.save({
        "train_codes": out["train_codes"],
        "test_codes": out["test_codes"],
        "train_labels": out["train_labels"],
        "test_labels": out["test_labels"],
    }, output_dir / "lwdl_codes.pt")
    if args.save_reconstructions:
        torch.save({
            "train_recon_maps": out["train_recon_maps"],
            "test_recon_maps": out["test_recon_maps"],
        }, output_dir / "lwdl_reconstructions.pt")

    print(json.dumps(row, indent=2))
    print(f"Saved LWDL row to {output_dir / 'lwdl_row.json'}")


if __name__ == "__main__":
    main()
