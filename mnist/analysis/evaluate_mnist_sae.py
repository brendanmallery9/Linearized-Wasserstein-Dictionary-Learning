"""
Evaluation + visualization for MNIST WDL experiment checkpoints.

For each run in <checkpoint_dir>/metrics.json and <checkpoint_dir>/config.json:
    1. Reconstruction samples  -- target vs reconstructed transport maps.
    2. Atoms                   -- each atom's pushforward.
    3. Code embedding          -- atom usage histogram, per-class mean-code
                                   heatmap, 2D PCA of codes colored by digit.

Aggregated outputs:
    - recon_summary.png        -- bar chart of mean test reconstruction error
    - summary.json             -- mean recon error + mean active atoms per run
    - summary.txt              -- printable table

Usage:
    python mnist/analysis/evaluate_mnist_sae.py \\
        --checkpoint_dir 2D_WDL_results \\
        --data_dir datasets/mnist_ot \\
        --output_dir 2D_WDL_results/eval
"""

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MNIST_PIPELINE_DIR = REPO_ROOT / "mnist" / "pipeline"
if str(MNIST_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(MNIST_PIPELINE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from mnist_sae_models import DisplacementFieldSAE, SinkhornAtoms


# ============================================================
# Data loading with digit labels
# ============================================================

def load_with_labels(data_dir, digits=None):
    """
    Load base measure and transport maps, and return per-map digit labels.

    Returns:
        X: (n, 2)
        maps: (N, n, 2)
        labels: (N,) long tensor of digit labels in {0, ..., 9}
    """
    data_dir = Path(data_dir)
    X = torch.load(data_dir / "base_measure.pt", map_location="cpu").float()

    if digits is None:
        digits = sorted(
            int(p.name.split("_")[1])
            for p in data_dir.iterdir()
            if p.is_dir() and p.name.startswith("digit_")
        )

    all_maps, all_labels = [], []
    for d in digits:
        path = data_dir / f"digit_{d}" / "mappings.pt"
        if not path.exists():
            continue
        m = torch.load(path, map_location="cpu").float()
        all_maps.append(m)
        all_labels.append(torch.full((m.shape[0],), d, dtype=torch.long))
    maps = torch.cat(all_maps, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return X, maps, labels


def select_subset(labels, n_per_class, seed=0):
    """Deterministically pick n_per_class indices per digit class."""
    rng = np.random.RandomState(seed)
    classes = torch.unique(labels).tolist()
    idx = []
    for c in classes:
        c_idx = (labels == c).nonzero(as_tuple=True)[0].numpy()
        take = min(n_per_class, len(c_idx))
        picked = rng.choice(c_idx, size=take, replace=False)
        idx.append(picked)
    return np.concatenate(idx)


# ============================================================
# Model reconstruction from run config
# ============================================================

def build_model_from_cfg(cfg, X, device):
    """Rebuild a model from a train_mnist_sae config/result dict."""
    model = DisplacementFieldSAE(
        X,
        m=cfg["m"],
        eps=cfg["eps"],
        grid_side=cfg["grid_side"],
        lista_steps=cfg["lista_steps"],
        activation_type=cfg.get("activation_type", "relu"),
        normalize_atoms=cfg.get("normalize_atoms", True),
        grid_points=None,
        atoms_type=cfg.get("atoms_type", "gibbs"),
        n_sinkhorn=cfg.get("n_sinkhorn", 30),
        topk_k=cfg.get("topk_k", 3),
        per_atom_gain=cfg.get("per_atom_gain", True),
        lateral_init=cfg.get("lateral_init", "damped_identity"),
    ).to(device)
    return model


def load_run_configs(checkpoint_dir):
    """Load train_mnist_sae outputs and return per-checkpoint model configs."""
    config_path = checkpoint_dir / "config.json"
    metrics_path = checkpoint_dir / "metrics.json"
    if not config_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(
            f"Expected {config_path} and {metrics_path}. "
            "Run train_mnist_sae.py first."
        )

    with open(config_path) as f:
        config = json.load(f)
    with open(metrics_path) as f:
        metrics = json.load(f)

    runs = []
    for row in metrics:
        method = row["method"]
        eps = row["epsilon"]
        sparsity_coeff = row["sparsity_coeff"]
        run_cfg = {
            "name": f"{method}_eps{eps}_c{sparsity_coeff}",
            "method": method,
            "eps": eps,
            "sparsity_coeff": sparsity_coeff,
            "m": row.get("m", config.get("m", 30)),
            "grid_side": config.get("grid_side", 64),
            "lista_steps": config.get("lista_steps", 20),
            "activation_type": config.get("activation_type", "relu"),
            "normalize_atoms": config.get("normalize_atoms", True),
            "per_atom_gain": config.get("per_atom_gain", True),
            "lateral_init": config.get("lateral_init", "damped_identity"),
            "atoms_type": config.get("atoms_type", "gibbs"),
            "n_sinkhorn": config.get("n_sinkhorn", 30),
            "topk_k": config.get("topk_k", 3),
        }
        runs.append(run_cfg)
    return runs


# ============================================================
# Forward pass over a subset
# ============================================================

@torch.no_grad()
def forward_subset(model, maps, device, batch_size=128):
    """
    Run model over `maps` in batches.  Returns T_hat (N, n, 2), lam (N, m),
    per_sample_recon (N,) = (1/(2n)) ||T - T_hat||^2.
    """
    model.eval()
    n = model.n
    all_T_hat, all_lam, all_recon = [], [], []
    for i in range(0, maps.shape[0], batch_size):
        T = maps[i:i + batch_size].to(device)
        T_hat, lam = model(T)
        recon = 0.5 * ((T - T_hat) ** 2).sum(dim=(1, 2)) / n
        all_T_hat.append(T_hat.cpu())
        all_lam.append(lam.cpu())
        all_recon.append(recon.cpu())
    return (torch.cat(all_T_hat, dim=0),
            torch.cat(all_lam, dim=0),
            torch.cat(all_recon, dim=0))


# ============================================================
# Plots
# ============================================================

def plot_reconstructions(X, maps, labels, T_hat, out_path, n_per_class=2):
    """
    Grid: rows = digit classes (0..9), cols = n_per_class samples.
    Each cell: X (grey), T (red), T_hat (blue) scatter overlay.
    """
    X_np = X.cpu().numpy()
    maps_np = maps.cpu().numpy()
    T_hat_np = T_hat.cpu().numpy()
    labels_np = labels.cpu().numpy()

    classes = sorted(np.unique(labels_np).tolist())
    n_rows = len(classes)
    n_cols = n_per_class

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows),
                             squeeze=False)
    for r, c in enumerate(classes):
        c_idx = np.where(labels_np == c)[0][:n_per_class]
        for k in range(n_cols):
            ax = axes[r, k]
            if k < len(c_idx):
                j = c_idx[k]
                ax.scatter(X_np[:, 0], X_np[:, 1], s=2, c="lightgrey", alpha=0.5)
                ax.scatter(maps_np[j, :, 0], maps_np[j, :, 1], s=3, c="red",
                           alpha=0.6, label="T" if (r == 0 and k == 0) else None)
                ax.scatter(T_hat_np[j, :, 0], T_hat_np[j, :, 1], s=3, c="blue",
                           alpha=0.6, label="T_hat" if (r == 0 and k == 0) else None)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal")
            if k == 0:
                ax.set_ylabel(f"digit {c}", fontsize=9)
    axes[0, 0].legend(loc="upper left", fontsize=6)
    fig.suptitle("Reconstructions  (grey: X, red: T, blue: T_hat)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_atoms(model, out_path, name):
    """
    Grid of m atom scatter plots.  Each atom j is shown as its pushforward
    T_j(X) colored by x's horizontal position (to reveal the deformation).

    For SinkhornAtoms: a second file (out_path suffixed with '_nu') shows the
    learned target measure nu_j = softmax(psi_j) as a heatmap on the grid.
    """
    device = next(model.parameters()).device
    with torch.no_grad():
        atoms = model.atoms_module().cpu().numpy()  # (m, n, 2)
    X = model.atoms_module.X.cpu().numpy()          # (n, 2)
    m = atoms.shape[0]

    ncol = 6
    nrow = int(np.ceil(m / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2 * ncol, 2 * nrow), squeeze=False)
    for j in range(nrow * ncol):
        ax = axes[j // ncol, j % ncol]
        if j < m:
            ax.scatter(atoms[j, :, 0], atoms[j, :, 1], s=3,
                       c=X[:, 0], cmap="viridis", alpha=0.8)
            ax.set_title(f"atom {j}", fontsize=8)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
    fig.suptitle(f"{name} — atom pushforwards T_j(X)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    # Sinkhorn-only: target measures nu_j on the grid
    if isinstance(model.atoms_module, SinkhornAtoms):
        with torch.no_grad():
            log_b = F.log_softmax(model.atoms_module.psi, dim=1)  # (m, K)
            b = log_b.exp().cpu().numpy()
        Y = model.atoms_module.Y.cpu().numpy()
        K = Y.shape[0]
        side = int(round(np.sqrt(K)))
        # nu_j may not be on a regular grid if grid_points was custom; fall
        # back to a scatter with color = b_j if the grid isn't a square.
        on_grid = side * side == K

        nu_path = out_path.with_name(out_path.stem + "_nu.png")
        fig, axes = plt.subplots(nrow, ncol, figsize=(2 * ncol, 2 * nrow), squeeze=False)
        for j in range(nrow * ncol):
            ax = axes[j // ncol, j % ncol]
            if j < m:
                if on_grid:
                    img = b[j].reshape(side, side)
                    ax.imshow(img, origin="lower", extent=[0, 1, 0, 1], cmap="magma")
                else:
                    ax.scatter(Y[:, 0], Y[:, 1], s=3, c=b[j], cmap="magma")
                ax.set_title(f"nu_{j}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal")
        fig.suptitle(f"{name} — Sinkhorn target measures nu_j", fontsize=11)
        fig.tight_layout()
        fig.savefig(nu_path, dpi=130)
        plt.close(fig)


def plot_code_embedding(lam, labels, out_dir, name):
    """
    Three plots:
      1. atom_usage.png     -- mean |lam_j| over samples (bar) + activity fraction
      2. per_class_mean.png -- heatmap of mean lam_j per digit class
      3. pca.png            -- 2D PCA of codes colored by digit
    """
    lam_np = lam.numpy()                 # (N, m)
    labels_np = labels.cpu().numpy()     # (N,)
    N, m = lam_np.shape
    classes = sorted(np.unique(labels_np).tolist())

    # 1. Atom usage
    mean_abs = np.abs(lam_np).mean(axis=0)                    # (m,)
    active = (np.abs(lam_np) > 1e-6).astype(np.float32).mean(axis=0)  # (m,)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(6, 0.25 * m), 4))
    ax1.bar(range(m), mean_abs, color="steelblue")
    ax1.set_title(f"{name} — mean |lambda_j| over samples")
    ax1.set_xlabel("atom index"); ax1.set_ylabel("mean |lam|")
    ax2.bar(range(m), active, color="darkorange")
    ax2.set_title("activity fraction (|lam| > 1e-6)")
    ax2.set_xlabel("atom index"); ax2.set_ylabel("frac samples")
    ax2.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "atom_usage.png", dpi=130)
    plt.close(fig)

    # 2. Per-class mean heatmap: (n_classes, m)
    per_class = np.stack([lam_np[labels_np == c].mean(axis=0) for c in classes], axis=0)
    fig, ax = plt.subplots(figsize=(max(6, 0.3 * m), 0.4 * len(classes) + 1.5))
    im = ax.imshow(per_class, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([f"digit {c}" for c in classes])
    ax.set_xlabel("atom index")
    ax.set_title(f"{name} — mean code per class")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out_dir / "per_class_mean.png", dpi=130)
    plt.close(fig)

    # 3. PCA 2D
    try:
        from sklearn.decomposition import PCA
        codes_centered = lam_np - lam_np.mean(axis=0, keepdims=True)
        # Guard against degenerate case (all zeros)
        if np.allclose(codes_centered.std(), 0):
            raise ValueError("codes have zero variance")
        pca = PCA(n_components=2)
        proj = pca.fit_transform(codes_centered)  # (N, 2)
        fig, ax = plt.subplots(figsize=(6, 5))
        cmap = plt.get_cmap("tab10", len(classes))
        for ci, c in enumerate(classes):
            mask = labels_np == c
            ax.scatter(proj[mask, 0], proj[mask, 1], s=8, alpha=0.6,
                       color=cmap(ci), label=f"{c}")
        ax.set_title(f"{name} — PCA of codes (var explained: "
                     f"{pca.explained_variance_ratio_.sum():.2f})")
        ax.legend(markerscale=1.5, fontsize=8, loc="best", ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / "pca.png", dpi=130)
        plt.close(fig)
    except Exception as e:
        print(f"  [{name}] PCA skipped: {e}")


def plot_recon_summary(summary, out_path):
    """Bar chart of mean reconstruction loss across runs."""
    names = [s["name"] for s in summary]
    recons = [s["mean_recon"] for s in summary]
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(names)), 4.5))
    bars = ax.bar(range(len(names)), recons, color="steelblue")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("mean test recon loss")
    ax.set_title("Reconstruction error across runs")
    for b, v in zip(bars, recons):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main(checkpoint_dir, data_dir, output_dir, device_str="auto",
         n_per_class=50, recon_per_class=2, seed=0):
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve device
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")

    runs = load_run_configs(checkpoint_dir)

    # Load data (once, on CPU)
    print("Loading data...")
    X, maps, labels = load_with_labels(data_dir)
    print(f"  X: {tuple(X.shape)}, maps: {tuple(maps.shape)}, labels: {tuple(labels.shape)}")

    # Subset for evaluation
    idx = select_subset(labels, n_per_class=n_per_class, seed=seed)
    sub_maps = maps[idx]
    sub_labels = labels[idx]
    print(f"  Eval subset: {len(idx)} samples ({n_per_class}/class)")

    summary = []
    for cfg in runs:
        name = cfg["name"]
        ckpt_path = checkpoint_dir / f"{name}.pt"
        if not ckpt_path.exists():
            print(f"[skip] {name}: no checkpoint at {ckpt_path}")
            continue
        print(f"\n=== {name} ===")

        model = build_model_from_cfg(cfg, X, device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        T_hat, lam, per_recon = forward_subset(model, sub_maps, device)

        mean_recon = per_recon.mean().item()
        mean_active = (lam.abs() > 1e-6).float().sum(dim=1).mean().item()
        mean_l1 = lam.abs().sum(dim=1).mean().item()
        print(f"  mean_recon={mean_recon:.6f}  mean_active={mean_active:.2f}  "
              f"mean_l1={mean_l1:.4f}")

        run_out = output_dir / name
        run_out.mkdir(parents=True, exist_ok=True)

        # Pick a small subset of the eval subset for reconstructions viz
        recon_idx = select_subset(sub_labels, n_per_class=recon_per_class, seed=seed)
        plot_reconstructions(
            X, sub_maps[recon_idx], sub_labels[recon_idx],
            T_hat[recon_idx], run_out / "reconstructions.png",
            n_per_class=recon_per_class,
        )
        plot_atoms(model, run_out / "atoms.png", name)
        plot_code_embedding(lam, sub_labels, run_out, name)

        summary.append({
            "name": name,
            "mean_recon": mean_recon,
            "mean_active": mean_active,
            "mean_l1": mean_l1,
            "atoms_type": cfg.get("atoms_type", "gibbs"),
            "activation_type": cfg["activation_type"],
            "eps": cfg["eps"],
            "eps_anneal": cfg.get("eps_anneal", False),
        })

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Global summary
    plot_recon_summary(summary, output_dir / "recon_summary.png")
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Printable table
    lines = []
    header = f"{'name':<28} {'atoms':<10} {'act':<14} {'eps':>8} {'mean_recon':>12} {'mean_active':>12} {'mean_l1':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for s in summary:
        lines.append(f"{s['name']:<28} {s['atoms_type']:<10} {s['activation_type']:<14} "
                     f"{s['eps']:>8.4g} {s['mean_recon']:>12.6f} "
                     f"{s['mean_active']:>12.2f} {s['mean_l1']:>10.4f}")
    table = "\n".join(lines)
    print("\n" + table)
    with open(output_dir / "summary.txt", "w") as f:
        f.write(table + "\n")

    print(f"\nAll outputs under: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate + visualize trained WDL models.")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="2D_WDL_results")
    parser.add_argument("--data_dir", type=str, default="datasets/mnist_ot")
    parser.add_argument("--output_dir", type=str,
                        default="2D_WDL_results/eval")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--n_per_class", type=int, default=50,
                        help="Samples/class for code-embedding + recon metrics.")
    parser.add_argument("--recon_per_class", type=int, default=2,
                        help="Samples/class shown in reconstruction figures.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    main(
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device_str=args.device,
        n_per_class=args.n_per_class,
        recon_per_class=args.recon_per_class,
        seed=args.seed,
    )
