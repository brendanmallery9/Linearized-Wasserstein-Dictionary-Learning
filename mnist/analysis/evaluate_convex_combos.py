"""
Evaluate a sweep of convex-combination SAE runs in one shot.

Given a parent results directory containing one or more run subdirectories
(each produced by pointcloud_run_experiments.py: a config.json plus one or more
`<run_name>.pt` checkpoints), this script, for every checkpoint:

  1. Rebuilds the trained model and encodes the whole dataset to codes.
  2. Extracts the learned dictionary atoms (the transport-map images T_j(X)).
  3. Globally matches the learned atoms to the true generating digits with a
     Hungarian assignment, then computes MSE of the (simplex-normalized) codes
     against the true mixing weights.

It then emits a SINGLE combined dashboard so all runs can be compared at a
glance, plus a sorted MSE bar chart and a summary.csv -- rather than a pile of
per-run images.

Dashboard layout (one figure):
    row 0           : the true generating atoms + the true simplex (reference)
    one row per run : [ m learned-atom panels | learned 2-simplex | MSE bars ]

Usage:
    python mnist/analysis/evaluate_convex_combos.py \
        --results_dir pointcloud/results/convex_combos_034 \
        --data_dir    datasets/convex_combos_034 \
        --output_dir  pointcloud/results/convex_combos_034/eval
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment

# Resolve repo layout and make the model module importable.
REPO_ROOT = Path(__file__).resolve().parents[2]          # .../wdl_repo
MNIST_PIPELINE = REPO_ROOT / "mnist" / "pipeline"
for p in (REPO_ROOT, MNIST_PIPELINE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from mnist_sae_models import DisplacementFieldSAE, TransportMapSAE  # noqa: E402

# A stable palette so each true digit keeps one color across every panel.
_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


# ============================================================
# Device
# ============================================================
def resolve_device(device_str):
    if device_str == "auto":
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
# Dataset / run discovery
# ============================================================
def load_dataset(data_dir):
    """Load the convex-combos dataset: maps, true weights, true atoms, meta."""
    data_dir = Path(data_dir)
    X = torch.load(data_dir / "base_measure.pt", map_location="cpu").float()
    maps = torch.load(data_dir / "maps" / "maps.pt", map_location="cpu").float()
    coeffs = torch.load(data_dir / "coefficients" / "coefficients.pt",
                        map_location="cpu").float()
    meta = json.loads((data_dir / "metadata.json").read_text())
    base_maps_path = data_dir / "base_maps.pt"
    base_maps = (torch.load(base_maps_path, map_location="cpu").float()
                 if base_maps_path.exists() else None)
    digits = meta.get("digits", list(range(coeffs.shape[1])))
    return X, maps, coeffs, base_maps, digits, meta


def discover_runs(results_dir):
    """Find every (checkpoint, config) pair under results_dir.

    A run subdir holds a config.json and one or more `*.pt` checkpoints. The eval
    output dir (if nested) is skipped. Returns a list of dicts sorted for display.
    """
    results_dir = Path(results_dir)
    runs = []
    for cfg_path in sorted(results_dir.rglob("config.json")):
        run_dir = cfg_path.parent
        cfg = json.loads(cfg_path.read_text())
        for ckpt in sorted(run_dir.glob("*.pt")):
            runs.append({"ckpt": ckpt, "config": cfg, "dir": run_dir})
    if not runs:
        raise FileNotFoundError(
            f"No runs (config.json + *.pt) found under {results_dir}")

    def sort_key(r):
        act = r["config"].get("activation_type", "relu")
        # relu L1 sweep first (largest L1 first), enforced-sparsity variants last
        is_topk = act.startswith("topk")
        return (
            is_topk,
            float(r["config"].get("lr", 0.0)),
            _parse_eps(r["ckpt"]) or 0.0,
            -_parse_c(r["ckpt"]),
            act,
        )

    runs.sort(key=sort_key)
    return runs


def _parse_c(ckpt_path):
    """Pull the sparsity coeff out of a `..._c<value>.pt` checkpoint name."""
    stem = ckpt_path.stem
    if "_c" in stem:
        try:
            return float(stem.rsplit("_c", 1)[1].split("_", 1)[0])
        except ValueError:
            return 0.0
    return 0.0


def _parse_eps(ckpt_path):
    """Pull epsilon out of a `..._eps<value>_c...pt` checkpoint name."""
    stem = ckpt_path.stem
    if "_eps" not in stem:
        return None
    try:
        rest = stem.split("_eps", 1)[1]
        return float(rest.split("_c", 1)[0])
    except ValueError:
        return None


def _fmt_float(x):
    if x is None:
        return "?"
    try:
        return f"{float(x):g}"
    except (TypeError, ValueError):
        return str(x)


def run_label(run):
    """Human-readable label, e.g. 'relu L1=0.005' or 'topk_simplex k=2 lr=...'."""
    cfg = run["config"]
    act = cfg.get("activation_type", "relu")
    role = " best" if run["ckpt"].stem.endswith("_best") else ""
    if act.startswith("topk"):
        lr = _fmt_float(cfg.get("lr"))
        eps = _fmt_float(_parse_eps(run["ckpt"]))
        return f"{act} k={cfg.get('topk_k', '?')} lr={lr} eps={eps}{role}"
    return f"{act}  L1={_parse_c(run['ckpt']):g}{role}"


# ============================================================
# Model rebuild + encode
# ============================================================
def rebuild_model(ckpt_path, cfg, device):
    """Reconstruct a trained model exactly as the runner built it."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    method = ckpt.get("method", "displacement")
    model_cls = DisplacementFieldSAE if method == "displacement" else TransportMapSAE
    model = model_cls(
        ckpt["X"].float(),
        m=int(ckpt["m"]),
        eps=float(ckpt["eps"]),
        grid_side=int(cfg.get("grid_side", 32)),
        lista_steps=int(ckpt["lista_steps"]),
        grid_points=(ckpt["grid_points"].float()
                     if ckpt.get("grid_points") is not None else None),
        normalize_atoms=True,
        per_atom_gain=True,
        lateral_init="damped_identity",
        activation_type=cfg.get("activation_type", "relu"),
        topk_k=int(cfg.get("topk_k", 3)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def encode_dataset(model, maps, device, batch_size=256):
    """Return (codes (N,m), mean per-sample recon loss, mean #active atoms)."""
    n = model.n
    codes, recon_losses = [], []
    for start in range(0, maps.shape[0], batch_size):
        batch = maps[start:start + batch_size].to(device).float()
        recon, lam = model(batch)
        per_sample = 0.5 * ((batch - recon) ** 2).sum(dim=(1, 2)) / n
        codes.append(lam.cpu())
        recon_losses.append(per_sample.cpu())
    codes = torch.cat(codes).numpy()
    recon = float(torch.cat(recon_losses).mean())
    mean_active = float((codes > 1e-8).sum(axis=1).mean())
    return codes, recon, mean_active


@torch.no_grad()
def learned_atoms(model):
    """Dictionary atoms as transport-map images T_j(X): (m, n, d)."""
    return model.atoms_module().detach().cpu().numpy()


# ============================================================
# Matching + MSE against true mixing weights
# ============================================================
def simplex_normalize(codes):
    """L1-normalize each row onto the probability simplex (zero rows stay zero)."""
    s = codes.sum(axis=1, keepdims=True)
    return codes / np.clip(s, 1e-8, None)


def match_and_mse(codes, true_w):
    """Globally match learned atoms to true classes, then MSE of proportions.

    codes:  (N, m) raw nonneg codes
    true_w: (N, K) true mixing weights (rows sum to 1)

    Returns dict with the simplex-normalized matched codes, the learned->true
    column assignment, overall MSE, and per-true-class MSE.
    """
    P = simplex_normalize(codes)                  # (N, m)
    m, K = P.shape[1], true_w.shape[1]
    # cost[i, j] = MSE between learned-atom i's proportion and true-class j's
    cost = np.zeros((m, K))
    for i in range(m):
        for j in range(K):
            cost[i, j] = np.mean((P[:, i] - true_w[:, j]) ** 2)
    row_ind, col_ind = linear_sum_assignment(cost)   # min(m,K) matches
    learned_for_true = {int(j): int(i) for i, j in zip(row_ind, col_ind)}

    # Reorder matched learned columns into true-class order for a fair MSE.
    matched = np.zeros((P.shape[0], K))
    per_class = np.full(K, np.nan)
    for j in range(K):
        if j in learned_for_true:
            col = P[:, learned_for_true[j]]
            matched[:, j] = col
            per_class[j] = np.mean((col - true_w[:, j]) ** 2)
    overall = float(np.nanmean(per_class))
    return {
        "P": P,
        "matched": matched,
        "learned_for_true": learned_for_true,  # true class idx -> learned atom idx
        "mse": overall,
        "per_class_mse": per_class,
    }


# ============================================================
# Plot helpers
# ============================================================
def _plot_atom(ax, atom_xy, base_x, color_by, title, lims):
    ax.scatter(atom_xy[:, 0], atom_xy[:, 1], s=4, c=color_by,
               cmap="viridis", alpha=0.8, linewidths=0)
    ax.set_xlim(lims[0]); ax.set_ylim(lims[1])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def _ternary_xy(P3):
    """Map rows of a 3-col simplex array to 2D triangle coordinates."""
    v = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3) / 2]])
    return P3 @ v


def _draw_triangle(ax, corner_labels, corner_colors):
    v = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3) / 2]])
    tri = np.vstack([v, v[0]])
    ax.plot(tri[:, 0], tri[:, 1], color="0.6", lw=1.0, zorder=1)
    offs = [(-0.06, -0.06), (0.06, -0.06), (0.0, 0.06)]
    for k in range(3):
        ax.scatter([v[k, 0]], [v[k, 1]], s=60, c=corner_colors[k],
                   marker="o", zorder=4, edgecolors="k", linewidths=0.5)
        ax.annotate(corner_labels[k], v[k], textcoords="offset points",
                    xytext=(offs[k][0] * 100, offs[k][1] * 100),
                    ha="center", fontsize=10, fontweight="bold")
    ax.set_xlim(-0.15, 1.15); ax.set_ylim(-0.15, 1.05)
    ax.set_aspect("equal"); ax.axis("off")


def _plot_simplex(ax, P3, dom_class, digits, digit_colors, max_pts=3000):
    """Scatter 3-col simplex points, colored by true dominant class."""
    corner_colors = [digit_colors[d] for d in digits]
    _draw_triangle(ax, [str(d) for d in digits], corner_colors)
    keep = P3.sum(axis=1) > 1e-8
    P3, dom_class = P3[keep], dom_class[keep]
    if P3.shape[0] > max_pts:
        sel = np.random.RandomState(0).choice(P3.shape[0], max_pts, replace=False)
        P3, dom_class = P3[sel], dom_class[sel]
    xy = _ternary_xy(P3)
    cols = [digit_colors[digits[c]] for c in dom_class]
    ax.scatter(xy[:, 0], xy[:, 1], s=6, c=cols, alpha=0.45, linewidths=0, zorder=3)


def _plot_mse_bars(ax, per_class, digits, digit_colors, overall, recon, active):
    cols = [digit_colors[d] for d in digits]
    y = np.arange(len(digits))
    vals = np.nan_to_num(per_class, nan=0.0)
    ax.barh(y, vals, color=cols)
    ax.set_yticks(y); ax.set_yticklabels([f"digit {d}" for d in digits], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("per-class MSE", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_title(f"MSE={overall:.4f}\nrecon={recon:.4f}  active={active:.2f}",
                 fontsize=9)
    for yi, vv in zip(y, vals):
        ax.text(vv, yi, f" {vv:.4f}", va="center", fontsize=7)


# ============================================================
# Main evaluation
# ============================================================
def evaluate(results_dir, data_dir, output_dir, device_str="auto",
             max_samples=None, batch_size=256):
    device = resolve_device(device_str)
    print(f"Device: {device}")

    X, maps, true_w, base_maps, digits, meta = load_dataset(data_dir)
    print(f"Dataset: maps={tuple(maps.shape)} true_w={tuple(true_w.shape)} "
          f"digits={digits}")
    true_w_np = true_w.numpy()

    # Optional subsample for speed (kept identical across all runs).
    if max_samples is not None and maps.shape[0] > max_samples:
        idx = np.sort(np.random.RandomState(0).choice(
            maps.shape[0], max_samples, replace=False))
        maps, true_w_np = maps[idx], true_w_np[idx]
        print(f"Subsampled to {maps.shape[0]} maps for evaluation")

    digit_colors = {d: _PALETTE[i % len(_PALETTE)] for i, d in enumerate(digits)}
    dom_class = true_w_np.argmax(axis=1)

    runs = discover_runs(results_dir)
    print(f"Found {len(runs)} run(s):")
    for r in runs:
        print(f"  - {run_label(r):24s}  {r['ckpt'].relative_to(Path(results_dir).parent)}")

    results = []
    for r in runs:
        model = rebuild_model(r["ckpt"], r["config"], device)
        codes, recon, active = encode_dataset(model, maps, device, batch_size)
        atoms = learned_atoms(model)
        match = match_and_mse(codes, true_w_np)
        results.append({"run": r, "model_m": atoms.shape[0], "atoms": atoms,
                        "recon": recon, "active": active, **match})
        print(f"  {run_label(r):24s}  MSE={match['mse']:.5f}  "
              f"recon={recon:.5f}  active={active:.2f}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _build_dashboard(results, base_maps, X, true_w_np, dom_class, digits,
                     digit_colors, meta, out / "dashboard.png")
    _build_mse_comparison(results, out / "mse_comparison.png")
    _write_summary_csv(results, digits, out / "summary.csv")
    _print_summary(results, digits)
    print(f"\nSaved dashboard:      {out / 'dashboard.png'}")
    print(f"Saved MSE comparison: {out / 'mse_comparison.png'}")
    print(f"Saved summary table:  {out / 'summary.csv'}")
    return results


def _auto_lims(*arrays, pad=0.08):
    pts = np.concatenate([a.reshape(-1, a.shape[-1]) for a in arrays], axis=0)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    d = (hi - lo) * pad
    return [(lo[0] - d[0], hi[0] + d[0]), (lo[1] - d[1], hi[1] + d[1])]


def _build_dashboard(results, base_maps, X, true_w, dom_class, digits,
                     digit_colors, meta, path):
    m = results[0]["model_m"]
    n_rows = len(results) + 1                 # +1 reference row (true atoms)
    n_cols = m + 2                            # m atoms + simplex + mse
    base_x = X.numpy()[:, 0]
    can_ternary = (m == 3 and len(digits) == 3)

    atom_arrays = [r["atoms"] for r in results]
    if base_maps is not None:
        atom_arrays.append(base_maps.numpy())
    lims = _auto_lims(*[a for a in atom_arrays])

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.0 * n_cols, 3.0 * n_rows), squeeze=False)

    # ---- Reference row: true generating atoms + true simplex ----
    for j in range(m):
        ax = axes[0][j]
        if base_maps is not None and j < base_maps.shape[0]:
            _plot_atom(ax, base_maps[j].numpy(), base_x, base_x,
                       f"TRUE digit {digits[j]}", lims)
        else:
            ax.axis("off")
    ax_s = axes[0][m]
    if can_ternary:
        _plot_simplex(ax_s, true_w, dom_class, digits, digit_colors)
        ax_s.set_title("TRUE mixing weights", fontsize=9)
    else:
        ax_s.axis("off")
    ax_t = axes[0][m + 1]
    ax_t.axis("off")
    ax_t.text(0.0, 0.5,
              f"dataset: {meta.get('format', '?')}\n"
              f"digits = {digits}\n"
              f"subset_size = {meta.get('subset_size', '?')}\n"
              f"n_samples = {meta.get('n_samples', '?')}\n"
              f"support n = {meta.get('support_size', '?')}\n\n"
              f"MSE = mean sq. error of\nsimplex-normalized codes\nvs true weights "
              f"(Hungarian-\nmatched atoms)",
              fontsize=9, va="center", family="monospace")

    # ---- One row per run ----
    for ri, res in enumerate(results):
        row = ri + 1
        label = run_label(res["run"])
        inv = res["learned_for_true"]              # true j -> learned i
        learned_to_true = {i: j for j, i in inv.items()}
        for i in range(m):
            ax = axes[row][i]
            tgt = learned_to_true.get(i)
            ttl = (f"atom {i} -> digit {digits[tgt]}" if tgt is not None
                   else f"atom {i} (unmatched)")
            _plot_atom(ax, res["atoms"][i], base_x, base_x, ttl, lims)
        # row label on the left-most atom panel
        axes[row][0].set_ylabel(label, fontsize=11, fontweight="bold")
        ax_s = axes[row][m]
        if can_ternary:
            # reorder matched columns into the true digit order for the corners
            _plot_simplex(ax_s, res["matched"], dom_class, digits, digit_colors)
        else:
            ax_s.axis("off")
        _plot_mse_bars(axes[row][m + 1], res["per_class_mse"], digits,
                       digit_colors, res["mse"], res["recon"], res["active"])

    fig.suptitle("Convex-combinations SAE sweep -- learned atoms, code simplex, "
                 "and MSE vs true mixing weights", fontsize=14, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _build_mse_comparison(results, path):
    labels = [run_label(r["run"]) for r in results]
    mses = [r["mse"] for r in results]
    order = np.argsort(mses)
    labels = [labels[i] for i in order]
    mses = [mses[i] for i in order]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(labels) + 1.5))
    y = np.arange(len(labels))
    bars = ax.barh(y, mses, color="#4c72b0")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("MSE vs true mixing weights (lower is better)")
    ax.set_title("Convex-combos sweep: matched MSE by run")
    for yi, vv in zip(y, mses):
        ax.text(vv, yi, f"  {vv:.5f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _write_summary_csv(results, digits, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["run", "checkpoint_role", "activation", "topk_k", "lr", "eps",
                  "epochs", "batch_size", "lista_steps", "grid_support_size",
                  "l1", "recon_loss", "mean_active", "matched_mse"]
        header += [f"mse_digit_{d}" for d in digits]
        w.writerow(header)
        for r in results:
            cfg = r["run"]["config"]
            act = cfg.get("activation_type", "relu")
            role = "best" if r["run"]["ckpt"].stem.endswith("_best") else "last"
            row = [run_label(r["run"]), role, act, cfg.get("topk_k", ""),
                   _fmt_float(cfg.get("lr")),
                   _fmt_float(_parse_eps(r["run"]["ckpt"])),
                   cfg.get("epochs", ""), cfg.get("batch_size", ""),
                   cfg.get("lista_steps", ""),
                   cfg.get("grid_support_size", ""),
                   _parse_c(r["run"]["ckpt"]),
                   f"{r['recon']:.6f}", f"{r['active']:.3f}", f"{r['mse']:.6f}"]
            row += [f"{v:.6f}" if not np.isnan(v) else "nan"
                    for v in r["per_class_mse"]]
            w.writerow(row)


def _print_summary(results, digits):
    print("\n" + "=" * 72)
    print(f"{'run':24s} {'matched_mse':>12s} {'recon':>10s} {'active':>8s}")
    print("-" * 72)
    for r in sorted(results, key=lambda x: x["mse"]):
        print(f"{run_label(r['run']):24s} {r['mse']:12.5f} "
              f"{r['recon']:10.5f} {r['active']:8.2f}")
    print("=" * 72)


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a convex-combinations SAE sweep into one dashboard.")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Parent dir with run subdirs (config.json + *.pt).")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Convex-combos dataset dir (maps, coefficients, base_maps).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write the dashboard (default: <results_dir>/eval).")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--max_samples", type=int, default=20000,
                        help="Subsample the dataset to this many maps for speed.")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    output_dir = args.output_dir or str(Path(args.results_dir) / "eval")
    evaluate(args.results_dir, args.data_dir, output_dir,
             device_str=args.device, max_samples=args.max_samples,
             batch_size=args.batch_size)
