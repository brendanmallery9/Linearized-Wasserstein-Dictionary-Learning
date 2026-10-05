"""
Nearest-neighbor diagnostics for convex-combo learned atoms.

For each trained checkpoint, this script:
  1. Rebuilds the SAE and extracts learned transport-map atoms.
  2. Matches learned atoms to planted convex-combo atoms by L2(rho).
  3. Searches a validation pool of real OT maps, with the planted atoms appended
     as flagged reference candidates.
  4. Saves a row-per-atom figure: learned atom + top nearest validation maps.

The default preset covers the recent convex-combo runs discussed in the
experiment thread.
"""

import argparse
import csv
import html
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[2]
MNIST_PIPELINE = REPO_ROOT / "mnist" / "pipeline"
for p in (REPO_ROOT, MNIST_PIPELINE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from mnist_sae_models import (  # noqa: E402
    CenteredDisplacementFieldSAE,
    DisplacementFieldSAE,
    PCRemovedCenteredDisplacementFieldSAE,
    TransportMapSAE,
    WhitenedCenteredDisplacementFieldSAE,
    load_model_state,
)


@dataclass(frozen=True)
class Case:
    label: str
    results_dir: Path
    data_dir: Path
    ot_dir: Path


SELECTED_CASES = [
    Case(
        "02468_final_softtopk",
        Path("mnist/results/convex_combos_02468_grid64_finalsofttopk_k2_tau025to005_hold1000_l10_lista10_wd0_e3000"),
        Path("datasets/convex_combos_02468"),
        Path("datasets/mnist_ot"),
    ),
    Case(
        "02468_final_softtopk_simplex",
        Path("mnist/results/convex_combos_02468_grid64_finalsofttopk_simplex_k2_tau025to005_hold1000_l10_lista10_wd0_e3000"),
        Path("datasets/convex_combos_02468"),
        Path("datasets/mnist_ot"),
    ),
    Case(
        "02468_pure100_simplex",
        Path("mnist/results/convex_combos_02468_pure100_grid64_finalsofttopk_simplex_k2_tau025to005_hold1000_l10_lista10_wd0_e3000"),
        Path("datasets/convex_combos_02468_pure100"),
        Path("datasets/mnist_ot"),
    ),
    Case(
        "034_no_pure_simplex",
        Path("mnist/results/convex_combos_034_grid64_finalsofttopk_simplex_k2_tau025to005_hold1000_l10_lista10_wd0_e3000"),
        Path("datasets/convex_combos_034"),
        Path("datasets/mnist_ot"),
    ),
    Case(
        "chinese_123456_lista30",
        Path("mnist/results/convex_combos_chinese_mnist_123456_grid64_finalsofttopk_k2_tau025to005_hold1000_l10_lista30_wd0_e3000"),
        Path("datasets/convex_combos_chinese_mnist_123456_uniform_supp400"),
        Path("datasets/chinese_mnist_ot_123456_uniform_supp400"),
    ),
    Case(
        "02468_final_jumprelu_l1",
        Path("mnist/results/convex_combos_02468_grid64_finaljumprelu_l1_5e_05_reconbest"),
        Path("datasets/convex_combos_02468"),
        Path("datasets/mnist_ot"),
    ),
    Case(
        "chinese_123456_final_jumprelu_l1_sweep",
        Path("mnist/results/convex_combos_chinese_mnist_123456_uniform_supp400_grid64_finaljumprelu_l1_sweep"),
        Path("datasets/convex_combos_chinese_mnist_123456_uniform_supp400"),
        Path("datasets/chinese_mnist_ot_123456_uniform_supp400"),
    ),
]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def l2rho_sq(a, b):
    diff = np.asarray(a) - np.asarray(b)
    return np.mean(np.sum(diff * diff, axis=-1), axis=-1)


def train_test_indices(n, test_fraction, seed):
    n_test = int(n * test_fraction)
    gen = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=gen).numpy()
    return perm[: n - n_test], perm[n - n_test :]


def infer_atoms_type(state_dict):
    if "atoms_module.H_raw" in state_dict:
        return "gibbs"
    if "atoms_module.psi" in state_dict:
        return "sinkhorn"
    raise ValueError("Could not infer atom parameterization from checkpoint.")


def load_combo_dataset(data_dir):
    data_dir = resolve(data_dir)
    X = torch.load(data_dir / "base_measure.pt", map_location="cpu").float()
    true_atoms = torch.load(data_dir / "base_maps.pt", map_location="cpu").float()
    meta = json.loads((data_dir / "metadata.json").read_text())
    digits = [int(d) for d in meta["digits"]]
    return X, true_atoms, digits, meta


def rebuild_model(ckpt_path, config_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = json.loads(Path(config_path).read_text())
    state = ckpt["model_state"]
    method = ckpt.get("method", "displacement")
    if method == "displacement":
        model_cls = DisplacementFieldSAE
    elif method == "displacement_centered":
        model_cls = CenteredDisplacementFieldSAE
    elif method == "displacement_centered_pc_removed":
        model_cls = PCRemovedCenteredDisplacementFieldSAE
    elif method == "displacement_centered_whitened":
        model_cls = WhitenedCenteredDisplacementFieldSAE
    else:
        model_cls = TransportMapSAE
    grid_points = ckpt.get("grid_points")
    model_kwargs = dict(
        m=int(ckpt["m"]),
        eps=float(ckpt["eps"]),
        grid_side=int(cfg.get("grid_side", 32)),
        lista_steps=int(ckpt["lista_steps"]),
        activation_type=cfg.get("activation_type", "relu"),
        normalize_atoms=True,
        grid_points=grid_points.float() if grid_points is not None else None,
        atoms_type=infer_atoms_type(state),
        n_sinkhorn=int(cfg.get("n_sinkhorn", 30)),
        topk_k=int(cfg.get("topk_k", 3)),
        softtopk_tau=float(cfg.get("softtopk_tau", 0.1)),
        per_atom_gain=("encoder.gain" in state),
        lateral_init="damped_identity",
    )
    if method == "displacement_centered":
        center = ckpt.get("displacement_center")
        if center is None:
            raise KeyError(f"{ckpt_path} is displacement_centered but has no displacement_center")
        model = model_cls(ckpt["X"].float(), center.float(), **model_kwargs).to(device)
    elif method == "displacement_centered_pc_removed":
        center = ckpt.get("displacement_center")
        pc_component = ckpt.get("displacement_pc_component")
        if center is None:
            raise KeyError(f"{ckpt_path} is displacement_centered_pc_removed but has no displacement_center")
        if pc_component is None:
            raise KeyError(f"{ckpt_path} is displacement_centered_pc_removed but has no displacement_pc_component")
        model = model_cls(
            ckpt["X"].float(), center.float(), pc_component.float(),
            **model_kwargs,
        ).to(device)
    elif method == "displacement_centered_whitened":
        center = ckpt.get("displacement_center")
        whitening_matrix = ckpt.get("whitening_matrix")
        unwhitening_matrix = ckpt.get("unwhitening_matrix")
        if center is None:
            raise KeyError(f"{ckpt_path} is displacement_centered_whitened but has no displacement_center")
        if whitening_matrix is None or unwhitening_matrix is None:
            raise KeyError(f"{ckpt_path} is displacement_centered_whitened but has no whitening matrices")
        model = model_cls(
            ckpt["X"].float(), center.float(),
            whitening_matrix.float(), unwhitening_matrix.float(),
            **model_kwargs,
        ).to(device)
    else:
        model = model_cls(ckpt["X"].float(), **model_kwargs).to(device)
    load_model_state(model, state)
    model.eval()
    return model, ckpt, cfg


@torch.no_grad()
def get_learned_atoms(ckpt_path, config_path, device):
    model, ckpt, cfg = rebuild_model(ckpt_path, config_path, device)
    atoms = model.atoms_module().detach().cpu().numpy()
    return atoms, ckpt, cfg


def match_atoms(learned_atoms, true_atoms):
    cost = np.zeros((learned_atoms.shape[0], true_atoms.shape[0]), dtype=np.float64)
    for i in range(learned_atoms.shape[0]):
        for j in range(true_atoms.shape[0]):
            cost[i, j] = float(l2rho_sq(learned_atoms[i], true_atoms[j]))
    rows, cols = linear_sum_assignment(cost)
    true_to_learned = {int(j): int(i) for i, j in zip(rows, cols)}
    return cost, true_to_learned


def digit_dirs(ot_dir, allowed_digits=None):
    allowed = None if allowed_digits is None else {int(d) for d in allowed_digits}
    dirs = []
    for p in sorted(resolve(ot_dir).glob("digit_*")):
        if (p / "mappings.pt").exists():
            try:
                val = int(p.name.split("_", 1)[1])
            except ValueError:
                continue
            if allowed is not None and val not in allowed:
                continue
            dirs.append((val, p))
    if not dirs:
        suffix = "" if allowed is None else f" for digits {sorted(allowed)}"
        raise FileNotFoundError(f"No digit_*/mappings.pt found under {ot_dir}{suffix}")
    return dirs


def validation_pool(ot_dir, combo_X, combo_digits, meta, test_fraction, seed,
                    restrict_to_combo_digits=True):
    ot_dir = resolve(ot_dir)
    ot_X = torch.load(ot_dir / "base_measure.pt", map_location="cpu").float()
    if tuple(ot_X.shape) != tuple(combo_X.shape) or not torch.allclose(ot_X, combo_X, atol=1e-6):
        raise ValueError(
            f"Base support mismatch: combo {tuple(combo_X.shape)} vs "
            f"{ot_dir} {tuple(ot_X.shape)}"
        )

    original_idx = int(meta.get("digit_sample_index", 0))
    maps, digits, indices, is_planted_ref = [], [], [], []
    all_digit_maps = {}
    allowed_digits = combo_digits if restrict_to_combo_digits else None
    for digit, path in digit_dirs(ot_dir, allowed_digits=allowed_digits):
        maps_d = torch.load(path / "mappings.pt", map_location="cpu").float().numpy()
        all_digit_maps[digit] = maps_d
        _, test_idx = train_test_indices(len(maps_d), test_fraction, seed + digit)
        test_idx = np.array([i for i in test_idx if i != original_idx], dtype=int)
        if test_idx.size == 0 and len(maps_d) > 1:
            test_idx = np.array([i for i in range(len(maps_d)) if i != original_idx], dtype=int)
        if test_idx.size:
            maps.append(maps_d[test_idx])
            digits.append(np.full(test_idx.size, digit, dtype=int))
            indices.append(test_idx)
            is_planted_ref.append(np.zeros(test_idx.size, dtype=bool))

    planted_maps = []
    for digit in combo_digits:
        if digit not in all_digit_maps:
            raise KeyError(f"Digit {digit} missing from validation OT dir {ot_dir}")
        if original_idx >= len(all_digit_maps[digit]):
            raise IndexError(f"digit_sample_index={original_idx} is out of range for digit {digit}")
        planted_maps.append(all_digit_maps[digit][original_idx])
    planted_maps = np.stack(planted_maps, axis=0)
    maps.append(planted_maps)
    digits.append(np.array(combo_digits, dtype=int))
    indices.append(np.full(len(combo_digits), original_idx, dtype=int))
    is_planted_ref.append(np.ones(len(combo_digits), dtype=bool))

    return {
        "maps": np.concatenate(maps, axis=0),
        "digits": np.concatenate(digits, axis=0),
        "indices": np.concatenate(indices, axis=0),
        "is_planted_ref": np.concatenate(is_planted_ref, axis=0),
    }


def discover_checkpoints(case, role):
    results_dir = resolve(case.results_dir)
    ckpts = []
    for cfg in sorted(results_dir.rglob("config.json")):
        candidates = sorted(cfg.parent.glob("*.pt"))
        if role == "best":
            candidates = [p for p in candidates if p.stem.endswith("_best")]
        elif role == "last":
            candidates = [p for p in candidates if not p.stem.endswith("_best")]
        for ckpt in candidates:
            ckpts.append((ckpt, cfg))
    if not ckpts:
        raise FileNotFoundError(f"No {role} checkpoints found under {results_dir}")
    return ckpts


def checkpoint_label(ckpt_path, cfg):
    stem = ckpt_path.stem
    role = "best" if stem.endswith("_best") else "last"
    l1 = "?"
    if "_c" in stem:
        l1 = stem.rsplit("_c", 1)[1].replace("_best", "")
    return (
        f"{cfg.get('activation_type', 'relu')} {role} "
        f"L1={l1} steps={cfg.get('lista_steps', '?')}"
    )


def safe_stem(text):
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def auto_lims(*arrays, pad=0.06):
    pts = np.concatenate([np.asarray(a).reshape(-1, np.asarray(a).shape[-1]) for a in arrays], axis=0)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    delta = np.maximum(hi - lo, 1e-3) * pad
    return (lo[0] - delta[0], hi[0] + delta[0]), (lo[1] - delta[1], hi[1] + delta[1])


def scatter_map(ax, pts, color, lims, *, s=5, alpha=0.86):
    ax.scatter(pts[:, 0], pts[:, 1], color=color, s=s, alpha=alpha, linewidths=0)
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("0.78")


def color_for_digit(digit):
    if 0 <= int(digit) <= 9:
        return mpl.colormaps["tab10"](int(digit))
    return mpl.colormaps["tab20"](abs(int(digit)) % 20)


def make_figure(case_label, ckpt_label_text, learned_atoms, true_atoms, digits, match_cost,
                true_to_learned, pool, top_k, out_png, summary_writer):
    lims = auto_lims(learned_atoms, true_atoms, pool["maps"])
    n_rows = len(digits)
    n_cols = top_k + 2
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(1.55 * n_cols + 1.2, 1.58 * n_rows + 1.0),
        squeeze=False,
        constrained_layout=True,
    )

    for row, digit in enumerate(digits):
        learned_j = true_to_learned[row]
        learned_atom = learned_atoms[learned_j]
        losses = l2rho_sq(pool["maps"], learned_atom)
        order = np.argsort(losses)[:top_k]

        ax = axes[row, 0]
        scatter_map(ax, true_atoms[row], color="0.25", lims=lims, s=5, alpha=0.78)
        ax.set_title(f"planted {digit}", fontsize=7, pad=5)

        ax = axes[row, 1]
        scatter_map(ax, learned_atom, color="#d95f02", lims=lims, s=5, alpha=0.86)
        ax.set_title(
            f"learned {learned_j}\natom MSE {match_cost[learned_j, row]:.1e}",
            fontsize=7,
            pad=5,
        )

        for rank, pool_idx in enumerate(order, start=1):
            ax = axes[row, rank + 1]
            nd = int(pool["digits"][pool_idx])
            is_original_atom = bool(pool["is_planted_ref"][pool_idx])
            scatter_map(ax, pool["maps"][pool_idx], color=color_for_digit(nd), lims=lims, s=5, alpha=0.84)
            if is_original_atom:
                for spine in ax.spines.values():
                    spine.set_color("#111111")
                    spine.set_linewidth(1.6)
            original_note = "\nORIGINAL ATOM" if is_original_atom else ""
            ax.set_title(
                f"#{rank} digit {nd}{original_note}\nMSE {losses[pool_idx]:.1e}",
                fontsize=7,
                pad=5,
            )
            summary_writer.writerow({
                "case": case_label,
                "checkpoint": ckpt_label_text,
                "matched_planted_digit": digit,
                "matched_learned_atom": learned_j,
                "rank": rank,
                "neighbor_digit": nd,
                "neighbor_index": int(pool["indices"][pool_idx]),
                "neighbor_l2rho_sq": f"{float(losses[pool_idx]):.8g}",
                "is_planted_reference": bool(pool["is_planted_ref"][pool_idx]),
                "matched_atom_l2rho_sq": f"{float(match_cost[learned_j, row]):.8g}",
            })

    fig.suptitle(
        f"{case_label}: nearest validation OT maps to learned atoms\n"
        f"{ckpt_label_text}; dark outline means appended planted atom reference",
        fontsize=10,
    )
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def run_case(case, out_root, role, top_k, device, restrict_to_combo_digits=True):
    X, true_atoms_t, digits, meta = load_combo_dataset(case.data_dir)
    true_atoms = true_atoms_t.numpy()
    test_fraction = float(meta.get("test_fraction", 0.1))
    seed = int(meta.get("seed", 42))
    pool = validation_pool(
        case.ot_dir, X, digits, meta, test_fraction, seed,
        restrict_to_combo_digits=restrict_to_combo_digits,
    )

    case_out = out_root / safe_stem(case.label)
    case_out.mkdir(parents=True, exist_ok=True)
    csv_path = case_out / "nearest_neighbors.csv"
    with csv_path.open("w", newline="") as f:
        fields = [
            "case",
            "checkpoint",
            "matched_planted_digit",
            "matched_learned_atom",
            "rank",
            "neighbor_digit",
            "neighbor_index",
            "neighbor_l2rho_sq",
            "is_planted_reference",
            "matched_atom_l2rho_sq",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        outputs = []
        for ckpt_path, cfg_path in discover_checkpoints(case, role):
            learned_atoms, _ckpt, cfg = get_learned_atoms(ckpt_path, cfg_path, device)
            cost, true_to_learned = match_atoms(learned_atoms, true_atoms)
            label = checkpoint_label(ckpt_path, cfg)
            stem = safe_stem(f"{ckpt_path.parent.name}_{ckpt_path.stem}")
            out_png = case_out / f"{stem}_nearest_neighbors.png"
            make_figure(
                case.label,
                label,
                learned_atoms,
                true_atoms,
                digits,
                cost,
                true_to_learned,
                pool,
                top_k,
                out_png,
                writer,
            )
            outputs.append(out_png)
            print(f"saved {resolve(out_png).relative_to(REPO_ROOT)}")
    return outputs, csv_path


def parse_case(text):
    parts = text.split(":", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--case must be label:results_dir:data_dir:ot_dir"
        )
    return Case(parts[0], Path(parts[1]), Path(parts[2]), Path(parts[3]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["selected"], default="selected")
    parser.add_argument("--case", action="append", type=parse_case,
                        help="label:results_dir:data_dir:ot_dir; overrides preset")
    parser.add_argument("--output_dir", type=Path,
                        default=Path("mnist/results/convex_combo_atom_neighbors_selected"))
    parser.add_argument("--role", choices=["best", "last", "all"], default="best")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--all_mnist_neighbors", action="store_true",
                        help="Search all digit_* maps in ot_dir instead of only combo digits.")
    args = parser.parse_args()

    cases = args.case if args.case else SELECTED_CASES
    out_root = resolve(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    manifest_rows = []
    for case in cases:
        outputs, csv_path = run_case(
            case, out_root, args.role, args.top_k, device,
            restrict_to_combo_digits=not args.all_mnist_neighbors,
        )
        manifest_rows.append({
            "case": case.label,
            "n_figures": len(outputs),
            "csv": str(csv_path.relative_to(REPO_ROOT)),
            "figures": ";".join(str(p.relative_to(REPO_ROOT)) for p in outputs),
        })

    manifest = out_root / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "n_figures", "csv", "figures"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    gallery = out_root / "index.html"
    write_gallery(gallery, manifest_rows)
    print(f"saved {manifest.relative_to(REPO_ROOT)}")
    print(f"saved {gallery.relative_to(REPO_ROOT)}")


def write_gallery(path, manifest_rows):
    path = Path(path)

    def rel_from_gallery(target):
        return os.path.relpath(resolve(target), start=path.parent)

    blocks = []
    for row in manifest_rows:
        figures = [f for f in row["figures"].split(";") if f]
        imgs = "\n".join(
            f'<figure><img src="{html.escape(rel_from_gallery(fig))}" '
            f'alt="{html.escape(Path(fig).name)}"><figcaption>{html.escape(Path(fig).name)}</figcaption></figure>'
            for fig in figures
        )
        csv_rel = rel_from_gallery(row["csv"])
        blocks.append(
            f"<section><h2>{html.escape(row['case'])}</h2>"
            f"<p><code>{html.escape(str(csv_rel))}</code></p>{imgs}</section>"
        )
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Convex-combo atom nearest neighbors</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #1f2933; }}
    h1 {{ font-size: 22px; }}
    h2 {{ font-size: 16px; margin-top: 28px; }}
    section {{ border-top: 1px solid #d6dde5; padding-top: 14px; }}
    figure {{ margin: 16px 0 28px; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #d6dde5; }}
    figcaption {{ font-size: 12px; color: #52616f; margin-top: 6px; }}
    code {{ font-size: 12px; }}
  </style>
</head>
<body>
  <h1>Convex-combo atom nearest neighbors</h1>
  <p>Rows are planted atoms matched to learned atoms. Neighbor columns are sorted by L2(rho) distance. A dark outline marks the appended planted-atom reference.</p>
  {''.join(blocks)}
</body>
</html>
"""
    path.write_text(doc)


if __name__ == "__main__":
    main()
