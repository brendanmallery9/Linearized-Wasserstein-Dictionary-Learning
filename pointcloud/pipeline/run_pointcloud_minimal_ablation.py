"""
Minimal point-cloud ablation: LWDL-EOT vs. PCA vs. conventional sparse coding.

All three unsupervised representations are compared on the *same* point-cloud OT
dataset and the *same* train/test split:

    PCA            -- dense linear PCA on flattened displacement maps.
    Sparse Coding  -- unconstrained dictionary learning on the same flattened maps.
    LWDL-EOT       -- the existing trained transport-map SAE, re-encoded here.

This is an ablation, not a SOTA point-cloud benchmark: the question is whether
Wasserstein-constrained sparse atoms buy a useful tradeoff relative to dense
linear PCA and unconstrained sparse dictionaries.

Outputs (written under --output_dir, default pointcloud/results/minimal_ablation):
    split_indices.json     -- the shared train/test split (index lists).
    ablation_metrics.json  -- full per-method metric rows.
    ablation_table.csv      -- compact comparison table.
    ablation_table.tex      -- LaTeX version of the same table.

No existing result directory is modified.

Usage:
    python pointcloud/pipeline/run_pointcloud_minimal_ablation.py \
        --data_dir datasets/modelnet10_6cls_ot \
        --lwdl_results_dir pointcloud/results/modelnet10_6cls_uniform
"""

import argparse
import csv
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

from pointcloud.pipeline.pointcloud_data import (
    load_raw_clouds,
    load_with_labels,
    make_split_indices,
)
from pointcloud.pipeline.pointcloud_ablation_utils import (
    chamfer_loss,
    flatten_displacement_maps,
    linear_probe,
    map_l2_loss,
    run_pca_baseline,
    run_sparse_coding_baseline,
    sparsity_stats,
    unflatten_displacement_maps,
    wasserstein_loss,
)
from pointcloud.pipeline.evaluate_pointcloud_lwdl_for_ablation import evaluate_lwdl


DEFAULT_DATA_DIR = str(REPO_ROOT / "datasets" / "modelnet10_6cls_ot")
DEFAULT_LWDL_RESULTS_DIR = str(
    REPO_ROOT / "pointcloud" / "results" / "modelnet10_6cls_uniform")
DEFAULT_OUTPUT_DIR = str(
    REPO_ROOT / "pointcloud" / "results" / "minimal_ablation")


# ============================================================
# Reading LWDL defaults (m, target_l0) from the trained run
# ============================================================

def read_lwdl_m(results_dir, fallback=20):
    cfg = Path(results_dir) / "config.json"
    if cfg.exists():
        try:
            with open(cfg) as f:
                val = json.load(f).get("m")
            if val is not None:
                return int(val)
        except Exception:
            pass
    return int(fallback)


def read_lwdl_target_l0(results_dir, fallback=10):
    """Read the trained LWDL mean active coefficient count (train split)."""
    metrics = Path(results_dir) / "metrics.json"
    if metrics.exists():
        try:
            with open(metrics) as f:
                rows = json.load(f)
            if rows:
                r = rows[0]
                for key in ("train_mean_active", "test_mean_active",
                            "best_train_mean_active", "best_test_mean_active"):
                    if r.get(key) is not None:
                        return float(r[key])
        except Exception:
            pass
    return float(fallback)


# ============================================================
# Shared split (supports optional --max_samples for smoke runs)
# ============================================================

def _subsampled_split(labels, max_samples, test_fraction, seed):
    """Stratified subsample of `max_samples` items, then a stratified split."""
    from sklearn.model_selection import train_test_split

    labels_list = [int(v) for v in labels.tolist()]
    all_idx = list(range(len(labels_list)))
    max_samples = min(int(max_samples), len(all_idx))
    pool, _ = train_test_split(
        all_idx, train_size=max_samples, random_state=seed,
        shuffle=True, stratify=labels_list,
    )
    pool_labels = [labels_list[i] for i in pool]
    train_idx, test_idx = train_test_split(
        pool, test_size=test_fraction, random_state=seed,
        shuffle=True, stratify=pool_labels,
    )
    return sorted(int(i) for i in train_idx), sorted(int(i) for i in test_idx)


def build_or_load_split(labels, split_path, test_fraction, seed, max_samples):
    split_path = Path(split_path)
    if split_path.exists():
        return make_split_indices(labels, split_path=split_path)
    if max_samples is not None and max_samples > 0:
        train_idx, test_idx = _subsampled_split(
            labels, max_samples, test_fraction, seed)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with open(split_path, "w") as f:
            json.dump({
                "seed": int(seed),
                "test_fraction": float(test_fraction),
                "stratified": True,
                "max_samples": int(max_samples),
                "train_indices": train_idx,
                "test_indices": test_idx,
            }, f)
        print(f"Created subsampled split (max_samples={max_samples}): "
              f"{len(train_idx)} train, {len(test_idx)} test -> {split_path}")
        return train_idx, test_idx
    return make_split_indices(
        labels, test_fraction=test_fraction, seed=seed,
        split_path=split_path, stratified=True)


# ============================================================
# Uniform metric row for one method
# ============================================================

def compute_metric_row(method_name, *, train_codes, test_codes,
                       train_recon, test_recon,
                       wass_target_train, wass_target_test,
                       train_labels, test_labels, runtime, dense_code,
                       map_l2_ref_train=None, map_l2_ref_test=None,
                       mean_l0_override=None, wasserstein_metric="sqeuclidean",
                       max_wasserstein_samples=None, seed=42, extra=None):
    """
    Uniform metric row for one method.

    Reconstruction is scored in two independent references:
      * map-space L2 against the OT map T_i (`map_l2_ref_*`) -- LOT-only, needs a
        point correspondence, so it is None for correspondence-free methods.
      * Wasserstein / Chamfer against the raw target cloud mu_i
        (`wass_target_*`) -- the common ground-truth yardstick for every method.
    """
    # Map-space L2 only where a corresponding reference map is provided.
    if map_l2_ref_train is not None and map_l2_ref_test is not None:
        train_map_l2 = map_l2_loss(map_l2_ref_train, train_recon)
        test_map_l2 = map_l2_loss(map_l2_ref_test, test_recon)
    else:
        train_map_l2 = None
        test_map_l2 = None

    wass, n_wass = wasserstein_loss(
        wass_target_test, test_recon, metric=wasserstein_metric,
        max_samples=max_wasserstein_samples)
    chamfer = chamfer_loss(
        wass_target_test, test_recon, max_samples=max_wasserstein_samples)
    spars = sparsity_stats(test_codes)
    probe = linear_probe(train_codes, train_labels, test_codes, test_labels,
                         seed=seed)

    mean_l0 = float(mean_l0_override) if mean_l0_override is not None else spars["mean_l0"]

    row = {
        "method": method_name,
        "train_map_l2": train_map_l2,
        "test_map_l2": test_map_l2,
        "test_wasserstein": wass,
        "wasserstein_metric": wasserstein_metric,
        "n_wasserstein_samples": n_wass,
        "test_chamfer": chamfer,
        "mean_l0": mean_l0,
        "mean_l1": spars["mean_l1"],
        "frac_nonzero": spars["frac_nonzero"],
        "dense_code": bool(dense_code),
        "accuracy": probe["accuracy"],
        "macro_f1": probe["macro_f1"],
        "runtime_seconds": float(runtime),
    }
    if extra:
        row.update(extra)
    return row


# ============================================================
# Table writers
# ============================================================

CSV_COLUMNS = [
    "method", "test_map_l2", "test_wasserstein", "test_chamfer",
    "mean_l0", "dense_code", "accuracy", "macro_f1", "runtime_seconds",
]


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in CSV_COLUMNS})
    print(f"Wrote CSV table: {path}")


def write_latex(rows, path, wasserstein_metric):
    def fmt(x, nd=6):
        if x is None:
            return "--"
        if isinstance(x, float):
            return f"{x:.{nd}g}"
        return str(x)

    wass_label = ("$W_2^2$" if wasserstein_metric == "sqeuclidean" else "$W_1$")
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        (r"Method & Map $L_2$ $\downarrow$ & " + wass_label + r" $\downarrow$ & "
         r"Chamfer $\downarrow$ & Mean $L_0$ $\downarrow$ & Acc $\uparrow$ & "
         r"Macro-F1 $\uparrow$ & Time (s) \\"),
        r"\midrule",
    ]
    for r in rows:
        l0 = (f"{r['mean_l0']:.3g} (dense)" if r.get("dense_code")
              else f"{r['mean_l0']:.3g}")
        lines.append(
            f"{r['method']} & {fmt(r['test_map_l2'])} & "
            f"{fmt(r['test_wasserstein'])} & {fmt(r['test_chamfer'])} & "
            f"{l0} & {fmt(r['accuracy'], 4)} & {fmt(r['macro_f1'], 4)} & "
            f"{r['runtime_seconds']:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote LaTeX table: {path}")


def print_table(rows):
    header = ["Method", "Map L2", "Wass", "Chamfer", "Mean L0",
              "Acc", "Macro-F1", "Time(s)"]
    widths = [14, 11, 11, 11, 12, 7, 9, 8]
    line = " | ".join(h.ljust(w) for h, w in zip(header, widths))
    print("\n" + line)
    print("-" * len(line))
    for r in rows:
        l0 = f"{r['mean_l0']:.2f}" + (" (d)" if r.get("dense_code") else "")
        map_l2 = ("--" if r.get("test_map_l2") is None
                  else f"{r['test_map_l2']:.4e}")
        cells = [
            r["method"],
            map_l2,
            f"{r['test_wasserstein']:.4e}",
            f"{r['test_chamfer']:.4e}",
            l0,
            f"{r['accuracy']:.3f}",
            f"{r['macro_f1']:.3f}",
            f"{r['runtime_seconds']:.1f}",
        ]
        print(" | ".join(str(c).ljust(w) for c, w in zip(cells, widths)))
    print()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--lwdl_results_dir", type=str,
                        default=DEFAULT_LWDL_RESULTS_DIR)
    parser.add_argument("--lwdl_checkpoint", type=str, default=None,
                        help="Explicit LWDL checkpoint (else auto-discovered).")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--m", type=int, default=None,
                        help="Dictionary width for PCA/sparse coding "
                             "(default: read from LWDL config, fallback 20).")
    parser.add_argument("--target_l0", type=float, default=None,
                        help="Target mean L0 for sparse coding "
                             "(default: read from LWDL metrics, fallback 10).")
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--wasserstein_metric", type=str, default="sqeuclidean",
                        choices=["sqeuclidean", "euclidean"])
    parser.add_argument("--max_wasserstein_samples", type=int, default=None,
                        help="Cap the number of test clouds used for Wasserstein "
                             "/ Chamfer (default: all).")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional cap on total dataset samples (stratified) "
                             "for fast smoke runs.")
    parser.add_argument("--sparse_max_iter", type=int, default=200,
                        help="MiniBatchDictionaryLearning max_iter.")
    parser.add_argument("--skip_lwdl", action="store_true",
                        help="Skip the LWDL row (e.g. if no checkpoint yet).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / "split_indices.json"

    m = args.m if args.m is not None else read_lwdl_m(args.lwdl_results_dir)
    target_l0 = (args.target_l0 if args.target_l0 is not None
                 else read_lwdl_target_l0(args.lwdl_results_dir))
    if args.m is not None and target_l0 is not None and target_l0 > m:
        target_l0 = min(target_l0, m)
    print(f"Dictionary width m={m}; sparse-coding target L0={target_l0:.3f}")

    # ---- Load data + build the shared split -------------------------------
    print("\nLoading labeled maps...")
    X, maps, labels, classes = load_with_labels(args.data_dir)
    X = X.float()
    maps = maps.float()

    # Raw target clouds mu_i are the common ground-truth for Wasserstein/Chamfer.
    raw_clouds = load_raw_clouds(args.data_dir, classes)
    if raw_clouds is None:
        print("  WARNING: raw target clouds unavailable; falling back to the OT "
              "maps T_i as the Wasserstein/Chamfer target (regenerate the "
              "dataset with the updated prepare script for the mu_i yardstick).")
        wass_source = maps
        wass_target_kind = "ot_map_Ti"
    else:
        wass_source = raw_clouds.float()
        wass_target_kind = "raw_cloud_mui"

    train_idx, test_idx = build_or_load_split(
        labels, split_path, args.test_fraction, args.seed, args.max_samples)
    train_idx_t = torch.as_tensor(train_idx, dtype=torch.long)
    test_idx_t = torch.as_tensor(test_idx, dtype=torch.long)

    train_maps = maps[train_idx_t]
    test_maps = maps[test_idx_t]
    train_labels = labels[train_idx_t]
    test_labels = labels[test_idx_t]
    wass_target_train = wass_source[train_idx_t]
    wass_target_test = wass_source[test_idx_t]

    # ---- Flattened displacement representation (train fit only) -----------
    flat = flatten_displacement_maps(X, maps)
    train_flat = flat[train_idx_t].numpy()
    test_flat = flat[test_idx_t].numpy()

    rows = []
    # Shared kwargs. LOT methods (PCA/sparse/LWDL) also get the OT map T_i as the
    # map-space L2 reference; the raw cloud mu_i is the Wasserstein/Chamfer target.
    metric_kwargs = dict(
        wass_target_train=wass_target_train, wass_target_test=wass_target_test,
        train_labels=train_labels, test_labels=test_labels,
        wasserstein_metric=args.wasserstein_metric,
        max_wasserstein_samples=args.max_wasserstein_samples, seed=args.seed,
    )
    lot_l2_kwargs = dict(map_l2_ref_train=train_maps, map_l2_ref_test=test_maps)

    # ---- PCA baseline -----------------------------------------------------
    print("\n=== PCA baseline ===")
    pca = run_pca_baseline(train_flat, test_flat, m=m, seed=args.seed)
    pca_train_recon = unflatten_displacement_maps(X, torch.as_tensor(
        pca["train_recon_flat"], dtype=torch.float32))
    pca_test_recon = unflatten_displacement_maps(X, torch.as_tensor(
        pca["test_recon_flat"], dtype=torch.float32))
    rows.append(compute_metric_row(
        "PCA",
        train_codes=pca["train_codes"], test_codes=pca["test_codes"],
        train_recon=pca_train_recon, test_recon=pca_test_recon,
        runtime=pca["runtime_seconds"], dense_code=True,
        mean_l0_override=pca["n_components"],
        extra={"explained_variance_ratio_sum": pca["explained_variance_ratio_sum"],
               "n_components": pca["n_components"]},
        **lot_l2_kwargs, **metric_kwargs))

    # ---- Sparse coding baseline ------------------------------------------
    print("\n=== Sparse coding baseline ===")
    sc = run_sparse_coding_baseline(
        train_flat, test_flat, m=m, target_l0=target_l0, seed=args.seed,
        max_iter=args.sparse_max_iter, batch_size=args.batch_size)
    sc_train_recon = unflatten_displacement_maps(X, torch.as_tensor(
        sc["train_recon_flat"], dtype=torch.float32))
    sc_test_recon = unflatten_displacement_maps(X, torch.as_tensor(
        sc["test_recon_flat"], dtype=torch.float32))
    rows.append(compute_metric_row(
        "Sparse Coding",
        train_codes=sc["train_codes"], test_codes=sc["test_codes"],
        train_recon=sc_train_recon, test_recon=sc_test_recon,
        runtime=sc["runtime_seconds"], dense_code=False,
        extra={"chosen_alpha": sc["chosen_alpha"], "target_l0": sc["target_l0"],
               "train_mean_l0": sc["train_mean_l0"], "alpha_grid": sc["alpha_grid"],
               "n_components": sc["n_components"]},
        **lot_l2_kwargs, **metric_kwargs))

    # ---- LWDL-EOT ---------------------------------------------------------
    if not args.skip_lwdl:
        print("\n=== LWDL-EOT ===")
        lwdl = evaluate_lwdl(
            data_dir=args.data_dir, results_dir=args.lwdl_results_dir,
            split_path=str(split_path), checkpoint=args.lwdl_checkpoint,
            batch_size=args.batch_size, device=args.device, classes=classes,
            test_fraction=args.test_fraction, seed=args.seed)
        rows.append(compute_metric_row(
            "LWDL-EOT",
            train_codes=lwdl["train_codes"], test_codes=lwdl["test_codes"],
            train_recon=lwdl["train_recon_maps"],
            test_recon=lwdl["test_recon_maps"],
            runtime=lwdl["runtime_seconds"], dense_code=False,
            extra={"lwdl_method": lwdl["method"], "checkpoint": lwdl["checkpoint"]},
            **lot_l2_kwargs, **metric_kwargs))
    else:
        print("\n=== LWDL-EOT skipped (--skip_lwdl) ===")

    # ---- Persist ----------------------------------------------------------
    meta = {
        "data_dir": str(args.data_dir),
        "lwdl_results_dir": str(args.lwdl_results_dir),
        "m": m,
        "target_l0": target_l0,
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "wasserstein_metric": args.wasserstein_metric,
        "wasserstein_target": wass_target_kind,
        "max_wasserstein_samples": args.max_wasserstein_samples,
        "max_samples": args.max_samples,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "classes": classes,
        "split_path": str(split_path),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    metrics_path = output_dir / "ablation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=2)
    print(f"\nWrote metrics JSON: {metrics_path}")

    write_csv(rows, output_dir / "ablation_table.csv")
    write_latex(rows, output_dir / "ablation_table.tex", args.wasserstein_metric)
    print_table(rows)


if __name__ == "__main__":
    main()
