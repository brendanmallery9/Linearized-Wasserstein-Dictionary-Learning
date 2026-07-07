"""
Generate a synthetic dataset of convex combinations of MNIST OT maps.

1. Loads (or computes, via prepare_mnist_ot.py) the MNIST OT dataset.
2. Picks one sample from each of the 10 digit classes -> 10 "base" maps.
3. Generates N convex combinations: for each sample, pick `subset_size` distinct
   digits uniformly, draw Dirichlet(1,...,1) weights on the simplex, and form
   the weighted sum of the chosen base maps.
4. Saves maps + per-sample coefficient vectors (length 10, zeros outside the
   chosen subset). Row order matches between the two tensors.

Output layout:
    <output_dir>/
        base_measure.pt                  # (n, d) copy of MNIST OT base measure
        maps/maps.pt                     # (N, n, d) mixed maps
        coefficients/coefficients.pt     # (N, 10) mixing weights, row-aligned
        metadata.json

Usage:
    python wdl_repo/pointcloud/pipeline/prepare_convex_combos.py \
        --output_dir wdl_repo/datasets/convex_combos \
        --mnist_ot_dir wdl_repo/datasets/mnist_ot \
        --n_samples 10000 --subset_size 3
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def ensure_mnist_ot(mnist_ot_dir, python_exe):
    """Run prepare_mnist_ot.py if any required artifact is missing."""
    mnist_ot_dir = Path(mnist_ot_dir)
    digits_ok = (mnist_ot_dir / "base_measure.pt").exists() and all(
        (mnist_ot_dir / f"digit_{d}" / "mappings.pt").exists() for d in range(10)
    )
    if digits_ok:
        print(f"Found existing MNIST OT dataset at {mnist_ot_dir}")
        return
    print(f"MNIST OT dataset incomplete at {mnist_ot_dir}; running prepare_mnist_ot.py")
    prep_script = REPO_ROOT / "mnist" / "pipeline" / "prepare_mnist_ot.py"
    cmd = [python_exe, str(prep_script), "--output_dir", str(mnist_ot_dir)]
    subprocess.run(cmd, check=True)


def load_base_digit_maps(mnist_ot_dir, sample_index, digits=None):
    """Load the base measure and one map per chosen digit class.

    Args:
        mnist_ot_dir: directory with base_measure.pt and digit_<d>/mappings.pt
        sample_index: which sample to take from each digit_<d>/mappings.pt
        digits:       list of digit classes to use as base maps. None = all 10.

    Returns:
        base:      (n, d) base measure
        base_maps: (n_classes, n, d) one map per chosen digit (taken at `sample_index`)
        digits:    list[int] the digit classes used, aligned with base_maps rows
    """
    mnist_ot_dir = Path(mnist_ot_dir)
    base = torch.load(mnist_ot_dir / "base_measure.pt", map_location="cpu").float()
    if digits is None:
        digits = list(range(10))
    per_digit = []
    for d in digits:
        m = torch.load(mnist_ot_dir / f"digit_{d}" / "mappings.pt", map_location="cpu")
        if sample_index >= m.shape[0]:
            raise IndexError(
                f"digit_{d}/mappings.pt has {m.shape[0]} samples; "
                f"requested index {sample_index}"
            )
        per_digit.append(m[sample_index].float())
    base_maps = torch.stack(per_digit, dim=0)
    return base, base_maps, list(digits)


def generate_convex_combos(base_maps, n_samples, subset_size, seed):
    """Sample N Dirichlet-weighted mixtures of distinct base maps.

    Returns:
        mixed:  (N, n, d)
        coeffs: (N, n_classes)  zeros outside the chosen subset
    """
    rng = np.random.default_rng(seed)
    n_classes, n_pts, dim = base_maps.shape
    if subset_size > n_classes:
        raise ValueError(f"subset_size={subset_size} > n_classes={n_classes}")
    if subset_size < 1:
        raise ValueError("subset_size must be >= 1")

    mixed = torch.empty((n_samples, n_pts, dim), dtype=base_maps.dtype)
    coeffs = torch.zeros((n_samples, n_classes), dtype=torch.float32)

    for i in range(n_samples):
        chosen = rng.choice(n_classes, size=subset_size, replace=False)
        w = rng.dirichlet(np.ones(subset_size)).astype(np.float32)
        w_t = torch.from_numpy(w)
        mixed[i] = (w_t[:, None, None] * base_maps[chosen]).sum(dim=0)
        coeffs[i, chosen] = w_t
    return mixed, coeffs


def main():
    parser = argparse.ArgumentParser(
        description="Generate convex-combination dataset from MNIST OT maps."
    )
    parser.add_argument("--output_dir", type=str,
                        default=str(REPO_ROOT / "datasets" / "convex_combos"))
    parser.add_argument("--mnist_ot_dir", type=str,
                        default=str(REPO_ROOT / "datasets" / "mnist_ot"))
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--subset_size", type=int, default=3,
                        help="Number of distinct digits per mixture (default 3).")
    parser.add_argument("--digits", type=int, nargs="+", default=None,
                        help="Digit classes to use as base maps, e.g. --digits 0 3 4. "
                             "Default: all 10. Coefficient columns align to this order.")
    parser.add_argument("--digit_sample_index", type=int, default=0,
                        help="Which sample to take from each digit_<d>/mappings.pt.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python", type=str, default=sys.executable,
                        help="Python used to run prepare_mnist_ot.py if needed.")
    args = parser.parse_args()

    t0 = time.time()
    ensure_mnist_ot(args.mnist_ot_dir, args.python)

    base, base_maps, digits = load_base_digit_maps(
        args.mnist_ot_dir, args.digit_sample_index, digits=args.digits,
    )
    print(f"Base measure shape: {tuple(base.shape)}")
    print(f"Base digit maps:    {tuple(base_maps.shape)} "
          f"(digits={digits}, sample_index={args.digit_sample_index})")

    mixed, coeffs = generate_convex_combos(
        base_maps, args.n_samples, args.subset_size, args.seed,
    )
    print(f"Generated {mixed.shape[0]} mixtures (subset_size={args.subset_size})")

    out = Path(args.output_dir)
    maps_dir = out / "maps"
    coeffs_dir = out / "coefficients"
    maps_dir.mkdir(parents=True, exist_ok=True)
    coeffs_dir.mkdir(parents=True, exist_ok=True)

    torch.save(base, out / "base_measure.pt")
    torch.save(mixed, maps_dir / "maps.pt")
    torch.save(coeffs, coeffs_dir / "coefficients.pt")
    # The true generating atoms (one per chosen digit), row-aligned with the
    # coefficient columns. Saved so evaluation can plot/compare against them.
    torch.save(base_maps, out / "base_maps.pt")

    metadata = {
        "format": "convex_combo_maps",
        "version": 1,
        "source": "mnist_ot",
        "mnist_ot_dir": str(Path(args.mnist_ot_dir).resolve()),
        "digit_sample_index": args.digit_sample_index,
        "digits": [int(d) for d in digits],
        "n_samples": int(args.n_samples),
        "subset_size": int(args.subset_size),
        "n_classes": int(base_maps.shape[0]),
        "support_size": int(base.shape[0]),
        "dim": int(base.shape[1]),
        "seed": args.seed,
        "elapsed_seconds": time.time() - t0,
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print()
    print(f"Saved:")
    print(f"  base_measure: {out / 'base_measure.pt'}  shape={tuple(base.shape)}")
    print(f"  base_maps:    {out / 'base_maps.pt'}  shape={tuple(base_maps.shape)}  digits={digits}")
    print(f"  maps:         {maps_dir / 'maps.pt'}  shape={tuple(mixed.shape)}")
    print(f"  coefficients: {coeffs_dir / 'coefficients.pt'}  shape={tuple(coeffs.shape)}")
    print(f"  metadata:     {out / 'metadata.json'}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
