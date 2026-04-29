"""
Step 1: Download MNIST, convert each digit image to a probability measure on [0,1]^2,
compute OT maps from a FIXED uniform base measure to each digit measure, and save results.

Output structure:
    <output_dir>/
        base_measure.pt          # The fixed uniform base measure points (supp_size, 2)
        digit_0/
            mappings.pt          # Stacked OT maps for digit 0: (num_samples, supp_size, 2)
        digit_1/
            mappings.pt
        ...
        digit_9/
            mappings.pt

Usage:
    python mnist/pipeline/prepare_mnist_ot.py --output_dir datasets/mnist_ot --base_supp_size 400 --max_per_digit 1000
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from OT_utils import measure, image_to_empirical, wass_map


def make_mixture_base_measure(mnist, supp_size, n_components, noise_std, rng):
    """
    Build a base measure by sampling from a mixture of `n_components` random
    MNIST digit images (drawn from all classes).

    Each component contributes roughly supp_size // n_components points.
    Gaussian noise with std `noise_std` is added to all sampled points.

    Returns (points, masses, component_indices) where component_indices
    lists the MNIST dataset indices of the images used.
    """
    n_images = len(mnist)
    chosen_indices = rng.choice(n_images, size=n_components, replace=False)

    points_list = []
    points_per_component = supp_size // n_components
    remainder = supp_size - points_per_component * n_components

    for j, idx in enumerate(chosen_indices):
        image = np.array(mnist[idx][0], dtype='float64')  # (28, 28)
        image = np.maximum(image, 0)
        total = image.sum()
        if total == 0:
            # Degenerate all-black image: fall back to uniform
            image = np.ones_like(image)
            total = image.sum()
        image = image / total

        h, w = image.shape
        rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        coords = np.stack([rows.ravel() / h, cols.ravel() / w], axis=1)  # (784, 2)
        probs = image.ravel()

        # Give the first `remainder` components one extra point
        n_pts = points_per_component + (1 if j < remainder else 0)
        sample_indices = rng.choice(len(probs), size=n_pts, replace=True, p=probs)
        points_list.append(coords[sample_indices].astype('float32'))

    points = np.concatenate(points_list, axis=0)  # (supp_size, 2)
    points = points + rng.normal(0, noise_std, size=points.shape).astype('float32')

    masses = np.ones(supp_size, dtype='float64') / supp_size
    labels_used = [int(mnist[idx][1]) for idx in chosen_indices]
    return points, masses, chosen_indices.tolist(), labels_used


def make_digit_base_measure(mnist, digit, supp_size, noise_std, rng):
    """
    Build a base measure by sampling from a random MNIST digit image.

    1. Pick a random image of the given digit class.
    2. Normalize pixel intensities to a probability distribution.
    3. Sample `supp_size` points from that distribution (with replacement),
       then add isotropic Gaussian noise with std `noise_std`.

    Returns (points, masses) with uniform masses (like the uniform base).
    """
    # Find all indices for this digit and pick one at random
    digit_indices = [i for i, (_, label) in enumerate(mnist) if label == digit]
    idx = rng.choice(digit_indices)
    image = np.array(mnist[idx][0], dtype='float64')  # (28, 28)

    # Normalize to a probability distribution
    image = np.maximum(image, 0)
    image = image / image.sum()

    # Build (row, col) grid of pixel coordinates normalized to [0,1]^2
    h, w = image.shape
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = np.stack([rows.ravel() / h, cols.ravel() / w], axis=1)  # (784, 2)
    probs = image.ravel()  # (784,)

    # Sample support points from the pixel distribution
    sample_indices = rng.choice(len(probs), size=supp_size, replace=True, p=probs)
    points = coords[sample_indices].astype('float32')  # (supp_size, 2)

    # Add Gaussian noise
    points = points + rng.normal(0, noise_std, size=points.shape).astype('float32')

    masses = np.ones(supp_size, dtype='float64') / supp_size
    return points, masses, idx


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare MNIST OT maps.")
    parser.add_argument('--output_dir', type=str, default='datasets/mnist_ot',
                        help='Root directory for saving results')
    parser.add_argument('--base_supp_size', type=int, default=400,
                        help='Number of points in the uniform base measure')
    parser.add_argument('--max_per_digit', type=int, default=2000,
                        help='Max number of images to process per digit')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--base_mode', type=str, default='uniform',
                        choices=['uniform', 'digit', 'mixture'],
                        help="'uniform': random uniform base; "
                             "'digit': sample from a single MNIST digit image; "
                             "'mixture': sample from a mixture of N random digit images")
    parser.add_argument('--base_digit', type=int, default=0,
                        help='Which digit class to use as base (only for --base_mode digit)')
    parser.add_argument('--base_noise_std', type=float, default=0.02,
                        help='Std of Gaussian noise added to sampled base points '
                             '(only for --base_mode digit or mixture)')
    parser.add_argument('--base_n_components', type=int, default=100,
                        help='Number of digit images to mix '
                             '(only for --base_mode mixture)')
    parser.add_argument('--progress_every', type=int, default=10,
                        help='Print progress every N images per digit')
    return parser.parse_args()


def main():
    args = parse_args()
    t_start = time.time()
    np.random.seed(args.seed)  # global seed for any library calls that use it
    digit_counts = {}
    base_details = {}

    # --- Download MNIST ---
    print("Downloading/loading MNIST...", flush=True)
    mnist = datasets.MNIST(root='./mnist_raw', train=True, download=True)
    print(f"MNIST loaded: {len(mnist)} training images", flush=True)

    # --- Create base measure ---
    rng = np.random.RandomState(args.seed)

    if args.base_mode == 'mixture':
        base_points, base_masses, comp_indices, comp_labels = make_mixture_base_measure(
            mnist, args.base_supp_size, args.base_n_components, args.base_noise_std, rng,
        )
        base_details = {
            "component_indices": comp_indices,
            "component_labels": comp_labels,
        }
        from collections import Counter
        label_counts = Counter(comp_labels)
        print(f"Base measure: mixture of {args.base_n_components} digit images, "
              f"{args.base_supp_size} total points, noise_std={args.base_noise_std}",
              flush=True)
        print(f"  Digit class distribution: {dict(sorted(label_counts.items()))}",
              flush=True)
    elif args.base_mode == 'digit':
        base_points, base_masses, src_idx = make_digit_base_measure(
            mnist, args.base_digit, args.base_supp_size, args.base_noise_std, rng,
        )
        base_details = {"source_index": int(src_idx)}
        print(f"Base measure: sampled {args.base_supp_size} points from digit "
              f"{args.base_digit} (image index {src_idx}), noise_std={args.base_noise_std}",
              flush=True)
    else:
        base_points = rng.rand(args.base_supp_size, 2).astype('float32')
        base_masses = np.ones(args.base_supp_size, dtype='float64') / args.base_supp_size
        print(f"Base measure: {args.base_supp_size} uniform random points on [0,1]^2",
              flush=True)

    base = measure(base_points, base_masses)

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(torch.tensor(base_points), os.path.join(args.output_dir, 'base_measure.pt'))

    # --- Group images by digit ---
    digit_indices = {d: [] for d in range(10)}
    for idx in range(len(mnist)):
        _, label = mnist[idx]
        digit_indices[label].append(idx)

    # --- Compute OT maps per digit ---
    for digit in range(10):
        indices = digit_indices[digit][:args.max_per_digit]
        digit_counts[str(digit)] = len(indices)
        digit_dir = os.path.join(args.output_dir, f'digit_{digit}')
        os.makedirs(digit_dir, exist_ok=True)

        digit_start = time.time()
        print(
            f"Digit {digit}: computing {len(indices)} OT maps "
            f"(support={args.base_supp_size})",
            flush=True,
        )
        maps_list = []
        for i, idx in enumerate(indices):
            if i == 0 or (args.progress_every > 0 and i % args.progress_every == 0):
                elapsed = time.time() - digit_start
                rate = i / elapsed if elapsed > 0 else 0.0
                remaining = (len(indices) - i) / rate if rate > 0 else float("nan")
                print(
                    f"  Digit {digit}: {i}/{len(indices)} "
                    f"elapsed={elapsed:.1f}s eta={remaining:.1f}s",
                    flush=True,
                )
            image = np.array(mnist[idx][0])  # PIL Image -> numpy
            target = image_to_empirical(image)
            ot_map = wass_map(base, target, 'emd')
            maps_list.append(torch.tensor(ot_map, dtype=torch.float32))

        mapping_tensor = torch.stack(maps_list)  # (num_samples, supp_size, 2)
        torch.save(mapping_tensor, os.path.join(digit_dir, 'mappings.pt'))
        digit_elapsed = time.time() - digit_start
        total_elapsed = time.time() - t_start
        print(
            f"Digit {digit}: saved {len(maps_list)} OT maps, "
            f"shape {mapping_tensor.shape}, digit_elapsed={digit_elapsed:.1f}s, "
            f"total_elapsed={total_elapsed:.1f}s",
            flush=True,
        )

    total_elapsed = time.time() - t_start
    metadata = {
        "format": "mnist_ot_maps",
        "version": 1,
        "parameters": {
            "base_supp_size": args.base_supp_size,
            "max_per_digit": args.max_per_digit,
            "seed": args.seed,
            "base_mode": args.base_mode,
            "base_digit": args.base_digit,
            "base_noise_std": args.base_noise_std,
            "base_n_components": args.base_n_components,
        },
        "digit_counts": digit_counts,
        "base_details": base_details,
        "elapsed_seconds": total_elapsed,
    }
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata written to: {metadata_path}", flush=True)
    print(f"Done. Total elapsed: {total_elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
