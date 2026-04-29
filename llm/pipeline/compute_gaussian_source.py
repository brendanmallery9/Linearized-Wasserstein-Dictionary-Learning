"""
Preprocessing step for the pile-10k pipeline.

1. Streams all activation .pt files to compute:
   - global mean across all tokens
   - average within-document variance (each doc viewed as a point cloud)
2. Fits a PCA transform (sklearn) on a random sample of activations.
3. Samples n_samples points from N(mean, diag(avg_var)) in the original space.
4. Saves:
   - source.pt  : {"acts": (n_samples, D)} Gaussian samples
   - pca.pt     : fitted sklearn PCA object

Usage:
  python compute_gaussian_source.py \
    --activations_dir /data/.../pile-10k/activations/pile \
    --source_output   /data/.../pile-10k/source.pt \
    --pca_output      /data/.../pile-10k/pca.pt \
    --n_samples 1000 \
    --pca_components 50
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--activations_dir", type=str, required=True,
                   help="Directory of *.pt activation files (each with key 'acts', shape (T, D))")
    p.add_argument("--source_output", type=str, required=True,
                   help="Output path for Gaussian source samples, saved as {'acts': (n_samples, D)}")
    p.add_argument("--pca_output", type=str, required=True,
                   help="Output path for fitted sklearn PCA object")
    p.add_argument("--n_samples", type=int, default=1000,
                   help="Number of Gaussian samples to draw (default: 1000)")
    p.add_argument("--pca_components", type=int, default=80)
    p.add_argument("--pca_sample_rows", type=int, default=10000,
                   help="Rows to subsample for fitting PCA (default: 10000)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def atomic_torch_save(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    act_dir = Path(args.activations_dir)
    pt_files = sorted(act_dir.glob("*.pt"))
    if not pt_files:
        raise ValueError(f"No .pt files found in {act_dir}")
    print(f"Found {len(pt_files)} activation files in {act_dir}")

    # ---- Pass 1: running mean + variance, collect PCA sample rows ----
    total_sum = None
    total_sq_sum = None
    total_tokens = 0
    pca_rows = []          # collect random rows for PCA fitting
    rows_per_file = max(1, args.pca_sample_rows // len(pt_files))

    for i, fp in enumerate(pt_files):
        d = torch.load(fp, map_location="cpu", weights_only=False)
        t = d["acts"].float()          # (T, D)
        T, D = t.shape

        # running statistics
        s1 = t.sum(dim=0)
        s2 = (t * t).sum(dim=0)
        if total_sum is None:
            total_sum = s1
            total_sq_sum = s2
        else:
            total_sum += s1
            total_sq_sum += s2
        total_tokens += T

        # reservoir for PCA
        n_take = min(rows_per_file, T)
        idx = torch.randperm(T)[:n_take]
        pca_rows.append(t[idx].numpy())

        if (i + 1) % 100 == 0 or (i + 1) == len(pt_files):
            print(f"  Pass 1: {i+1}/{len(pt_files)} files  total_tokens={total_tokens}")

    mean = (total_sum / total_tokens).numpy()           # (D,)
    var  = (total_sq_sum / total_tokens).numpy() - mean**2
    var  = np.clip(var, 0.0, None)                      # numerical safety
    std  = np.sqrt(var)
    print(f"Activation dim D={D}, total_tokens={total_tokens}")
    print(f"Mean norm: {np.linalg.norm(mean):.4f}  Avg std: {std.mean():.4f}")

    # ---- Fit PCA ----
    pca_data = np.concatenate(pca_rows, axis=0)
    # shuffle and trim to pca_sample_rows
    shuffle_idx = rng.permutation(len(pca_data))[:args.pca_sample_rows]
    pca_data = pca_data[shuffle_idx]
    print(f"Fitting PCA ({args.pca_components} components) on {len(pca_data)} rows ...")
    pca = PCA(n_components=args.pca_components, random_state=args.seed)
    pca.fit(pca_data)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA explains {explained*100:.1f}% of variance with {args.pca_components} components")

    Path(args.pca_output).parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(pca, args.pca_output)
    print(f"Saved PCA transform → {args.pca_output}")

    # ---- Sample Gaussian source in original space ----
    # N(mean, diag(var)) — independent per dimension
    noise = rng.standard_normal((args.n_samples, D)).astype(np.float32)
    samples = mean + std * noise                        # (n_samples, D)
    samples_t = torch.from_numpy(samples)               # (n_samples, D) float32

    Path(args.source_output).parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save({"acts": samples_t}, args.source_output)
    print(f"Saved Gaussian source ({args.n_samples} samples, D={D}) → {args.source_output}")


if __name__ == "__main__":
    main()
