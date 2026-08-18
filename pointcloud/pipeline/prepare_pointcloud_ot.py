"""
Step 1: Convert each mesh in a small 3D shape dataset to a fixed-size point
cloud, compute OT maps from a FIXED base measure on [0,1]^3 to each cloud,
and save results.

Two datasets are supported via --dataset:
    geometric_shapes  (default, tiny -- good for testing)
        torch_geometric.datasets.GeometricShapes: 8 primitive shape classes
        (cone, cube, cylinder, plane, sphere, torus, ...). ~few MB.
    modelnet10
        Princeton ModelNet10: 10 furniture/object classes. ~451 MB download.

Output structure (mirrors prepare_mnist_ot.py):
    <output_dir>/
        base_measure.pt          # (base_supp_size, 3) uniform points in [0,1]^3
        class_<name>/
            mappings.pt          # (num_samples, base_supp_size, 3) OT maps
        ...
        _raw_examples/
            <name>.pt            # one example cloud per class (for plotting)
        metadata.json

Usage:
    # GeometricShapes (default, fast, lightweight):
    python wdl_repo/pointcloud/pipeline/prepare_pointcloud_ot.py \
        --output_dir wdl_repo/datasets/geomshapes_ot

    # ModelNet10:
    python wdl_repo/pointcloud/pipeline/prepare_pointcloud_ot.py \
        --dataset modelnet10 \
        --output_dir wdl_repo/datasets/modelnet10_ot \
        --max_per_class 100
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from OT_utils import measure, wass_map


MODELNET10_URL = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
MODELNET10_CLASSES = [
    "bathtub", "bed", "chair", "desk", "dresser",
    "monitor", "night_stand", "sofa", "table", "toilet",
]


# ============================================================
# .off parser (ModelNet10) + surface sampling (shared)
# ============================================================

def parse_off(path):
    """
    Minimal parser for the .off mesh format.

    Handles two header variants:
      OFF\\n<n_v> <n_f> <n_e>\\n
    and the malformed-but-common
      OFF<n_v> <n_f> <n_e>\\n
    where the header is glued to the counts on one line.
    """
    with open(path, "r") as f:
        head = f.readline().strip()
        if head == "OFF":
            counts_line = f.readline().strip()
        elif head.startswith("OFF"):
            counts_line = head[3:].strip()
        else:
            raise ValueError(f"Not an OFF file: {path} (first line: {head!r})")

        n_v, n_f, _ = (int(x) for x in counts_line.split()[:3])

        vertices = np.empty((n_v, 3), dtype="float64")
        for i in range(n_v):
            parts = f.readline().split()
            vertices[i] = (float(parts[0]), float(parts[1]), float(parts[2]))

        faces = []
        for _ in range(n_f):
            parts = f.readline().split()
            k = int(parts[0])
            idx = [int(x) for x in parts[1:1 + k]]
            faces.append(idx)

    return vertices, faces


def triangulate(faces):
    """Fan-triangulate any polygonal faces into triangles -> (T, 3) int64."""
    tris = []
    for face in faces:
        if len(face) == 3:
            tris.append(face)
        else:
            for j in range(1, len(face) - 1):
                tris.append([face[0], face[j], face[j + 1]])
    return np.asarray(tris, dtype=np.int64)


def sample_surface(vertices, tris, n_points, rng):
    """
    Sample n_points uniformly on a triangle mesh's surface.

    Triangles are picked with probability proportional to area, then a
    uniform point on each is drawn via the standard barycentric sqrt trick.
    """
    v0 = vertices[tris[:, 0]]
    v1 = vertices[tris[:, 1]]
    v2 = vertices[tris[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total = areas.sum()
    if total <= 0 or not np.isfinite(total):
        idx = rng.choice(len(vertices), size=n_points, replace=True)
        return vertices[idx].astype("float32")

    probs = areas / total
    tri_idx = rng.choice(len(tris), size=n_points, replace=True, p=probs)
    r1 = rng.uniform(0, 1, size=n_points)
    r2 = rng.uniform(0, 1, size=n_points)
    sqrt_r1 = np.sqrt(r1)
    a = 1.0 - sqrt_r1
    b = sqrt_r1 * (1.0 - r2)
    c = sqrt_r1 * r2

    pts = (a[:, None] * v0[tri_idx]
           + b[:, None] * v1[tri_idx]
           + c[:, None] * v2[tri_idx])
    return pts.astype("float32")


def normalize_to_unit_cube(pts, padding=0.05):
    """
    Recenter and rescale a point cloud to fit inside [0,1]^3.

    Center the per-axis bounding box on (0.5, 0.5, 0.5) and scale so the
    longest side is (1 - 2*padding), preserving aspect ratio.
    """
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    extent = (hi - lo).max()
    if extent <= 0:
        return np.full_like(pts, 0.5)
    scale = (1.0 - 2 * padding) / extent
    centered = pts - 0.5 * (lo + hi)
    scaled = centered * scale + 0.5
    return scaled.astype("float32")


# ============================================================
# Dataset adapters: each yields (class_name, vertices, tris) tuples
# ============================================================

def iter_modelnet10(cache_dir, classes, split, max_per_class, zip_path=None):
    """Iterate ModelNet10 meshes after ensuring the dataset is extracted."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    extracted = cache_dir / "ModelNet10"
    if not (extracted.exists() and any(extracted.iterdir())):
        if zip_path is None:
            zip_target = cache_dir / "ModelNet10.zip"
            if not zip_target.exists():
                print(f"Downloading ModelNet10 from {MODELNET10_URL} ...")
                print("(~451 MB; one-time cost.)")
                urllib.request.urlretrieve(MODELNET10_URL, zip_target)
                print(f"Saved to {zip_target}")
            zip_path = zip_target
        else:
            zip_path = Path(zip_path)
            if not zip_path.exists():
                raise FileNotFoundError(f"--zip_path does not exist: {zip_path}")
        print(f"Extracting {zip_path} -> {cache_dir} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(cache_dir)

    classes = classes or MODELNET10_CLASSES
    for cls in classes:
        cls_dir = extracted / cls / split
        if not cls_dir.exists():
            print(f"Warning: no {split} dir for class {cls}", flush=True)
            continue
        mesh_paths = sorted(cls_dir.glob("*.off"))[:max_per_class]
        for mesh_path in mesh_paths:
            try:
                vertices, faces = parse_off(mesh_path)
                tris = triangulate(faces)
            except Exception as exc:
                print(f"  Skipping {mesh_path.name}: {exc!r}", flush=True)
                continue
            yield cls, vertices, tris


def iter_geometric_shapes(cache_dir, classes, split, max_per_class):
    """
    Iterate torch_geometric.datasets.GeometricShapes meshes.

    Each item is a torch_geometric Data object with `pos` (V,3) and
    `face` (3,F). Class names come from the dataset's `categories` attribute
    (or are inferred from raw subdirs as a fallback).
    """
    try:
        from torch_geometric.datasets import GeometricShapes
    except ImportError as exc:
        raise ImportError(
            "torch_geometric is required for --dataset geometric_shapes. "
            "Install with: pip install torch_geometric"
        ) from exc

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    train = (split == "train")
    ds = GeometricShapes(root=str(cache_dir), train=train)

    # Resolve class index -> name
    raw_dir = Path(ds.raw_dir)
    if raw_dir.exists():
        idx_to_name = sorted(p.name for p in raw_dir.iterdir() if p.is_dir())
    else:
        idx_to_name = [f"class_{i}" for i in range(int(ds.data.y.max().item()) + 1)]

    # Bucket items by class
    per_class = {name: [] for name in idx_to_name}
    for i in range(len(ds)):
        item = ds[i]
        name = idx_to_name[int(item.y.item())]
        per_class[name].append(item)

    requested = classes or idx_to_name
    for cls in requested:
        items = per_class.get(cls, [])[:max_per_class]
        if not items:
            print(f"Warning: no items for class {cls!r}", flush=True)
            continue
        for item in items:
            vertices = item.pos.detach().cpu().numpy().astype("float64")
            tris = item.face.detach().cpu().numpy().astype(np.int64).T  # (F, 3)
            yield cls, vertices, tris


DATASET_ITERATORS = {
    "geometric_shapes": iter_geometric_shapes,
    "modelnet10": iter_modelnet10,
}


# ============================================================
# Base measure sources
# ============================================================

def sample_base_from_class_mesh(args, cache_dir, n_points, rng):
    """
    Pick one mesh from `--base_class`, surface-sample `n_points` from it, and
    normalize to the unit cube. Returns (n_points, 3) float32.
    """
    if not args.base_class:
        raise ValueError("--base_class is required when --base_source class_mesh")

    if args.dataset == "modelnet10":
        extracted = Path(cache_dir) / "ModelNet10"
        cls_dir = extracted / args.base_class / args.split
        if not cls_dir.exists():
            raise FileNotFoundError(f"No {args.split} dir for class "
                                    f"{args.base_class!r}: {cls_dir}")
        mesh_paths = sorted(cls_dir.glob("*.off"))
        if not mesh_paths:
            raise FileNotFoundError(f"No .off meshes under {cls_dir}")
        if args.base_mesh_index >= len(mesh_paths):
            raise IndexError(f"--base_mesh_index {args.base_mesh_index} out of "
                             f"range; only {len(mesh_paths)} meshes available")
        mesh_path = mesh_paths[args.base_mesh_index]
        vertices, faces = parse_off(mesh_path)
        tris = triangulate(faces)
        source_id = mesh_path.name
    elif args.dataset == "geometric_shapes":
        items = list(iter_geometric_shapes(
            cache_dir, [args.base_class], args.split, max_per_class=10**9
        ))
        if args.base_mesh_index >= len(items):
            raise IndexError(f"--base_mesh_index {args.base_mesh_index} out of "
                             f"range for geometric_shapes class "
                             f"{args.base_class!r} ({len(items)} items)")
        _, vertices, tris = items[args.base_mesh_index]
        source_id = f"{args.base_class}#{args.base_mesh_index}"
    else:
        raise ValueError(f"Unsupported dataset for class_mesh base: {args.dataset}")

    pts = sample_surface(vertices, tris, n_points, rng)
    pts = normalize_to_unit_cube(pts)
    print(f"Base measure: {n_points} points sampled from {source_id} "
          f"(class={args.base_class!r}, normalized to [0,1]^3)", flush=True)
    return pts.astype("float32"), source_id


# ============================================================
# Main pipeline
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare 3D point-cloud OT maps.")
    parser.add_argument("--dataset", type=str, default="geometric_shapes",
                        choices=list(DATASET_ITERATORS.keys()),
                        help="Source dataset (default: geometric_shapes).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save base_measure.pt + class_*/mappings.pt. "
                             "Default depends on --dataset.")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Where to download/extract raw dataset files. "
                             "Default depends on --dataset.")
    parser.add_argument("--zip_path", type=str, default=None,
                        help="(modelnet10 only) pre-downloaded ModelNet10.zip "
                             "to skip the network download.")
    parser.add_argument("--base_supp_size", type=int, default=1000,
                        help="Number of points in the base measure.")
    parser.add_argument("--base_source", type=str, default="uniform",
                        choices=["uniform", "class_mesh"],
                        help="How to construct the base measure. 'uniform' = iid "
                             "uniform points on [0,1]^3 (default). 'class_mesh' = "
                             "surface-sample a single mesh from --base_class.")
    parser.add_argument("--base_class", type=str, default=None,
                        help="(--base_source class_mesh) class name to source the "
                             "base mesh from, e.g. 'chair'.")
    parser.add_argument("--base_mesh_index", type=int, default=0,
                        help="(--base_source class_mesh) index into the sorted "
                             "mesh list of --base_class (default 0 = first mesh).")
    parser.add_argument("--cloud_supp_size", type=int, default=1024,
                        help="Number of points sampled per cloud (target measure).")
    parser.add_argument("--max_per_class", type=int, default=100,
                        help="Max number of meshes to process per class.")
    parser.add_argument("--samples_per_mesh", type=int, default=None,
                        help="How many independent point-cloud samples to draw "
                             "from each mesh (each becomes its own OT map). "
                             "Default: 1 for modelnet10, 20 for geometric_shapes "
                             "(which only has 1 mesh per class).")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test"])
    parser.add_argument("--classes", type=str, nargs="*", default=None,
                        help="Subset of class names; default = all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ot_method", type=str, default="emd",
                        choices=["emd", "entropic"])
    parser.add_argument("--progress_every", type=int, default=5)
    return parser.parse_args()


def resolve_default_dirs(args):
    """
    Per-dataset defaults so different datasets don't trample each other.

    Paths are resolved relative to the script's location (REPO_ROOT == wdl_repo/)
    so the script works regardless of the caller's cwd.
    """
    if args.dataset == "modelnet10":
        out = args.output_dir or str(REPO_ROOT / "datasets" / "modelnet10_ot")
        cache = args.cache_dir or str(REPO_ROOT / "pointcloud_raw" / "modelnet10")
    else:
        out = args.output_dir or str(REPO_ROOT / "datasets" / "geomshapes_ot")
        cache = args.cache_dir or str(REPO_ROOT / "pointcloud_raw" / "geometric_shapes")
    return out, cache


def main():
    args = parse_args()
    t_start = time.time()
    rng = np.random.RandomState(args.seed)

    if args.samples_per_mesh is None:
        args.samples_per_mesh = 20 if args.dataset == "geometric_shapes" else 1
    if args.samples_per_mesh < 1:
        raise ValueError("--samples_per_mesh must be >= 1")

    out_dir, cache_dir = resolve_default_dirs(args)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Build iterator
    if args.dataset == "modelnet10":
        mesh_iter = iter_modelnet10(
            cache_dir, args.classes, args.split, args.max_per_class,
            zip_path=args.zip_path,
        )
    elif args.dataset == "geometric_shapes":
        mesh_iter = iter_geometric_shapes(
            cache_dir, args.classes, args.split, args.max_per_class,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Base measure
    base_source_id = None
    if args.base_source == "uniform":
        base_points = rng.rand(args.base_supp_size, 3).astype("float32")
        print(f"Dataset: {args.dataset}")
        print(f"Base measure: {args.base_supp_size} uniform random points on [0,1]^3",
              flush=True)
    elif args.base_source == "class_mesh":
        print(f"Dataset: {args.dataset}")
        base_points, base_source_id = sample_base_from_class_mesh(
            args, cache_dir, args.base_supp_size, rng,
        )
    else:
        raise ValueError(f"Unknown --base_source {args.base_source!r}")

    base_masses = np.full(args.base_supp_size, 1.0 / args.base_supp_size,
                          dtype="float64")
    base = measure(base_points, base_masses)
    torch.save(torch.tensor(base_points), out / "base_measure.pt")

    # Group meshes by class as we go
    raw_dir = out / "_raw_examples"
    raw_dir.mkdir(parents=True, exist_ok=True)

    class_buffers = {}  # class_name -> list of (n, 3) float32 clouds
    saved_example = set()

    for cls, vertices, tris in mesh_iter:
        for _ in range(args.samples_per_mesh):
            try:
                pts = sample_surface(vertices, tris, args.cloud_supp_size, rng)
                pts = normalize_to_unit_cube(pts)
            except Exception as exc:
                print(f"  Skipping {cls} mesh: {exc!r}", flush=True)
                break
            class_buffers.setdefault(cls, []).append(pts)

            # Save the first usable cloud per class as a raw example for plotting
            if cls not in saved_example:
                torch.save(torch.tensor(pts), raw_dir / f"{cls}.pt")
                saved_example.add(cls)

    # Compute OT maps per class (separated so we can print clean per-class progress)
    class_counts = {}
    for cls, clouds in class_buffers.items():
        cls_dir = out / f"class_{cls}"
        cls_dir.mkdir(parents=True, exist_ok=True)
        cls_start = time.time()
        print(f"\nClass {cls!r}: computing {len(clouds)} OT maps "
              f"(base={args.base_supp_size}, cloud={args.cloud_supp_size})",
              flush=True)

        target_masses = np.full(args.cloud_supp_size, 1.0 / args.cloud_supp_size,
                                dtype="float64")
        maps_list = []
        for i, pts in enumerate(clouds):
            if i == 0 or (args.progress_every > 0 and i % args.progress_every == 0):
                elapsed = time.time() - cls_start
                rate = i / elapsed if elapsed > 0 else 0.0
                eta = (len(clouds) - i) / rate if rate > 0 else float("nan")
                print(f"  {cls}: {i}/{len(clouds)}  elapsed={elapsed:.1f}s  eta={eta:.1f}s",
                      flush=True)
            target = measure(pts, target_masses)
            ot_map = wass_map(base, target, args.ot_method)
            maps_list.append(torch.tensor(ot_map, dtype=torch.float32))

        if not maps_list:
            print(f"  No usable clouds for class {cls!r}; skipping save.", flush=True)
            continue

        mappings = torch.stack(maps_list)  # (N_cls, base_supp_size, 3)
        torch.save(mappings, cls_dir / "mappings.pt")

        # Also persist the raw target clouds (mu_i) that generated these maps,
        # in the SAME order as `mappings`.  These are the pre-OT ground-truth
        # measures; downstream baselines that work directly on point clouds
        # (e.g. the PointNet autoencoder) and the native Wasserstein
        # reconstruction metric use them instead of the transported base points.
        raw_clouds = torch.stack([
            torch.as_tensor(p, dtype=torch.float32) for p in clouds[:len(maps_list)]
        ])  # (N_cls, cloud_supp_size, 3)
        torch.save(raw_clouds, cls_dir / "raw_clouds.pt")

        class_counts[cls] = len(maps_list)
        cls_elapsed = time.time() - cls_start
        total_elapsed = time.time() - t_start
        print(f"  Saved {len(maps_list)} OT maps for {cls!r}  "
              f"shape={tuple(mappings.shape)}  "
              f"class_elapsed={cls_elapsed:.1f}s  total_elapsed={total_elapsed:.1f}s",
              flush=True)

    metadata = {
        "format": "pointcloud_ot_maps",
        "version": 1,
        "dataset": args.dataset,
        "parameters": {
            "base_supp_size": args.base_supp_size,
            "cloud_supp_size": args.cloud_supp_size,
            "max_per_class": args.max_per_class,
            "samples_per_mesh": args.samples_per_mesh,
            "split": args.split,
            "ot_method": args.ot_method,
            "seed": args.seed,
            "base_source": args.base_source,
            "base_class": args.base_class,
            "base_mesh_index": args.base_mesh_index,
            "base_source_id": base_source_id,
        },
        "classes": sorted(class_counts.keys()),
        "class_counts": class_counts,
        "elapsed_seconds": time.time() - t_start,
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata written to {out / 'metadata.json'}")
    print(f"Done. Total elapsed: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
