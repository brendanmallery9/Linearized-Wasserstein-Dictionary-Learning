#!/usr/bin/env python3
"""
Download hyperspectral .mat files and convert them to Torch .pt files,
laid out so the downstream analysis scripts can read them directly:

    <BASE_DIR>/<name>/
        _raw_mat/<original.mat>           # temporary .mat sources unless --keep-raw-mat
        data/<name>_cube.pt               # float32 tensor, shape H x W x B
        ground_truth/<name>_gt.pt         # int64 tensor, shape H x W (if GT exists)

`data/` and `ground_truth/` each hold a single .pt so they round-trip
through find_single_pt / load_labels in the analysis layer.

Robust to GT-only .mat files (e.g. Indian_pines_gt.mat).
Works for classic MAT and MAT v7.3 (HDF5).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional, Any, Tuple

import numpy as np
import requests
import torch

BASE_DIR = Path("datasets/hsi_data")

DATASETS: Dict[str, Dict[str, str]] = {
    "indian_pines": {
        "Indian_pines_corrected.mat":
            "https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
        "Indian_pines_gt.mat":
            "https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
    },
    "salinas_a": {
        "SalinasA_corrected.mat":
            "https://www.ehu.eus/ccwintco/uploads/1/1a/SalinasA_corrected.mat",
        "SalinasA_gt.mat":
            "https://www.ehu.eus/ccwintco/uploads/a/aa/SalinasA_gt.mat",
    },
    "cuprite": {
        "Cuprite_f970619t01p02_r02_sc03.a.rfl.mat":
            "https://www.ehu.eus/ccwintco/uploads/7/7d/Cuprite_f970619t01p02_r02_sc03.a.rfl.mat",
    },
    "botswana": {
        "Botswana.mat": "https://www.ehu.eus/ccwintco/uploads/7/72/Botswana.mat",
        "Botswana_gt.mat": "https://www.ehu.eus/ccwintco/uploads/5/58/Botswana_gt.mat",
    },
    "pavia": {
        "Pavia.mat": "https://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat",
        "Pavia_gt.mat": "https://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat",
    },

}


# ----------------------------
# Download
# ----------------------------
import time
import requests
from pathlib import Path

def download(url: str, outpath: Path, chunk_size: int = 1024 * 1024, max_retries: int = 8) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # We reuse a Session for stability
    sess = requests.Session()

    for attempt in range(1, max_retries + 1):
        existing = outpath.stat().st_size if outpath.exists() else 0

        # Try to get remote size (not always present)
        remote_size = None
        try:
            head = sess.head(url, timeout=30, allow_redirects=True)
            if head.ok and head.headers.get("Content-Length"):
                remote_size = int(head.headers["Content-Length"])
        except Exception:
            pass

        # If we already have full file, skip
        if remote_size is not None and existing == remote_size and remote_size > 0:
            print(f"✓ Exists, complete: {outpath} ({existing/1e6:.1f}MB)")
            return

        headers = {}
        mode = "wb"
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
            mode = "ab"
            print(f"Resuming ({existing/1e6:.1f}MB already) → {outpath.name}")
        else:
            print(f"Downloading → {outpath.name}")

        try:
            with sess.get(url, stream=True, timeout=120, headers=headers, allow_redirects=True) as r:
                r.raise_for_status()

                # If server honored Range, Content-Range exists; Content-Length is remaining bytes
                remaining = r.headers.get("Content-Length")
                remaining = int(remaining) if remaining is not None else None
                total = (existing + remaining) if remaining is not None else remote_size

                downloaded = existing
                with open(outpath, mode) as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total is not None and total > 0:
                            pct = 100.0 * downloaded / total
                            print(f"\r  {downloaded/1e6:.1f}MB / {total/1e6:.1f}MB ({pct:.1f}%)", end="")
                        else:
                            print(f"\r  {downloaded/1e6:.1f}MB", end="")

            print("\n✓ Done\n")

            # Verify size if we know it
            final_size = outpath.stat().st_size
            if remote_size is not None and remote_size > 0 and final_size != remote_size:
                raise IOError(f"Downloaded size mismatch: got {final_size}, expected {remote_size}")

            return

        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ContentDecodingError,
                IOError) as e:
            print(f"\n⚠ Download interrupted (attempt {attempt}/{max_retries}): {e}")
            # exponential-ish backoff
            time.sleep(min(2 ** attempt, 30))
            continue

    raise RuntimeError(f"Failed to download after {max_retries} attempts: {url}")



# ----------------------------
# .mat loading (scipy or h5py)
# ----------------------------
def _load_mat_any(path: Path) -> Dict[str, Any]:
    """Return {varname: numpy array} best-effort for MAT (<v7.3) and v7.3 (HDF5)."""
    try:
        import scipy.io
        d = scipy.io.loadmat(str(path))
        return {k: v for k, v in d.items() if not k.startswith("__")}
    except NotImplementedError:
        pass
    except Exception:
        pass

    import h5py
    out: Dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        def _visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name] = np.array(obj[()])
        f.visititems(_visit)

    # simplify to leaf names
    simplified: Dict[str, Any] = {}
    for k, v in out.items():
        leaf = k.split("/")[-1]
        if leaf not in simplified:
            simplified[leaf] = v
    return simplified


# ----------------------------
# Heuristics to pick cube / gt
# ----------------------------
def _as_array(v: Any) -> Optional[np.ndarray]:
    return v if isinstance(v, np.ndarray) else None


def _score_cube(arr: np.ndarray) -> float:
    if arr.ndim != 3:
        return -1.0
    H, W, B = arr.shape
    if min(H, W, B) <= 1:
        return -1.0
    score = float(arr.size)
    if B < 5:
        score *= 0.1
    return score


def _score_gt(arr: np.ndarray) -> float:
    if arr.ndim != 2:
        return -1.0
    score = float(arr.size)
    try:
        u = np.unique(arr.astype(np.int64))
        if len(u) <= 512:
            score *= 1.5
    except Exception:
        pass
    return score


def extract_cube_gt_from_mat(vars_dict: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str], Optional[str]]:
    """
    From a single .mat variable dict, extract:
      - cube (best 3D candidate) OR None if not present
      - gt   (best 2D candidate) OR None if not present
    """
    arrays = {k: _as_array(v) for k, v in vars_dict.items()}
    arrays = {k: v for k, v in arrays.items() if v is not None}

    if not arrays:
        return None, None, None, None

    # best cube candidate (may still be non-3D -> treat as None)
    cube_key = max(arrays.keys(), key=lambda k: _score_cube(arrays[k]))
    cube = arrays[cube_key]
    if cube.ndim != 3:
        cube, cube_key = None, None

    # best gt candidate (independent of cube)
    gt_candidates = [(k, a) for k, a in arrays.items() if a.ndim == 2]
    gt = None
    gt_key = None
    if gt_candidates:
        gt_key, gt = max(gt_candidates, key=lambda kv: _score_gt(kv[1]))

    return cube, gt, cube_key, gt_key


def to_torch_save(cube: np.ndarray, gt: Optional[np.ndarray], dataset_root: Path, name: str) -> None:
    """
    Save cube to <dataset_root>/data/<name>_cube.pt
    and gt   to <dataset_root>/ground_truth/<name>_gt.pt (if provided).
    """
    cube_dir = dataset_root / "data"
    cube_dir.mkdir(parents=True, exist_ok=True)

    cube_np = np.asarray(cube)
    cube_t = torch.from_numpy(cube_np).to(torch.float32).contiguous()
    torch.save(cube_t, cube_dir / f"{name}_cube.pt")

    if gt is not None:
        gt_dir = dataset_root / "ground_truth"
        gt_dir.mkdir(parents=True, exist_ok=True)

        gt_np = np.asarray(gt)
        H, W, _ = cube_t.shape
        if gt_np.shape == (W, H):
            gt_np = gt_np.T
        gt_t = torch.from_numpy(gt_np.astype(np.int64, copy=False)).contiguous()
        torch.save(gt_t, gt_dir / f"{name}_gt.pt")


def save_gt_only(
    gt: np.ndarray,
    dataset_root: Path,
    name: str,
    expected_shape: Optional[Tuple[int, int]] = None,
) -> None:
    """Save ground truth when the cube was restored from bundled chunks."""
    gt_dir = dataset_root / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    gt_np = np.asarray(gt)
    if expected_shape is not None and gt_np.shape == (expected_shape[1], expected_shape[0]):
        gt_np = gt_np.T
    gt_t = torch.from_numpy(gt_np.astype(np.int64, copy=False)).contiguous()
    torch.save(gt_t, gt_dir / f"{name}_gt.pt")


def restore_pavia_cube_from_chunks(dataset_root: Path) -> bool:
    """Reassemble the bundled Pavia cube chunks, if present."""
    cube_path = dataset_root / "data" / "pavia_cube.pt"
    parts_dir = dataset_root / "data" / "pavia_cube.pt.parts"
    checksum_path = parts_dir / "sha256.txt"
    parts = sorted(parts_dir.glob("part-*"))

    if not parts:
        return False

    if cube_path.exists():
        print(f"✓ Pavia cube already assembled: {cube_path}")
        return True

    print(f"Reassembling bundled Pavia cube chunks → {cube_path}")
    cube_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cube_path.with_suffix(cube_path.suffix + ".tmp")

    try:
        with open(tmp_path, "wb") as out:
            for part in parts:
                print(f"  adding {part.name}")
                with open(part, "rb") as inp:
                    for chunk in iter(lambda: inp.read(1024 * 1024), b""):
                        out.write(chunk)

        if checksum_path.exists():
            expected = checksum_path.read_text().split()[0]
            h = hashlib.sha256()
            with open(tmp_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            actual = h.hexdigest()
            if actual != expected:
                raise RuntimeError(
                    f"Pavia cube checksum mismatch: got {actual}, expected {expected}"
                )

        tmp_path.replace(cube_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    print(f"✓ Restored: {cube_path}")
    return True


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Download hyperspectral datasets")
    parser.add_argument("--root", type=Path, default=BASE_DIR,
                        help="Root directory to download into (default: %(default)s)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        choices=list(DATASETS.keys()),
                        help="Datasets to download (default: all)")
    parser.add_argument("--keep-raw-mat", action="store_true",
                        help="Keep downloaded .mat source files after conversion. "
                             "By default they are treated as temporary files.")
    args = parser.parse_args()

    root = args.root
    names = args.datasets or list(DATASETS.keys())

    for name in names:
        files = DATASETS[name]
        dataset_root = root / name
        mat_dir = dataset_root / "_raw_mat"
        print(f"\n=== {name} ===")

        pavia_cube_restored = name == "pavia" and restore_pavia_cube_from_chunks(dataset_root)
        if pavia_cube_restored:
            gt_path = dataset_root / "ground_truth" / "pavia_gt.pt"
            if gt_path.exists():
                print(f"✓ Pavia GT already available: {gt_path}")
                print("✓ Using bundled Pavia cube chunks; skipping remote Pavia download.")
                continue
            files = {"Pavia_gt.mat": files["Pavia_gt.mat"]}

        mat_paths = []
        for fname, url in files.items():
            outpath = mat_dir / fname
            download(url, outpath)
            mat_paths.append(outpath)

        cube: Optional[np.ndarray] = None
        gt: Optional[np.ndarray] = None

        for p in mat_paths:
            vars_dict = _load_mat_any(p)
            c, g, ckey, gkey = extract_cube_gt_from_mat(vars_dict)

            if c is not None and cube is None:
                cube = c
                print(f"  cube from {p.name} (var='{ckey}', shape={tuple(c.shape)})")

            if g is not None and gt is None:
                gt = g
                print(f"  gt   from {p.name} (var='{gkey}', shape={tuple(g.shape)})")

        if cube is None:
            if pavia_cube_restored:
                if gt is not None:
                    save_gt_only(gt, dataset_root, name, expected_shape=(1096, 715))
                    print(f"✓ Saved: {dataset_root / 'ground_truth' / (name + '_gt.pt')}")
                else:
                    print("ℹ No GT found; only restored Pavia cube.")

                if not args.keep_raw_mat:
                    for p in mat_paths:
                        try:
                            p.unlink()
                            print(f"✓ Removed temporary source: {p}")
                        except FileNotFoundError:
                            pass
                    try:
                        mat_dir.rmdir()
                    except OSError:
                        pass
                continue

            raise RuntimeError(f"Failed to find a 3D cube for dataset '{name}'.")

        # If we found GT but it doesn't match cube H/W, warn (still save as-is)
        if gt is not None:
            H, W, _ = cube.shape
            if gt.shape not in [(H, W), (W, H)]:
                print(f"  ⚠ GT shape {gt.shape} doesn't match cube spatial {(H, W)}; saving anyway.")

        to_torch_save(cube, gt, dataset_root, name)

        print(f"✓ Saved: {dataset_root / 'data' / (name + '_cube.pt')}")
        if gt is not None:
            print(f"✓ Saved: {dataset_root / 'ground_truth' / (name + '_gt.pt')}")
        else:
            print("ℹ No GT found; only saved cube.")

        if not args.keep_raw_mat:
            for p in mat_paths:
                try:
                    p.unlink()
                    print(f"✓ Removed temporary source: {p}")
                except FileNotFoundError:
                    pass
            try:
                mat_dir.rmdir()
            except OSError:
                pass

    print(f"\nAll done. Data in {root}")


if __name__ == "__main__":
    # pip install requests scipy h5py torch
    main()
