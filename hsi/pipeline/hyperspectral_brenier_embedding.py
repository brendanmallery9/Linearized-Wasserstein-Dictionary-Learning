import os
import argparse
import traceback
from pathlib import Path
from datetime import datetime
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from brenier_embedding_functions import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import torch
import numpy as np

class measure:
    def __init__(self, points, masses):
        self.points = points
        self.masses = masses


# ---- globals for workers ----
_SOURCE_ARRAY = None
_EPS_REG = None

_CUBE = None

def _init_worker(cube_path_str: str, source_supp_size: int, eps_reg: float):
    global _CUBE, _SOURCE_ARRAY, _EPS_REG
    cube = torch.load(cube_path_str, map_location="cpu")
    if isinstance(cube, dict):
        cube = cube.get("data", cube.get("cube"))
    _CUBE = cube.numpy()
    _SOURCE_ARRAY = source_supp_size
    _EPS_REG = eps_reg

def _process_chunk(pixel_indices, source_supp_size, chunk_id):
    global _CUBE
    # Remove the cube loading here - use _CUBE directly
    
    results = []
    target_grid_spacing = np.linspace(0.0, 1.0, _CUBE.shape[2])
    source_grid_spacing = np.linspace(0.0, 1.0, source_supp_size)

    for idx, (i, j) in enumerate(pixel_indices):
        if idx % 50 == 0:
            print(f"Chunk {chunk_id}: {idx}/{len(pixel_indices)} pixels")
        try:
            target_masses = _CUBE[i, j, :].reshape(-1)
            result = wass_map_1D(source_grid_spacing, None, target_grid_spacing, target_masses)
            result_np = result.cpu().numpy() if torch.is_tensor(result) else np.array(result)
            results.append((i, j, result_np))
        except Exception as e:
            print(f"Error at pixel ({i},{j}): {e}")
            results.append((i, j, None))
    
    return results

def process_cube(cube_path, source_supp_size, eps_reg, num_workers, output_path):
    """Process entire hyperspectral cube in parallel with incremental saving."""
    cube = torch.load(cube_path, map_location="cpu")
    if isinstance(cube, dict):
        cube = cube.get("data", cube.get("cube"))
    
    width, height, bands = cube.shape
    print(f"Processing {width}x{height}x{bands} cube")
    
    # Pre-allocate output cube
    out_cube = torch.full((width, height, source_supp_size), float("nan"), dtype=torch.float32)
    
    all_pixels = [(i, j) for i in range(width) for j in range(height)]
    chunk_size = max(1, len(all_pixels) // (num_workers * 4))
    chunks = [all_pixels[i:i + chunk_size] for i in range(0, len(all_pixels), chunk_size)]
    
    with ProcessPoolExecutor(max_workers=num_workers, initializer=_init_worker,
                            initargs=(str(cube_path), source_supp_size, eps_reg)) as ex:
        futures = [ex.submit(_process_chunk, chunk, source_supp_size, i)
                for i, chunk in enumerate(chunks)]
        
        completed = 0
        for fut in tqdm(as_completed(futures), total=len(futures)):
            chunk_results = fut.result()
            
            # Update output cube with this chunk's results
            for i, j, result in chunk_results:
                if result is not None:
                    result_tensor = torch.as_tensor(result, dtype=torch.float32).reshape(-1)
                    out_cube[i, j, :] = result_tensor
            
            # Save every N chunks (e.g., every 5 chunks)
            completed += 1
            if completed % 5 == 0:
                torch.save({"cube": out_cube}, output_path)
                print(f"\nCheckpoint saved: {completed}/{len(futures)} chunks")
    
    # Final save
    torch.save({"cube": out_cube}, output_path)
    return out_cube


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--source_supp_size", type=int, required=True)
    parser.add_argument("--eps_reg", type=float, default=5e-3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_workers", type=int, default=max(1, os.cpu_count() // 2))
    args = parser.parse_args()

    AROOT = Path(args.data_root)
    PROOT = Path(args.output_root)
    PROOT.mkdir(parents=True, exist_ok=True)

    for pt_file in sorted(AROOT.glob("*.pt")):
        out_path = PROOT / pt_file.name
        if out_path.exists() and not args.overwrite:
            print(f"Skipping {pt_file.name}")
            continue
        
        print(f"\nProcessing {pt_file.name}")
        out_cube = process_cube(pt_file, args.source_supp_size, args.eps_reg, 
                                args.num_workers, out_path)  # ← Pass output path
        
        if out_cube is not None:
            print(f"Saved to {out_path}")
        else:
            print(f"Failed to process {pt_file.name}")

if __name__ == "__main__":
    main()

'''
python hyperspectral_brenier_embedding.py \
    --data_root datasets/hsi_data/salinas_a/data \
    --output_root datasets/hsi_data/salinas_a/transport_maps \
    --source_supp_size 204 \
    --num_workers 12 \
    --overwrite
'''
