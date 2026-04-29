import os
import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from brenier_embedding_functions import *
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm

from datetime import datetime
import traceback

def log_error(msg, log_path=None):
    ts = datetime.now().isoformat(timespec="seconds")
    line = f"[{ts}] {msg}"
    print(line)
    if log_path is not None:
        with open(log_path, "a") as f:
            f.write(line + "\n")

def atomic_torch_save(obj, path: Path):
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


# ---- globals for workers ----
import numpy as np

_SOURCE_ARRAY = None
_SOURCE_MASSES = None
_METHOD = None
_EPS_REG=None
_PCA_TRANSFORM = None  # <<<< CHANGE 1: Added global

def _init_worker(source_path: str, method: str, eps_reg: float, pca_path: str = None):  # <<<< CHANGE 2: Added pca_path parameter
    global _SOURCE_ARRAY, _SOURCE_MASSES, _METHOD, _EPS_REG, _PCA_TRANSFORM  # <<<< CHANGE 3: Added _PCA_TRANSFORM to global

    d = torch.load(source_path, map_location="cpu", weights_only=False)  # <<<< CHANGE 4: Added weights_only=False
    source_tensor = d["acts"]

    # <<<< CHANGE 5: Load and apply PCA if provided (START)
    if pca_path is not None and pca_path != "":
        _PCA_TRANSFORM = torch.load(pca_path, map_location="cpu", weights_only=False)
        source_array = _PCA_TRANSFORM.transform(source_tensor.numpy())
    else:
        _PCA_TRANSFORM = None
        source_array = source_tensor.numpy()
    # <<<< CHANGE 5: Load and apply PCA if provided (END)
    
    n1 = source_array.shape[0]
    source_masses = np.ones(n1) / n1

    _SOURCE_ARRAY = source_array
    _SOURCE_MASSES = source_masses
    _METHOD = method
    _EPS_REG=eps_reg

def _process_one(tgt_path_str: str, out_path_str: str, noise_dir_name: str, source_path: str,object:str, error_log:str):
    global _SOURCE_ARRAY, _METHOD, _EPS_REG, _PCA_TRANSFORM  # <<<< CHANGE 6: Added _PCA_TRANSFORM to global
    tgt_path = Path(tgt_path_str)
    out_path = Path(out_path_str)
    try:
        target_dict = torch.load(tgt_path, map_location="cpu", weights_only=False)
        target_tensor = target_dict["acts"]

        if _PCA_TRANSFORM is not None:
            target_tensor = torch.from_numpy(_PCA_TRANSFORM.transform(target_tensor.numpy()))

        source_masses=None
        target_masses=None
        if object=="potential":
            obj = brenier_potential(_SOURCE_ARRAY,None, target_tensor,None, method=_METHOD,eps_reg=_EPS_REG)
            obj = obj-obj.mean() #normalizes potential to have mean zero
        if object=="map":
            obj=wass_map(_SOURCE_ARRAY, target_tensor, method=_METHOD)

        payload = {
            "noise_dir": noise_dir_name,
            "source_path": source_path,
            "target_path": str(tgt_path),
            "method": _METHOD,
            "object": obj,
            "doc_index": target_dict.get("doc_index"),
            "text": target_dict.get("text"),
            "meta": target_dict.get("meta"),
        }
        atomic_torch_save(payload, out_path)
        return ("ok", str(out_path))
    except Exception as e:
        # log and skip
        log_error(f"FAILED {tgt_path}: {type(e).__name__}: {e}", log_path=error_log)
        log_error(traceback.format_exc(), log_path=error_log)
        return ("fail", str(tgt_path))

 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument( "--object", type=str, choices=["potential", "map"], required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--eps_reg", type=float, default=5e-3)
    parser.add_argument("--pca_transform", type=str, default=None, help="Path to PCA transform .pt file")  # <<<< CHANGE 9: Added argument
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_workers", type=int, default=max(1, os.cpu_count() // 2))
    args = parser.parse_args()

    AROOT = Path(args.activations_root)
    PROOT = Path(args.output_root)
    PROOT.mkdir(parents=True, exist_ok=True)

    error_log = str(PROOT / "output_errors.log")
    jobs = []
    for noise_dir in sorted([d for d in AROOT.iterdir() if d.is_dir()]):
        out_dir = PROOT / noise_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for tgt_path in sorted(noise_dir.glob("*.pt")):
            out_path = out_dir / tgt_path.name
            if out_path.exists() and not args.overwrite:
                continue
            jobs.append((str(tgt_path), str(out_path), noise_dir.name))

    print(f"Queued {len(jobs)} files with {args.num_workers} workers")
    if not jobs:
        print("Nothing to do.")
        return

    ok = fail = 0
    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=_init_worker,
        initargs=(args.source_path, args.method, args.eps_reg, args.pca_transform),  # <<<< CHANGE 10: Added args.pca_transform
    ) as ex:
        futures = [
            ex.submit(_process_one, tgt, out, noise_name, args.source_path, args.object,error_log)
            for (tgt, out, noise_name) in jobs
        ]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="{}".format(args.object)):
            try:
                status, _ = fut.result()   # will not raise now
            except Exception as e:
                log_error(f"FUTURE FAILED: {e}", log_path=error_log)
                continue
            if status == "ok":
                ok += 1
            else:
                fail += 1

    print(f"Done. ok={ok}, fail={fail}. Fail log: {error_log}")

if __name__ == "__main__":
    main()

'''
# Without PCA transform:
nohup python multi_dir_brenier_embedding.py \
  --activations_root datasets/noised_luther/activations/noised_activations/EleutherAI__pythia-410m-deduped/ \
  --output_root datasets/noised_luther/potentials/EleutherAI__pythia-410m-deduped/raw_potentials \
  --source_path datasets/noised_luther/activations/base/luther_L2_residual.pt \
  --object potential \
  --method emd \
  --eps_reg 0.0 \
    >& brenier_embed.log &

# With PCA transform:
python multi_dir_brenier_embedding.py \
  --activations_root datasets/noised_luther/activations/noised_activations_2/EleutherAI__pythia-410m-deduped/raw_activations \
  --output_root datasets/noised_luther/potentials_2/EleutherAI__pythia-410m-deduped/50pca_potentials \
  --source_path datasets/noised_luther/activations/base/EleutherAI__pythia-410m-deduped/base/luther_L2_residual.pt \
  --object potential \
  --method emd \
  --eps_reg 0.0 \
  --pca_transform datasets/noised_luther/activations/noised_activations_2/EleutherAI__pythia-410m-deduped/activation_shard_dir/50_pca_transform.pt
'''
