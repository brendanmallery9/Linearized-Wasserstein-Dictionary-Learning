import os
import argparse
import subprocess
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import numpy as np

from SAE_analysis_functions import *

SAE_PARAMETERS = {
    #     'JUMPRELUAE_512_1e-4':{'architecture': 'JumpReLU', 'l1': '1e-4', 'hidden_dim': 512,  'top_K': 0, 'normalize': True},
     'JUMPRELUAE_2048_1e-4':{'architecture': 'JumpReLU', 'l1': '1e-4', 'hidden_dim': 2048,  'top_K': 0, 'normalize': True},
          'JUMPRELUAE_2048_1e-2':{'architecture': 'JumpReLU', 'l1': '1e-4', 'hidden_dim': 2048,  'top_K': 0, 'normalize': True},

}

SAE_SCRIPT = str(REPO_ROOT / "SAE.py")
EXT = ".pt"


def parse_args():
    p = argparse.ArgumentParser(description="Stack activations and train SAEs on them")
    p.add_argument("--data_dir",  type=str, required=True,
                   help="Directory containing activation .pt files (each with 'acts' key)")
    p.add_argument("--out_root",  type=str, required=True,
                   help="Root dir for SAE run outputs")
    p.add_argument("--stack_dir", type=str, required=True,
                   help="Dir to save the stacked activations tensor (used as SAE data_path)")
    p.add_argument("--n_files",   type=int, default=-1,
                   help="Max files to load; -1 means use all (default: -1)")
    p.add_argument("--epochs",           type=int,   default=30)
    p.add_argument("--lr",               type=float, default=5e-6)
    p.add_argument("--batch_size",       type=int,   default=2048)
    p.add_argument("--seed",             type=int,   default=0)
    p.add_argument("--device",           type=str,   default="cuda",
                   help="Device, e.g. 'cuda', 'cuda:0', 'cuda:1', 'cpu'")
    p.add_argument("--normalize",        action="store_true", default=False,
                   help="Normalize input vectors (overridden per-trial by SAE_PARAMETERS)")
    p.add_argument("--pca_transform",    type=str,   default=None,
                   help="Path to fitted PCA .pt file; if given, activations are projected before training")
    p.add_argument("--validation_path",  type=str,   default=None,
                   help="Path to validation data directory (optional)")
    p.add_argument("--val_frequency",    type=float, default=0.2,
                   help="Fraction of epochs between validation runs (default: 0.2)")
    p.add_argument("--early_stop",       action="store_true", default=True,
                   help="Enable early stopping (default: True)")
    p.add_argument("--patience",         type=int,   default=200,
                   help="Epochs to wait for improvement before stopping (default: 200)")
    p.add_argument("--min_delta",        type=float, default=1e-5,
                   help="Minimum relative improvement to count as progress (default: 1e-5)")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    data_dir  = Path(args.data_dir)
    out_root  = Path(args.out_root)
    stack_dir = Path(args.stack_dir)
    stack_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    # ── Stacking ──────────────────────────────────────────────────────────────
    stack_path = stack_dir / "stacked_activations.pt"

    if stack_path.exists():
        print(f"Stacked activations already exist at {stack_path}, skipping stacking step.")
        stacked_tensor = torch.load(stack_path, map_location="cpu", weights_only=False)
        INPUT_DIM = stacked_tensor.shape[1]
        print(f"Loaded existing stacked tensor shape: {tuple(stacked_tensor.shape)}  INPUT_DIM={INPUT_DIM}")
    else:
        pt_files = sorted(data_dir.rglob(f"*{EXT}"))
        n_take = len(pt_files) if args.n_files == -1 else min(args.n_files, len(pt_files))
        print(f"Loading {n_take}/{len(pt_files)} activation files from {data_dir}", flush=True)

        all_tensors = []
        for i, p in enumerate(pt_files[:n_take]):
            obj = torch.load(p, map_location="cpu", weights_only=False)["acts"]
            t = obj if torch.is_tensor(obj) else torch.tensor(obj)
            if t.ndim == 1:
                t = t.unsqueeze(0)
            all_tensors.append(t)
            if (i + 1) % 500 == 0 or (i + 1) == n_take:
                print(f"  Loaded {i+1}/{n_take} files", flush=True)

        stacked_tensor = torch.cat(all_tensors, dim=0).float()

        if args.pca_transform is not None:
            pca = torch.load(args.pca_transform, map_location="cpu", weights_only=False)
            print(f"Applying PCA ({pca.n_components} components) from {args.pca_transform}")
            stacked_np = pca.transform(stacked_tensor.numpy())
            stacked_tensor = torch.from_numpy(stacked_np.astype(np.float32))

        INPUT_DIM = stacked_tensor.shape[1]
        print(f"Stacked tensor shape: {tuple(stacked_tensor.shape)}  INPUT_DIM={INPUT_DIM}")

        torch.save(stacked_tensor, stack_path)
        print(f"Saved stacked tensor -> {stack_path}")

    # ── GPU detection ─────────────────────────────────────────────────────────
    n_gpus = torch.cuda.device_count()
    if n_gpus > 0:
        devices = [f"cuda:{i}" for i in range(n_gpus)]
        print(f"Found {n_gpus} GPU(s): {devices}")
    else:
        devices = ["cpu"]
        print("No GPUs found, running on CPU.")

    # ── Launch all trials in parallel ─────────────────────────────────────────
    trials = list(SAE_PARAMETERS.items())
    procs  = []   # (trial_name, device, Popen, log_file_handle, run_dir)

    for idx, (trial_name, params) in enumerate(trials):
        device       = devices[idx % len(devices)]
        l1           = params["l1"]
        architecture = params["architecture"]
        hidden_dim   = params["hidden_dim"]
        top_k        = params["top_K"]
        normalize    = params.get("normalize", args.normalize)

        run_dir = out_root / trial_name
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", SAE_SCRIPT,
            "--data_path",    str(stack_dir),
            "--hidden_dim",   str(hidden_dim),
            "--architecture", str(architecture),
            "--input_dim",    str(INPUT_DIM),
            "--batch_size",   str(args.batch_size),
            "--epochs",       str(args.epochs),
            "--lr",           str(args.lr),
            "--l1",           str(l1),
            "--top_k",        str(top_k),
            "--log_every",    "1",
            "--device",       device,
            "--log_dir",      str(run_dir),
        ]
        if args.early_stop:
            cmd += ["--early_stop",
                    "--patience",  str(args.patience),
                    "--min_delta", str(args.min_delta)]
        cmd.append("--normalize" if normalize else "--no-normalize")
        if args.validation_path is not None:
            cmd += ["--validation_path", args.validation_path,
                    "--val_frequency",   str(args.val_frequency)]

        log_path = run_dir / "train.log"
        log_fh   = open(log_path, "w")

        print(f"Launching trial '{trial_name}' on {device}  (log -> {log_path})")
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
        procs.append((trial_name, device, proc, log_fh, run_dir))

    # ── Wait for all trials and collect checkpoint paths ──────────────────────
    failed = []
    for trial_name, device, proc, log_fh, run_dir in procs:
        ret = proc.wait()
        log_fh.close()

        log_path = run_dir / "train.log"
        log_text = log_path.read_text()
        print(f"\n{'='*60}\nTrial: {trial_name}  |  device={device}  |  exit={ret}\n{'='*60}")
        print(log_text)

        if ret != 0:
            failed.append(trial_name)
            continue

        ckpt_line = next((l for l in log_text.splitlines() if l.startswith("CKPT_PATH=")), None)
        if ckpt_line is None:
            print(f"WARNING: CKPT_PATH not found in log for trial '{trial_name}'")
            failed.append(trial_name)
        else:
            print(f"Checkpoint: {ckpt_line.split('=', 1)[1].strip()}")

    if failed:
        raise RuntimeError(f"The following trials failed: {failed}")
    print("\nAll trials completed successfully.")


if __name__ == "__main__":
    main()
