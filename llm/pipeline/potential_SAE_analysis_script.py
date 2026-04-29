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

from SAE_analysis_functions import *

SAE_PARAMETERS = {
    'TOPKAE_10':       {'architecture': 'TopKAE',   'l1': '0',    'hidden_dim': 10,  'top_K': 2, 'normalize': False},
    'TOPKAE_40':       {'architecture': 'TopKAE',   'l1': '0',    'hidden_dim': 40,  'top_K': 3, 'normalize': False},
    'JUMPRELUAE_10':   {'architecture': 'JumpReLU', 'l1': '1e-3', 'hidden_dim': 10,  'top_K': 0, 'normalize': False},
    'JUMPRELUAE_40':   {'architecture': 'JumpReLU', 'l1': '1e-3', 'hidden_dim': 40,  'top_K': 0, 'normalize': False},
    'JUMPRELUAE_512':  {'architecture': 'JumpReLU', 'l1': '1e-1', 'hidden_dim': 512, 'top_K': 0, 'normalize': True},
}

SAE_SCRIPT = str(REPO_ROOT / "SAE.py")
EXT = ".pt"


def parse_args():
    p = argparse.ArgumentParser(description="Stack Kantorovich potentials and train SAEs on them")
    p.add_argument("--data_dir",  type=str, required=True,
                   help="Root dir containing subdirs of potential .pt files")
    p.add_argument("--out_root",  type=str, required=True,
                   help="Root dir for SAE run outputs")
    p.add_argument("--stack_dir", type=str, required=True,
                   help="Dir to save the stacked potentials tensor (used as SAE data_path)")
    p.add_argument("--n_per_dir", type=int, default=-1,
                   help="Max files to load per subdir; -1 means use all (default: -1)")
    p.add_argument("--epochs",           type=int,   default=2000)
    p.add_argument("--lr",               type=float, default=5e-6)
    p.add_argument("--batch_size",       type=int,   default=1024)
    p.add_argument("--seed",             type=int,   default=0)
    p.add_argument("--device",           type=str,   default="cuda:0",
                   help="Device, e.g. 'cuda', 'cuda:0', 'cuda:1', 'cpu'")
    p.add_argument("--normalize",        action="store_true", default=False,
                   help="L2-normalize each input vector to unit norm (overridden per-trial by SAE_PARAMETERS)")
    p.add_argument("--validation_path",  type=str,   default=None,
                   help="Path to validation data directory (optional)")
    p.add_argument("--val_frequency",    type=float, default=0.2,
                   help="Fraction of epochs between validation runs (default: 0.2)")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    data_dir  = Path(args.data_dir)
    out_root  = Path(args.out_root)
    stack_dir = Path(args.stack_dir)
    stack_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    dir_names = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    print(f"Subdirs: {dir_names}")

    all_tensors = []
    labels = []
    name_to_id = {name: i for i, name in enumerate(dir_names)}
    id_to_name = {i: n for n, i in name_to_id.items()}

    for name in dir_names:
        d = data_dir / name
        pt_files = sorted(d.rglob(f"*{EXT}"))
        n_take = len(pt_files) if args.n_per_dir == -1 else args.n_per_dir
        if len(pt_files) < n_take:
            print(f"Warning: {name} has only {len(pt_files)} files (requested {n_take}); using all")
            n_take = len(pt_files)
        print(f"{name}: loading {n_take}/{len(pt_files)} files")
        for p in pt_files[:n_take]:
            obj = torch.load(p, map_location="cpu", weights_only=False)["object"]
            all_tensors.append(obj if torch.is_tensor(obj) else torch.tensor(obj))
            labels.append(name_to_id[name])

    stacked_tensor = torch.stack(all_tensors, dim=0).float()  # (N, D)
    INPUT_DIM = stacked_tensor.shape[1]
    print(f"Stacked tensor shape: {tuple(stacked_tensor.shape)}  INPUT_DIM={INPUT_DIM}")

    int_labels = torch.tensor(labels, dtype=torch.long)
    str_labels = [id_to_name[i] for i in labels]

    stack_path = stack_dir / "stacked_potentials.pt"
    torch.save(stacked_tensor, stack_path)
    print(f"Saved stacked tensor -> {stack_path}")

    for trial_name, params in SAE_PARAMETERS.items():
        l1 = params["l1"]
        architecture = params["architecture"]
        hidden_dim = params["hidden_dim"]
        top_k = params["top_K"]
        # Per-trial normalize overrides the global --normalize flag
        normalize = params.get("normalize", args.normalize)

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
            "--device",       args.device,
            "--log_dir",      str(run_dir),
        ]
        # Pass --normalize or --no-normalize explicitly so SAE.py always
        # gets a clear signal regardless of its default.
        if normalize:
            cmd.append("--normalize")
        else:
            cmd.append("--no-normalize")

        if args.validation_path is not None:
            cmd += ["--validation_path", args.validation_path,
                    "--val_frequency",   str(args.val_frequency)]

        print(f"\n{'='*60}\nTrial: {trial_name}  |  normalize={normalize}\n{'='*60}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        model_save_path = None
        for line in proc.stdout:
            print(line, end="")
            if line.startswith("CKPT_PATH="):
                model_save_path = Path(line.split("=", 1)[1].strip())

        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"SAE process failed for trial {trial_name}")
        if model_save_path is None:
            raise RuntimeError(f"CKPT_PATH not found for trial {trial_name}")


if __name__ == "__main__":
    main()


#

'''

nohup python potential_SAE_analysis_script.py \
  --data_dir datasets/noised_luther/potentials/EleutherAI__pythia-410m-deduped/raw_potentials \
  --out_root datasets/noised_luther \
  --stack_dir datasets/noised_luther \
  --n_per_dir -1 \
  --epochs 2000 \
  --lr 5e-6 \
  --batch_size 1024 \
  --seed 0 \
  --device cuda \
  >& run.log &


'''
'''
python potential_SAE_analysis_script.py \
    --data_dir    "datasets/pile-100k/potentials" \
    --out_root    "datasets/pile-100k/SAE_params_non_normalized" \
    --stack_dir   "datasets/pile-100k/non_normalized_stacked" \
    --n_per_dir   -1 \
    --epochs      3000 \
    --lr          5e-4 \
    --batch_size  1024
'''
