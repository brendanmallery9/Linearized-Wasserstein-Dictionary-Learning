from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

from common import REPO_ROOT, stream_command, timestamp, write_json


def maps_complete(
    data_dir: Path,
    *,
    base_supp_size: int,
    max_per_digit: int,
    seed: int,
    base_mode: str,
    base_digit: int,
    base_noise_std: float,
    base_n_components: int,
) -> bool:
    metadata_path = data_dir / "metadata.json"
    base_path = data_dir / "base_measure.pt"
    if not metadata_path.exists() or not base_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
        params = metadata["parameters"]
        base = torch.load(base_path, map_location="cpu")
    except Exception:
        return False
    expected = {
        "base_supp_size": base_supp_size,
        "max_per_digit": max_per_digit,
        "seed": seed,
        "base_mode": base_mode,
        "base_digit": base_digit,
        "base_n_components": base_n_components,
    }
    for key, value in expected.items():
        if params.get(key) != value:
            return False
    if abs(float(params.get("base_noise_std", -1.0)) - base_noise_std) > 1e-12:
        return False
    if tuple(base.shape) != (base_supp_size, 2):
        return False
    for digit in range(10):
        path = data_dir / f"digit_{digit}" / "mappings.pt"
        if not path.exists():
            return False
        try:
            maps = torch.load(path, map_location="cpu")
        except Exception:
            return False
        if tuple(maps.shape) != (max_per_digit, base_supp_size, 2):
            return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time the local MNIST OT-map SAE method.")
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "experiments" / "results" / f"mnist_sae_{timestamp()}")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--map-mode", choices=["force", "auto", "skip"], default="force")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--base-supp-size", type=int, default=400)
    parser.add_argument("--max-per-digit", type=int, default=2000)
    parser.add_argument("--base-mode", choices=["uniform", "digit", "mixture"], default="uniform")
    parser.add_argument("--base-digit", type=int, default=0)
    parser.add_argument("--base-noise-std", type=float, default=0.02)
    parser.add_argument("--base-n-components", type=int, default=100)
    parser.add_argument("--m", type=int, default=30)
    parser.add_argument("--lista-steps", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=0.025)
    parser.add_argument("--c", type=float, default=0.0001)
    parser.add_argument("--history-every", type=int, default=1)
    parser.add_argument("--force-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    data_dir = (args.data_dir or (run_dir / "data" / "mnist_ot")).resolve()
    output_dir = (args.output_dir or (run_dir / "outputs")).resolve()
    log_dir = (args.log_dir or (run_dir / "logs")).resolve()
    history_path = run_dir / "history.jsonl"
    summary_path = run_dir / "summary.json"

    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if history_path.exists():
        history_path.unlink()
    if output_dir.exists() and args.force_output:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    complete = maps_complete(
        data_dir,
        base_supp_size=args.base_supp_size,
        max_per_digit=args.max_per_digit,
        seed=args.seed,
        base_mode=args.base_mode,
        base_digit=args.base_digit,
        base_noise_std=args.base_noise_std,
        base_n_components=args.base_n_components,
    )
    run_maps = args.map_mode == "force" or (args.map_mode == "auto" and not complete)
    if args.map_mode == "skip" and not complete:
        raise RuntimeError(f"Requested --map-mode skip, but maps are incomplete: {data_dir}")

    prep_elapsed = 0.0
    prepare_cmd = [
        sys.executable,
        REPO_ROOT / "mnist" / "pipeline" / "prepare_mnist_ot.py",
        "--output_dir", data_dir,
        "--base_supp_size", args.base_supp_size,
        "--max_per_digit", args.max_per_digit,
        "--seed", args.seed,
        "--base_mode", args.base_mode,
        "--base_digit", args.base_digit,
        "--base_noise_std", args.base_noise_std,
        "--base_n_components", args.base_n_components,
        "--progress_every", args.progress_every,
    ]
    if run_maps:
        returncode, prep_elapsed = stream_command(
            prepare_cmd,
            cwd=REPO_ROOT,
            log_path=log_dir / "prepare_mnist_ot.log",
        )
        if returncode != 0:
            sys.exit(returncode)
    else:
        print(f"[maps] reused existing maps at {data_dir}")

    with history_path.open("a") as f:
        f.write(json.dumps({
            "event": "map_prep",
            "method": "mnist_ot_sae",
            "elapsed_seconds": prep_elapsed,
            "ran": run_maps,
            "data_dir": str(data_dir),
        }) + "\n")

    train_cmd = [
        sys.executable,
        REPO_ROOT / "mnist" / "pipeline" / "train_mnist_sae.py",
        "--data_dir", data_dir,
        "--output_dir", output_dir,
        "--device", args.device,
        "--seed", args.seed,
        "--m", args.m,
        "--lista_steps", args.lista_steps,
        "--epochs", args.epochs,
        "--batch_size", args.batch_size,
        "--lr", args.lr,
        "--epsilons", args.eps,
        "--sparsity_coeffs", args.c,
        "--history_path", history_path,
        "--history_every", args.history_every,
        "--history_time_offset", prep_elapsed,
    ]
    returncode, train_elapsed = stream_command(
        train_cmd,
        cwd=REPO_ROOT,
        log_path=log_dir / "train_mnist_sae.log",
    )
    if returncode != 0:
        sys.exit(returncode)

    total_elapsed = prep_elapsed + train_elapsed
    summary = {
        "method": "mnist_ot_sae",
        "elapsed_seconds": total_elapsed,
        "map_prep_elapsed_seconds": prep_elapsed,
        "train_elapsed_seconds": train_elapsed,
        "history_path": str(history_path),
        "run_dir": str(run_dir),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "log_dir": str(log_dir),
        "parameters": {
            "device": args.device,
            "seed": args.seed,
            "base_supp_size": args.base_supp_size,
            "max_per_digit": args.max_per_digit,
            "base_mode": args.base_mode,
            "base_digit": args.base_digit,
            "base_noise_std": args.base_noise_std,
            "base_n_components": args.base_n_components,
            "m": args.m,
            "lista_steps": args.lista_steps,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "eps": args.eps,
            "c": args.c,
            "map_mode": args.map_mode,
        },
        "commands": {
            "prepare": [str(part) for part in prepare_cmd],
            "train": [str(part) for part in train_cmd],
        },
    }
    write_json(summary_path, summary)


if __name__ == "__main__":
    main()

