#!/usr/bin/env python3
"""Deterministic two-stage seed sweep for clean 02378 train-mean models.

Stage 1 screens several initialization seeds with a shortened run whose cosine
schedule still uses the full 1000-epoch horizon. Stage 2 retrains the strongest
seeds for the full horizon. Checkpoints are ranked by the same top-5 MNIST-map
nearest-neighbor purity used in the analysis notebook.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[2]
MNIST_PIPELINE = REPO_ROOT / "mnist" / "pipeline"
for path in (REPO_ROOT, MNIST_PIPELINE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mnist.analysis.evaluate_convex_combos import rebuild_model  # noqa: E402


RUNNER = REPO_ROOT / "pointcloud" / "pipeline" / "pointcloud_run_experiments.py"
RUN_NAME = "displacement_centered_eps0.025_c0.0001"


def stamp(message: str) -> None:
    print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)


def train_test_indices(n: int, test_fraction: float, seed: int) -> np.ndarray:
    n_test = int(n * test_fraction)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).numpy()
    return perm[n - n_test :]


class NeighborScorer:
    def __init__(self, data_dir: Path, mnist_ot_dir: Path, top_k: int = 5):
        self.data_dir = data_dir
        self.top_k = top_k
        self.true_atoms = torch.load(data_dir / "base_maps.pt", map_location="cpu").float().numpy()
        self.meta = json.loads((data_dir / "metadata.json").read_text())
        self.digits = [int(d) for d in self.meta["digits"]]
        self.original_index = int(self.meta["digit_sample_index"])
        self.test_fraction = 0.1
        self.split_seed = 42

        maps_by_digit = {
            digit: torch.load(
                mnist_ot_dir / f"digit_{digit}" / "mappings.pt", map_location="cpu"
            ).float().numpy()
            for digit in range(10)
        }
        self.restricted_pool = self._build_pool(maps_by_digit, self.digits)
        self.all_digits_pool = self._build_pool(maps_by_digit, range(10))

    def _build_pool(self, maps_by_digit, search_digits):
        maps, labels = [], []
        for digit in search_digits:
            digit_maps = maps_by_digit[digit]
            indices = train_test_indices(
                len(digit_maps), self.test_fraction, self.split_seed + digit
            )
            indices = indices[indices != self.original_index]
            maps.append(digit_maps[indices])
            labels.append(np.full(len(indices), digit, dtype=np.int64))

        # Match the notebook protocol by appending the five planted references.
        maps.append(self.true_atoms)
        labels.append(np.asarray(self.digits, dtype=np.int64))
        return np.concatenate(maps), np.concatenate(labels)

    @staticmethod
    def _l2rho_cost(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return ((left[:, None] - right[None]) ** 2).sum(axis=-1).mean(axis=-1)

    def score(self, checkpoint: Path, config_path: Path) -> dict:
        config = json.loads(config_path.read_text())
        model = rebuild_model(checkpoint, config, torch.device("cpu"))
        with torch.no_grad():
            learned = model.atoms_module().detach().cpu().numpy()

        atom_cost = self._l2rho_cost(learned, self.true_atoms)
        learned_rows, true_cols = linear_sum_assignment(atom_cost)
        learned_for_true = {int(t): int(a) for a, t in zip(learned_rows, true_cols)}

        def purity(pool):
            pool_maps, pool_labels = pool
            correct = 0
            per_digit = {}
            for true_index, digit in enumerate(self.digits):
                atom_index = learned_for_true[true_index]
                distances = ((pool_maps - learned[atom_index]) ** 2).sum(-1).mean(-1)
                nearest = np.argsort(distances)[: self.top_k]
                labels = pool_labels[nearest]
                count = int((labels == digit).sum())
                correct += count
                per_digit[str(digit)] = {
                    "correct": count,
                    "neighbors": [int(x) for x in labels],
                }
            return correct, per_digit

        restricted_correct, restricted_detail = purity(self.restricted_pool)
        all_correct, all_detail = purity(self.all_digits_pool)
        checkpoint_blob = torch.load(checkpoint, map_location="cpu")
        metrics = checkpoint_blob.get("metrics", {})
        matched_atom_mse = float(atom_cost[learned_rows, true_cols].mean())
        return {
            "restricted_correct": restricted_correct,
            "restricted_total": self.top_k * len(self.digits),
            "all_digits_correct": all_correct,
            "all_digits_total": self.top_k * len(self.digits),
            "matched_atom_mse": matched_atom_mse,
            "test_recon_loss": float(metrics.get("test_recon_loss", np.nan)),
            "best_epoch": int(checkpoint_blob.get("epoch", -1)),
            "restricted_detail": restricted_detail,
            "all_digits_detail": all_detail,
        }


def run_trial(args, stage: str, seed: int, epochs: int) -> tuple[Path, Path]:
    trial_dir = args.output_root / stage / f"seed_{seed:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = trial_dir / f"{RUN_NAME}_best.pt"
    config_path = trial_dir / "config.json"
    if checkpoint.exists() and config_path.exists() and not args.force:
        stamp(f"{stage} seed={seed}: existing checkpoint found; skipping training")
        return checkpoint, config_path

    command = [
        str(args.python), "-u", str(RUNNER),
        "--data_dir", str(args.data_dir),
        "--output_dir", str(trial_dir),
        "--m", "5",
        "--epochs", str(epochs),
        "--batch_size", "1024",
        "--lr", "1.5e-4",
        "--test_fraction", "0.1",
        "--optimizer", "adamw",
        "--weight_decay", "0",
        "--scheduler", "cosine",
        "--scheduler_t_max", str(args.final_epochs),
        "--lr_min", "5e-6",
        "--grad_clip_norm", "1.0",
        "--epsilons", "0.025",
        "--sparsity_coeffs", "1e-4",
        "--methods", "displacement_centered",
        "--displacement_center_mode", "train_mean",
        "--activation_type", "relu",
        "--topk_k", "2",
        "--lista_steps", "20",
        "--grid_mode", "uniform",
        "--grid_side", "64",
        "--device", args.device,
        "--seed", "42",
        "--model_seed", str(seed),
    ]
    log_path = trial_dir / "train.log"
    stamp(f"{stage} seed={seed}: training {epochs} epochs")
    with log_path.open("w") as log:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Trial failed (see {log_path}): {' '.join(command)}")
    return checkpoint, config_path


def write_leaderboard(path: Path, rows: list[dict]) -> None:
    fields = [
        "stage", "seed", "epochs", "restricted_correct", "restricted_total",
        "all_digits_correct", "all_digits_total", "matched_atom_mse",
        "test_recon_loss", "best_epoch", "checkpoint",
    ]
    ordered = sorted(
        rows,
        key=lambda row: (
            -row["restricted_correct"],
            -row["all_digits_correct"],
            row["matched_atom_mse"],
            row["test_recon_loss"],
        ),
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)


def evaluate_trial(scorer, checkpoint, config_path, stage, seed, epochs):
    score = scorer.score(checkpoint, config_path)
    score.update({
        "stage": stage,
        "seed": seed,
        "epochs": epochs,
        "checkpoint": str(checkpoint),
    })
    detail_path = checkpoint.parent / "nn_score.json"
    detail_path.write_text(json.dumps(score, indent=2) + "\n")
    stamp(
        f"{stage} seed={seed}: restricted={score['restricted_correct']}/"
        f"{score['restricted_total']} all={score['all_digits_correct']}/"
        f"{score['all_digits_total']} atom_mse={score['matched_atom_mse']:.6g}"
    )
    return score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path,
                        default=Path("mnist/results/convex_combos_02378_train_mean_seed_sweep"))
    parser.add_argument("--data_dir", type=Path,
                        default=Path("datasets/convex_combos_02378"))
    parser.add_argument("--mnist_ot_dir", type=Path,
                        default=Path("datasets/mnist_ot"))
    parser.add_argument("--python", type=Path, default=REPO_ROOT / ".venv/bin/python")
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], default="mps")
    parser.add_argument("--n_seeds", type=int, default=16)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--screen_epochs", type=int, default=350)
    parser.add_argument("--final_epochs", type=int, default=1000)
    parser.add_argument("--n_finalists", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for name in ("output_root", "data_dir", "mnist_ot_dir", "python"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, REPO_ROOT / value)
    return args


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    scorer = NeighborScorer(args.data_dir, args.mnist_ot_dir)
    screen_rows = []
    seeds = range(args.seed_start, args.seed_start + args.n_seeds)

    for seed in seeds:
        checkpoint, config = run_trial(args, "screen", seed, args.screen_epochs)
        screen_rows.append(evaluate_trial(
            scorer, checkpoint, config, "screen", seed, args.screen_epochs
        ))
        write_leaderboard(args.output_root / "screen_leaderboard.csv", screen_rows)

    ranked = sorted(
        screen_rows,
        key=lambda row: (
            -row["restricted_correct"],
            -row["all_digits_correct"],
            row["matched_atom_mse"],
            row["test_recon_loss"],
        ),
    )
    finalist_seeds = [row["seed"] for row in ranked[: args.n_finalists]]
    (args.output_root / "finalist_seeds.json").write_text(
        json.dumps(finalist_seeds, indent=2) + "\n"
    )
    stamp(f"finalists: {finalist_seeds}")

    final_rows = []
    for seed in finalist_seeds:
        checkpoint, config = run_trial(args, "final", seed, args.final_epochs)
        final_rows.append(evaluate_trial(
            scorer, checkpoint, config, "final", seed, args.final_epochs
        ))
        write_leaderboard(args.output_root / "final_leaderboard.csv", final_rows)

    best = sorted(
        final_rows,
        key=lambda row: (
            -row["restricted_correct"],
            -row["all_digits_correct"],
            row["matched_atom_mse"],
            row["test_recon_loss"],
        ),
    )[0]
    (args.output_root / "best_run.json").write_text(json.dumps(best, indent=2) + "\n")
    stamp(f"complete; best seed={best['seed']} checkpoint={best['checkpoint']}")


if __name__ == "__main__":
    main()
