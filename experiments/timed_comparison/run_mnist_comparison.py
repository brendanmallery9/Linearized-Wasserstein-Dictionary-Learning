from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from common import REPO_ROOT, stream_command, timestamp, write_json


PRESETS = {
    "smoke": {
        "max_per_digit": 1,
        "base_supp_size": 32,
        "atoms": 3,
        "lista_steps": 2,
        "epochs": 1,
        "batch_size": 8,
        "heitz_sinkhorn_iters": 1,
        "heitz_max_optim_iter": 1,
    },
    "local": {
        "max_per_digit": 10,
        "base_supp_size": 400,
        "atoms": 10,
        "lista_steps": 20,
        "epochs": 500,
        "batch_size": 256,
        "heitz_sinkhorn_iters": 25,
        "heitz_max_optim_iter": 50,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MNIST timed comparison harness.")
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "experiments" / "results" / f"mnist_comparison_{timestamp()}")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--only", choices=["both", "heitz", "ours"], default="both")
    parser.add_argument("--order", choices=["heitz-first", "ours-first"], default="heitz-first")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="mps")
    parser.add_argument("--max-per-digit", type=int, default=None)
    parser.add_argument("--base-supp-size", type=int, default=None)
    parser.add_argument("--atoms", type=int, default=None)
    parser.add_argument("--lista-steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eps", type=float, default=0.025)
    parser.add_argument("--c", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heitz-sinkhorn-iters", type=int, default=None)
    parser.add_argument("--heitz-max-optim-iter", type=int, default=None)
    parser.add_argument("--heitz-loss-type", type=int, default=2)
    parser.add_argument("--heitz-gamma", type=float, default=2.0)
    parser.add_argument("--heitz-scale-dict-factor", type=float, default=100.0)
    parser.add_argument("--heitz-avx", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--skip-shared-eval", action="store_true",
                        help="Skip shared OT reconstruction evaluation after both methods finish")
    return parser.parse_args()


def choose(args: argparse.Namespace, key: str):
    value = getattr(args, key.replace("-", "_"), None)
    if value is not None:
        return value
    return PRESETS[args.preset][key.replace("-", "_")]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def generate_loss_plot(run_dir: Path) -> dict[str, str | None]:
    """Plot the recorded method objectives against wall-clock time."""
    heitz_history = read_jsonl(run_dir / "heitz_wdl" / "history.jsonl")
    ours_history = read_jsonl(run_dir / "mnist_ot_sae" / "history.jsonl")

    heitz_points = [
        (row["elapsed_seconds"], row["loss"])
        for row in heitz_history
        if row.get("event") == "loss_eval" and "elapsed_seconds" in row and "loss" in row
    ]
    ours_points = [
        (row["elapsed_seconds"], row["train_loss"])
        for row in ours_history
        if row.get("event") == "epoch" and "elapsed_seconds" in row and "train_loss" in row
    ]
    map_prep_seconds = next(
        (
            float(row.get("elapsed_seconds", 0.0))
            for row in ours_history
            if row.get("event") == "map_prep"
        ),
        None,
    )

    plot_path = run_dir / "loss_vs_wall_time.png"
    data_path = run_dir / "loss_vs_wall_time.json"
    payload = {
        "metric_note": (
            "Curves use each method's recorded training/objective loss. "
            "MNIST OT-SAE epoch times include map-prep offset, so its finite "
            "loss curve starts after map computation."
        ),
        "plot_path": str(plot_path),
        "heitz_wdl": [
            {"elapsed_seconds": t, "loss": loss}
            for t, loss in heitz_points
        ],
        "mnist_ot_sae": [
            {"elapsed_seconds": t, "loss": loss}
            for t, loss in ours_points
        ],
        "mnist_ot_sae_map_prep_seconds": map_prep_seconds,
    }
    write_json(data_path, payload)

    if not heitz_points and not ours_points:
        return {"plot_path": None, "data_path": str(data_path)}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    if heitz_points:
        x, y = zip(*heitz_points)
        ax.plot(x, y, marker="o", markersize=3, linewidth=1.6, label="Heitz WDL")
    if ours_points:
        x, y = zip(*ours_points)
        ax.plot(x, y, marker="o", markersize=2, linewidth=1.8, label="MNIST OT-SAE")
    if map_prep_seconds is not None and map_prep_seconds > 0:
        ax.axvspan(0, map_prep_seconds, color="0.9", alpha=0.8, label="OT-SAE map prep")
        ax.axvline(map_prep_seconds, color="0.45", linestyle="--", linewidth=1)

    ax.set_xlabel("Wall time (seconds)")
    ax.set_ylabel("Recorded objective loss")
    ax.set_title("Wall Time vs. Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return {"plot_path": str(plot_path), "data_path": str(data_path)}


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "preset": args.preset,
        "max_per_digit": choose(args, "max_per_digit"),
        "base_supp_size": choose(args, "base_supp_size"),
        "atoms": choose(args, "atoms"),
        "lista_steps": choose(args, "lista_steps"),
        "epochs": choose(args, "epochs"),
        "batch_size": choose(args, "batch_size"),
        "eps": args.eps,
        "c": args.c,
        "seed": args.seed,
        "device": args.device,
        "heitz_sinkhorn_iters": choose(args, "heitz_sinkhorn_iters"),
        "heitz_max_optim_iter": choose(args, "heitz_max_optim_iter"),
        "heitz_loss_type": args.heitz_loss_type,
        "heitz_gamma": args.heitz_gamma,
        "heitz_scale_dict_factor": args.heitz_scale_dict_factor,
        "heitz_avx": args.heitz_avx,
    }
    write_json(run_dir / "manifest.json", config)

    heitz_cmd = [
        sys.executable,
        REPO_ROOT / "experiments" / "timed_comparison" / "run_heitz_wdl.py",
        "--run-dir", run_dir / "heitz_wdl",
        "--max-per-digit", config["max_per_digit"],
        "--k", config["atoms"],
        "--loss-type", config["heitz_loss_type"],
        "--sinkhorn-iters", config["heitz_sinkhorn_iters"],
        "--max-optim-iter", config["heitz_max_optim_iter"],
        "--gamma", config["heitz_gamma"],
        "--scale-dict-factor", config["heitz_scale_dict_factor"],
        "--avx", config["heitz_avx"],
        "--force-data",
    ]
    ours_cmd = [
        sys.executable,
        REPO_ROOT / "experiments" / "timed_comparison" / "run_mnist_sae_timed.py",
        "--run-dir", run_dir / "mnist_ot_sae",
        "--map-mode", "force",
        "--device", config["device"],
        "--seed", config["seed"],
        "--base-supp-size", config["base_supp_size"],
        "--max-per-digit", config["max_per_digit"],
        "--m", config["atoms"],
        "--lista-steps", config["lista_steps"],
        "--epochs", config["epochs"],
        "--batch-size", config["batch_size"],
        "--eps", config["eps"],
        "--c", config["c"],
        "--force-output",
    ]

    jobs: list[tuple[str, Path, list[object]]] = []
    if args.only in {"both", "heitz"}:
        jobs.append(("heitz", run_dir / "heitz_wdl", heitz_cmd))
    if args.only in {"both", "ours"}:
        jobs.append(("ours", run_dir / "mnist_ot_sae", ours_cmd))
    if args.order == "ours-first":
        jobs = list(reversed(jobs))

    results = []
    for name, method_dir, cmd in jobs:
        print(f"\n=== Running {name} ===", flush=True)
        returncode, elapsed = stream_command(
            cmd,
            cwd=REPO_ROOT,
            log_path=run_dir / f"{name}_wrapper.log",
        )
        method_summary_path = method_dir / "summary.json"
        method_summary = None
        if method_summary_path.exists():
            method_summary = json.loads(method_summary_path.read_text())
        results.append({
            "name": name,
            "returncode": returncode,
            "wrapper_elapsed_seconds": elapsed,
            "method_elapsed_seconds": (
                method_summary.get("elapsed_seconds") if method_summary else None
            ),
            "summary_path": str(method_summary_path),
        })
        if returncode != 0:
            write_json(run_dir / "comparison_summary.json", {"config": config, "results": results})
            sys.exit(returncode)

    shared_eval = None
    if (
        not args.skip_shared_eval
        and (run_dir / "heitz_wdl" / "summary.json").exists()
        and (run_dir / "mnist_ot_sae" / "summary.json").exists()
    ):
        print("\n=== Running shared OT reconstruction evaluation ===", flush=True)
        eval_device = config["device"] if config["device"] in {"cuda", "mps"} else "cpu"
        shared_eval_path = run_dir / "shared_ot_reconstruction.json"
        eval_cmd = [
            sys.executable,
            REPO_ROOT / "experiments" / "timed_comparison" / "evaluate_mnist_ot_reconstruction.py",
            "--comparison-dir", run_dir,
            "--output", shared_eval_path,
            "--device", eval_device,
        ]
        returncode, elapsed = stream_command(
            eval_cmd,
            cwd=REPO_ROOT,
            log_path=run_dir / "shared_eval_wrapper.log",
        )
        shared_eval = {
            "returncode": returncode,
            "wrapper_elapsed_seconds": elapsed,
            "output_path": str(shared_eval_path),
        }
        if returncode != 0:
            write_json(
                run_dir / "comparison_summary.json",
                {"config": config, "results": results, "shared_eval": shared_eval},
            )
            sys.exit(returncode)

    loss_plot = generate_loss_plot(run_dir)

    write_json(
        run_dir / "comparison_summary.json",
        {
            "config": config,
            "results": results,
            "shared_eval": shared_eval,
            "loss_plot": loss_plot,
        },
    )
    print(f"\nComparison outputs: {run_dir}")
    if loss_plot.get("plot_path"):
        print(f"Loss plot: {loss_plot['plot_path']}")


if __name__ == "__main__":
    main()
