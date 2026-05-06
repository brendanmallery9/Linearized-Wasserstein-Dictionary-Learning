from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from common import REPO_ROOT, stream_command, timestamp, write_json


HEITZ_TRIALS = [
    {"gamma": 0.5, "sinkhorn_iters": 5, "max_optim_iter": 500},
    {"gamma": 0.5, "sinkhorn_iters": 25, "max_optim_iter": 500},
    {"gamma": 2.0, "sinkhorn_iters": 5, "max_optim_iter": 500},
    {"gamma": 2.0, "sinkhorn_iters": 25, "max_optim_iter": 500},
]


def label_float(value: float) -> str:
    return str(value).replace(".", "p")


def run_or_exit(name: str, cmd: list[object], *, log_path: Path, cwd: Path = REPO_ROOT) -> float:
    print(f"\n=== {name} ===", flush=True)
    returncode, elapsed = stream_command(cmd, cwd=cwd, log_path=log_path)
    if returncode != 0:
        print(f"{name} failed with return code {returncode}", file=sys.stderr, flush=True)
        sys.exit(returncode)
    return elapsed


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def run_evaluation(
    *,
    run_dir: Path,
    metadata_path: Path,
    method_kind: str,
    method_label: str,
    method_run_dir: Path,
    device: str,
    output_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        REPO_ROOT / "experiments" / "timed_comparison" / "evaluate_mnist_ot_reconstruction.py",
        "--comparison-dir", run_dir,
        "--metadata-path", metadata_path,
        "--output", output_path,
        "--methods", method_kind,
    ]
    if method_kind == "heitz":
        cmd.extend(["--heitz-run-dir", method_run_dir, "--heitz-label", method_label])
    else:
        cmd.extend([
            "--ours-run-dir", method_run_dir,
            "--ours-label", method_label,
            "--device", device,
        ])

    run_or_exit(f"Evaluating {method_label}", cmd, log_path=log_path)
    payload = read_json(output_path)
    return payload["per_method"][method_label]


def write_summary_table(df: pd.DataFrame, csv_path: Path, pickle_path: Path, png_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    df.to_pickle(pickle_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display_df = df.copy()
    display_df["loss"] = display_df["loss"].map(lambda value: f"{value:.6g}")
    display_df["clock_time_at_termination"] = display_df["clock_time_at_termination"].map(
        lambda value: f"{value:.1f}"
    )
    for column in ("map_prep_elapsed_seconds", "train_elapsed_seconds"):
        if column in display_df:
            display_df[column] = display_df[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.1f}"
            )

    fig_height = max(2.2, 0.42 * (len(display_df) + 1))
    fig, ax = plt.subplots(figsize=(11, fig_height), constrained_layout=True)
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.25)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8eef5")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f6f8fb")
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MNIST Heitz/OT-SAE sweep with shared evaluation.")
    parser.add_argument("--run-dir", type=Path,
                        default=REPO_ROOT / "experiments" / "results" / f"mnist_heitz_sweep_{timestamp()}")
    parser.add_argument("--otsae-devices", nargs="+", choices=["cpu", "cuda"], default=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-digit", type=int, default=10)
    parser.add_argument("--base-supp-size", type=int, default=400)
    parser.add_argument("--atoms", type=int, default=10)
    parser.add_argument("--lista-steps", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eps", type=float, default=0.025)
    parser.add_argument("--c", type=float, default=0.0001)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-elapsed-seconds", type=float, default=3600.0)
    parser.add_argument("--plateau-window", type=int, default=10)
    parser.add_argument("--plateau-min-delta", type=float, default=1e-5)
    parser.add_argument("--heitz-loss-type", type=int, default=2)
    parser.add_argument("--heitz-scale-dict-factor", type=float, default=100.0)
    parser.add_argument("--heitz-avx", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--heitz-with-openmp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    log_dir = run_dir / "logs"
    eval_dir = run_dir / "evaluations"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    shared_png_dir = run_dir / "shared_mnist_png"
    metadata_path = shared_png_dir / "metadata.json"

    manifest = {
        "max_per_digit": args.max_per_digit,
        "base_supp_size": args.base_supp_size,
        "atoms": args.atoms,
        "lista_steps": args.lista_steps,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eps": args.eps,
        "c": args.c,
        "lr": args.lr,
        "seed": args.seed,
        "otsae_devices": args.otsae_devices,
        "otsae_map_timing": {
            "mode": "force_per_device",
            "description": (
                "Each OT-SAE device run recomputes MNIST OT maps in its own run "
                "directory; SAE clock_time_at_termination includes map prep time."
            ),
        },
        "stopping": {
            "max_elapsed_seconds": args.max_elapsed_seconds,
            "plateau_window": args.plateau_window,
            "plateau_min_delta": args.plateau_min_delta,
        },
        "heitz_loss_type": args.heitz_loss_type,
        "heitz_scale_dict_factor": args.heitz_scale_dict_factor,
        "heitz_avx": args.heitz_avx,
        "heitz_with_openmp": args.heitz_with_openmp,
        "heitz_trials": HEITZ_TRIALS,
    }
    write_json(run_dir / "sweep_manifest.json", manifest)

    run_or_exit(
        "Preparing shared MNIST PNGs",
        [
            sys.executable,
            REPO_ROOT / "experiments" / "timed_comparison" / "prepare_mnist_png.py",
            "--output-dir", shared_png_dir,
            "--mnist-root", REPO_ROOT / "mnist_raw",
            "--max-per-digit", args.max_per_digit,
            "--force",
        ],
        log_path=log_dir / "prepare_mnist_png.log",
    )

    rows: list[dict[str, Any]] = []

    for trial in HEITZ_TRIALS:
        gamma = trial["gamma"]
        sinkhorn_iters = trial["sinkhorn_iters"]
        max_optim_iter = trial["max_optim_iter"]
        method_label = (
            f"heitz_gamma{label_float(gamma)}_sink{sinkhorn_iters}_optim{max_optim_iter}"
        )
        method_dir = run_dir / method_label
        cmd = [
            sys.executable,
            REPO_ROOT / "experiments" / "timed_comparison" / "run_heitz_wdl.py",
            "--run-dir", method_dir,
            "--input-dir", shared_png_dir / "all",
            "--k", args.atoms,
            "--loss-type", args.heitz_loss_type,
            "--sinkhorn-iters", sinkhorn_iters,
            "--max-optim-iter", max_optim_iter,
            "--gamma", gamma,
            "--scale-dict-factor", args.heitz_scale_dict_factor,
            "--avx", args.heitz_avx,
            "--max-elapsed-seconds", args.max_elapsed_seconds,
            "--plateau-window", args.plateau_window,
            "--plateau-min-delta", args.plateau_min_delta,
            "--deterministic",
        ]
        if args.heitz_with_openmp:
            cmd.append("--with-openmp")
        run_or_exit(f"Running {method_label}", cmd, log_path=log_dir / f"{method_label}.log")

        eval_summary = run_evaluation(
            run_dir=run_dir,
            metadata_path=metadata_path,
            method_kind="heitz",
            method_label=method_label,
            method_run_dir=method_dir,
            device="cpu",
            output_path=eval_dir / f"{method_label}.json",
            log_path=log_dir / f"evaluate_{method_label}.log",
        )
        summary = read_json(method_dir / "summary.json")
        rows.append({
            "method": method_label,
            "loss": eval_summary["mean_w2_squared"],
            "epochs_at_termination": summary.get("termination_iteration"),
            "clock_time_at_termination": summary.get(
                "termination_elapsed_seconds", summary.get("elapsed_seconds")
            ),
            "map_prep_elapsed_seconds": None,
            "train_elapsed_seconds": None,
        })

    for device in args.otsae_devices:
        method_label = f"mnist_ot_sae_{device}"
        method_dir = run_dir / method_label
        method_ot_dir = method_dir / "data" / "mnist_ot"
        run_or_exit(
            f"Running {method_label}",
            [
                sys.executable,
                REPO_ROOT / "experiments" / "timed_comparison" / "run_mnist_sae_timed.py",
                "--run-dir", method_dir,
                "--data-dir", method_ot_dir,
                "--map-mode", "force",
                "--device", device,
                "--seed", args.seed,
                "--base-supp-size", args.base_supp_size,
                "--max-per-digit", args.max_per_digit,
                "--m", args.atoms,
                "--lista-steps", args.lista_steps,
                "--epochs", args.epochs,
                "--batch-size", args.batch_size,
                "--lr", args.lr,
                "--eps", args.eps,
                "--c", args.c,
                "--max-elapsed-seconds", args.max_elapsed_seconds,
                "--plateau-window", args.plateau_window,
                "--plateau-min-delta", args.plateau_min_delta,
                "--force-output",
            ],
            log_path=log_dir / f"{method_label}.log",
        )

        eval_summary = run_evaluation(
            run_dir=run_dir,
            metadata_path=metadata_path,
            method_kind="ours",
            method_label=method_label,
            method_run_dir=method_dir,
            device=device,
            output_path=eval_dir / f"{method_label}.json",
            log_path=log_dir / f"evaluate_{method_label}.log",
        )
        summary = read_json(method_dir / "summary.json")
        rows.append({
            "method": method_label,
            "loss": eval_summary["mean_w2_squared"],
            "epochs_at_termination": summary.get("epochs_completed"),
            "clock_time_at_termination": summary.get(
                "termination_elapsed_seconds", summary.get("elapsed_seconds")
            ),
            "map_prep_elapsed_seconds": summary.get("map_prep_elapsed_seconds"),
            "train_elapsed_seconds": summary.get("train_elapsed_seconds"),
        })

    table_df = pd.DataFrame(rows, columns=[
        "method",
        "loss",
        "epochs_at_termination",
        "clock_time_at_termination",
        "map_prep_elapsed_seconds",
        "train_elapsed_seconds",
    ])
    write_summary_table(
        table_df,
        run_dir / "summary_table.csv",
        run_dir / "summary_table.pkl",
        run_dir / "summary_table.png",
    )
    write_json(run_dir / "summary_table.json", rows)
    print(f"\nSweep complete: {run_dir}", flush=True)
    print(f"Summary table CSV: {run_dir / 'summary_table.csv'}", flush=True)
    print(f"Summary table PNG: {run_dir / 'summary_table.png'}", flush=True)


if __name__ == "__main__":
    main()
