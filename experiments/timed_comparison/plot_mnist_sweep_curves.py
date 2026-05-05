from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import write_json


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


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


def heitz_label(run_dir: Path) -> str:
    summary = read_json(run_dir / "summary.json")
    params = summary.get("parameters", {})
    gamma = params.get("gamma")
    sinkhorn = params.get("sinkhorn_iters")
    max_optim = params.get("max_optim_iter")
    if gamma is not None and sinkhorn is not None and max_optim is not None:
        return f"Heitz gamma={gamma}, S={sinkhorn}, opt={max_optim}"
    return run_dir.name


def heitz_sort_key(run_dir: Path) -> tuple[float, int, int, str]:
    summary = read_json(run_dir / "summary.json")
    params = summary.get("parameters", {})
    return (
        float(params.get("gamma", 0.0)),
        int(params.get("sinkhorn_iters", 0)),
        int(params.get("max_optim_iter", 0)),
        run_dir.name,
    )


def collect_heitz_series(run_dir: Path) -> list[dict[str, Any]]:
    series = []
    for method_dir in sorted(run_dir.glob("heitz_*"), key=heitz_sort_key):
        history_path = method_dir / "history.jsonl"
        summary_path = method_dir / "summary.json"
        if not history_path.exists() or not summary_path.exists():
            continue
        points = [
            {"elapsed_seconds": row["elapsed_seconds"], "loss": row["loss"]}
            for row in read_jsonl(history_path)
            if row.get("event") == "loss_eval"
            and "elapsed_seconds" in row
            and "loss" in row
        ]
        if points:
            series.append({
                "label": heitz_label(method_dir),
                "method": "heitz_wdl",
                "run_dir": str(method_dir),
                "points": points,
            })
    return series


def collect_ours_series(run_dir: Path) -> tuple[dict[str, Any] | None, float | None]:
    method_dir = run_dir / "mnist_ot_sae"
    history = read_jsonl(method_dir / "history.jsonl")
    points = [
        {"elapsed_seconds": row["elapsed_seconds"], "loss": row["train_loss"]}
        for row in history
        if row.get("event") == "epoch"
        and "elapsed_seconds" in row
        and "train_loss" in row
    ]
    map_prep_seconds = next(
        (
            float(row.get("elapsed_seconds", 0.0))
            for row in history
            if row.get("event") == "map_prep"
        ),
        None,
    )
    if not points:
        return None, map_prep_seconds
    return {
        "label": "MNIST OT-SAE",
        "method": "mnist_ot_sae",
        "run_dir": str(method_dir),
        "points": points,
    }, map_prep_seconds


def plot_series(
    series: list[dict[str, Any]],
    *,
    output_path: Path,
    map_prep_seconds: float | None,
    log_y: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for item in series:
        points = item["points"]
        x = [point["elapsed_seconds"] for point in points]
        y = [point["loss"] for point in points]
        marker = "o" if item["method"] == "heitz_wdl" else "s"
        linewidth = 1.5 if item["method"] == "heitz_wdl" else 2.2
        markersize = 2.8 if item["method"] == "heitz_wdl" else 3.5
        ax.plot(x, y, marker=marker, markersize=markersize, linewidth=linewidth, label=item["label"])

    if map_prep_seconds is not None and map_prep_seconds > 0:
        ax.axvspan(0, map_prep_seconds, color="0.9", alpha=0.8, label="OT-SAE map prep")
        ax.axvline(map_prep_seconds, color="0.45", linestyle="--", linewidth=1)

    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Wall time (seconds)")
    ax.set_ylabel("Recorded objective loss")
    ax.set_title("MNIST Timed Comparison Sweep")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MNIST Heitz sweep loss curves.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--data-output", type=Path, default=None)
    parser.add_argument("--log-y", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_path = args.output or (run_dir / "loss_vs_wall_time_sweep.png")
    data_path = args.data_output or output_path.with_suffix(".json")

    heitz_series = collect_heitz_series(run_dir)
    ours_series, map_prep_seconds = collect_ours_series(run_dir)
    series = heitz_series + ([ours_series] if ours_series else [])
    if not series:
        raise RuntimeError(f"No history series found under {run_dir}")

    payload = {
        "metric_note": (
            "Curves use each method's recorded internal objective/training loss. "
            "MNIST OT-SAE epoch times include map-prep offset, so its finite "
            "loss curve starts after map computation."
        ),
        "run_dir": str(run_dir),
        "plot_path": str(output_path),
        "mnist_ot_sae_map_prep_seconds": map_prep_seconds,
        "series": series,
    }
    write_json(data_path, payload)
    plot_series(series, output_path=output_path, map_prep_seconds=map_prep_seconds, log_y=args.log_y)
    print(f"Wrote sweep loss plot to {output_path}")
    print(f"Wrote sweep loss data to {data_path}")


if __name__ == "__main__":
    main()
