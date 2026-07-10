from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

REPO_ROOT = SCRIPT_DIR.parents[1]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


from run_timing_suite import (
    HEITZ_GAMMAS,
    HSI_LEGACY_SAE,
    as_points,
    image_measure_1d,
    image_measure_2d,
    label_float,
    mnist_w2_errors,
    pavia_w2_errors,
    prepare_mnist_maps,
    prepare_mnist_subset,
    prepare_pavia_maps,
    prepare_pavia_subset,
    resolve_device,
    run_heitz_trial,
    save_table,
    set_seeds,
    train_ebcm,
    train_generic_sae,
    w2_squared,
    write_pavia_pngs,
)


MARK2_EPSILON = 0.025
DEFAULT_SAMPLE_SIZES = [100, 1000, 10000]
DEFAULT_DURATION_SECONDS = 1000.0
DEFAULT_PAVIA_METHODS = ["transport_map", "ebcm", "heitz"]
DEFAULT_MNIST_METHODS = ["ebcm", "heitz"]


def parse_float_list(values: list[str] | None, default: list[float]) -> list[float]:
    if not values:
        return list(default)
    out: list[float] = []
    for value in values:
        for part in value.replace(",", " ").split():
            out.append(float(part))
    return out


def parse_int_list(values: list[str] | None, default: list[int]) -> list[int]:
    if not values:
        return list(default)
    out: list[int] = []
    for value in values:
        for part in value.replace(",", " ").split():
            out.append(int(part))
    return out


def mark2_args(args: argparse.Namespace, *, pavia_samples: int | None = None, mnist_samples: int | None = None) -> SimpleNamespace:
    if mnist_samples is not None and mnist_samples % 10 != 0:
        raise ValueError("MNIST sample sizes must be divisible by 10 for balanced digit subsets")
    return SimpleNamespace(
        seed=args.seed,
        force_cache=args.force_cache,
        force_outputs=args.force_outputs,
        progress_every=args.progress_every,
        atoms=args.atoms,
        top_k=args.top_k,
        lista_steps=args.lista_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        sparsity_coeff=args.sparsity_coeff,
        weight_decay=args.weight_decay,
        test_fraction=args.test_fraction,
        grid_side=args.grid_side,
        history_every=args.history_every,
        max_elapsed_seconds=args.duration_seconds,
        plateau_window=0,
        plateau_min_delta=0.0,
        pavia_cube_path=args.pavia_cube_path,
        pavia_samples=pavia_samples,
        pavia_support_size=args.pavia_support_size,
        mnist_digits=list(range(10)),
        mnist_max_per_digit=None if mnist_samples is None else mnist_samples // 10,
        base_supp_size=args.base_supp_size,
        hsi_architecture=args.hsi_architecture,
        heitz_loss_type=args.heitz_loss_type,
        heitz_scale_dict_factor=args.heitz_scale_dict_factor,
        heitz_max_optim_iter=args.heitz_max_optim_iter,
        heitz_avx=args.heitz_avx,
        heitz_with_openmp=args.heitz_with_openmp,
        heitz_sinkhorn_iters=args.heitz_sinkhorn_iters,
    )


def finite(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def trial_row(
    *,
    experiment: str,
    sample_size: int,
    method: str,
    variant: str,
    embedding_seconds: float,
    train_seconds: float,
    epochs_completed: int | None,
    iterations_completed: int | None,
    termination_reason: str | None,
    mean_w2_squared: float | None,
    embedded_recon_loss: float | None,
    history_path: Path | None,
    artifact: Path | str,
    epsilon: float | None = None,
    gamma: float | None = None,
    sinkhorn_iters: int | None = None,
    eval_status: str = "ok",
    eval_warning: str | None = None,
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "sample_size": sample_size,
        "method": method,
        "variant": variant,
        "epsilon": epsilon,
        "gamma": gamma,
        "sinkhorn_iters": sinkhorn_iters,
        "embedding_seconds": embedding_seconds,
        "train_seconds": train_seconds,
        "clock_time_seconds": embedding_seconds + train_seconds,
        "epochs_completed": epochs_completed,
        "iterations_completed": iterations_completed,
        "termination_reason": termination_reason,
        "mean_w2_squared": finite(mean_w2_squared),
        "embedded_recon_loss": finite(embedded_recon_loss),
        "eval_status": eval_status,
        "eval_warning": eval_warning,
        "history_path": str(history_path) if history_path is not None else None,
        "artifact": str(artifact),
    }


def safe_evaluate_heitz_outputs(
    *,
    run_dir: Path,
    target_measures: list[tuple[Any, Any]],
    kind: str,
    strict: bool,
) -> tuple[float | None, str, str | None]:
    errors = []
    missing = []
    for index, (target_points, target_masses) in enumerate(target_measures):
        fitting_path = run_dir / "outputs" / f"finalFitting_{index:03d}.png"
        if not fitting_path.exists():
            missing.append(index)
            continue
        if kind == "1d":
            recon_points, recon_masses = image_measure_1d(fitting_path)
        elif kind == "2d":
            recon_points, recon_masses = image_measure_2d(fitting_path)
        else:
            raise ValueError(kind)
        errors.append(w2_squared(target_points, target_masses, recon_points, recon_masses))

    output_count = len(list((run_dir / "outputs").glob("finalFitting_*.png")))
    if missing:
        preview = ",".join(str(index) for index in missing[:10])
        suffix = "" if len(missing) <= 10 else f",...,+{len(missing) - 10} more"
        warning = (
            f"Missing {len(missing)}/{len(target_measures)} Heitz reconstruction PNGs "
            f"(first missing: {preview}{suffix}; output_count={output_count})"
        )
        if strict:
            raise FileNotFoundError(warning)
        print(f"[warning] {run_dir.name}: {warning}", flush=True)
        if not errors:
            return None, "missing_outputs", warning
        return float(sum(errors) / len(errors)), "partial_outputs", warning

    return float(sum(errors) / len(errors)) if errors else None, "ok", None


def run_pavia_size(
    args: argparse.Namespace,
    *,
    sample_size: int,
    run_dir: Path,
    cache_dir: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    print(f"\n=== Mark 2 Pavia 1D: n={sample_size} ===", flush=True)
    suite_args = mark2_args(args, pavia_samples=sample_size)
    exp_dir = run_dir / "pavia1d" / f"n{sample_size}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_dir = exp_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []

    data = prepare_pavia_subset(suite_args, cache_dir)

    if "transport_map" in args.pavia_methods:
        maps, embed_seconds, maps_path = prepare_pavia_maps(
            data,
            cache_dir,
            method="hsi_1d",
            eps=None,
            force=args.force_cache,
            progress_every=args.progress_every,
        )
        history_path = exp_dir / "transport_map_history.jsonl"
        t0 = time.monotonic()
        _, metrics, recon = train_generic_sae(
            maps.reshape(maps.shape[0], maps.shape[1]),
            architecture=args.hsi_architecture,
            hidden_dim=args.atoms,
            top_k=args.top_k,
            lista_steps=args.lista_steps,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=HSI_LEGACY_SAE["lr"],
            sparsity_coeff=args.sparsity_coeff,
            weight_decay=HSI_LEGACY_SAE["weight_decay"],
            seed=args.seed,
            device=device,
            history_path=history_path,
            run_name=f"mark2_pavia_n{sample_size}_transport",
            history_time_offset=0.0,
            max_elapsed_seconds=args.duration_seconds,
            plateau_window=0,
            plateau_min_delta=0.0,
            test_fraction=args.test_fraction,
            scheduler=HSI_LEGACY_SAE["scheduler"],
            grad_clip=HSI_LEGACY_SAE["grad_clip"],
            sparsity_mode=HSI_LEGACY_SAE["sparsity_mode"],
        )
        train_seconds = time.monotonic() - t0
        recon_maps = recon.reshape_as(maps)
        w2 = pavia_w2_errors(data, recon_maps)
        rows.append(trial_row(
            experiment="pavia1d",
            sample_size=sample_size,
            method="transport_map",
            variant=args.hsi_architecture,
            epsilon=None,
            embedding_seconds=embed_seconds,
            train_seconds=train_seconds,
            epochs_completed=metrics["epochs_completed"],
            iterations_completed=None,
            termination_reason=metrics["termination_reason"],
            mean_w2_squared=w2,
            embedded_recon_loss=metrics["all_recon_mse"],
            history_path=history_path,
            artifact=maps_path,
        ))

    if "ebcm" in args.pavia_methods:
        maps, embed_seconds, maps_path = prepare_pavia_maps(
            data,
            cache_dir,
            method="ebcm",
            eps=MARK2_EPSILON,
            force=args.force_cache,
            progress_every=args.progress_every,
        )
        history_path = exp_dir / f"ebcm_eps{label_float(MARK2_EPSILON)}_history.jsonl"
        t0 = time.monotonic()
        _, metrics, recon = train_ebcm(
            as_points(data["source_points"]),
            maps,
            eps=MARK2_EPSILON,
            grid_points=as_points(data["target_points"]),
            args=suite_args,
            device=device,
            history_path=history_path,
            run_name=f"mark2_pavia_n{sample_size}_ebcm_eps{MARK2_EPSILON:g}",
            history_time_offset=0.0,
        )
        train_seconds = time.monotonic() - t0
        w2 = pavia_w2_errors(data, recon)
        rows.append(trial_row(
            experiment="pavia1d",
            sample_size=sample_size,
            method="ebcm",
            variant="entropic_displacement",
            epsilon=MARK2_EPSILON,
            embedding_seconds=embed_seconds,
            train_seconds=train_seconds,
            epochs_completed=metrics["epochs_completed"],
            iterations_completed=None,
            termination_reason=metrics["termination_reason"],
            mean_w2_squared=w2,
            embedded_recon_loss=metrics["train_recon_loss"],
            history_path=history_path,
            artifact=maps_path,
        ))

    if "heitz" in args.pavia_methods and not args.skip_heitz:
        image_dir, image_paths = write_pavia_pngs(data, exp_dir / "heitz_input", force=args.force_outputs)
        target_measures = [image_measure_1d(path) for path in sorted(image_paths, key=lambda path: path.name)]
        shared_source = exp_dir / "heitz_external" / "WassersteinDictionaryLearning"
        shared_build = exp_dir / "heitz_build"
        for gamma in args.heitz_gammas:
            method_dir = exp_dir / f"heitz_gamma{label_float(gamma)}_sink{args.heitz_sinkhorn_iters}"
            summary, wrapper_elapsed = run_heitz_trial(
                input_dir=image_dir,
                run_dir=method_dir,
                args=suite_args,
                gamma=gamma,
                sinkhorn_iters=args.heitz_sinkhorn_iters,
                source_dir=shared_source,
                build_dir=shared_build,
                log_dir=log_dir,
            )
            w2, eval_status, eval_warning = safe_evaluate_heitz_outputs(
                run_dir=method_dir,
                target_measures=target_measures,
                kind="1d",
                strict=args.strict_heitz_eval,
            )
            rows.append(trial_row(
                experiment="pavia1d",
                sample_size=sample_size,
                method="heitz",
                variant="wasserstein_dictionary_learning",
                gamma=gamma,
                sinkhorn_iters=args.heitz_sinkhorn_iters,
                embedding_seconds=0.0,
                train_seconds=summary.get("termination_elapsed_seconds", wrapper_elapsed),
                epochs_completed=None,
                iterations_completed=summary.get("termination_iteration"),
                termination_reason=summary.get("termination_reason"),
                mean_w2_squared=w2,
                embedded_recon_loss=None,
                history_path=method_dir / "history.jsonl",
                artifact=method_dir,
                eval_status=eval_status,
                eval_warning=eval_warning,
            ))
    return rows


def run_mnist_size(
    args: argparse.Namespace,
    *,
    sample_size: int,
    run_dir: Path,
    cache_dir: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    print(f"\n=== Mark 2 MNIST 2D: n={sample_size} ===", flush=True)
    suite_args = mark2_args(args, mnist_samples=sample_size)
    exp_dir = run_dir / "mnist" / f"n{sample_size}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_dir = exp_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []

    data = prepare_mnist_subset(suite_args, cache_dir)
    if "ebcm" in args.mnist_methods:
        maps, embed_seconds, maps_path = prepare_mnist_maps(
            data,
            cache_dir,
            eps=MARK2_EPSILON,
            force=args.force_cache,
            progress_every=args.progress_every,
        )
        history_path = exp_dir / f"ebcm_eps{label_float(MARK2_EPSILON)}_history.jsonl"
        t0 = time.monotonic()
        _, metrics, recon = train_ebcm(
            as_points(data["source_points"]),
            maps,
            eps=MARK2_EPSILON,
            grid_points=None,
            args=suite_args,
            device=device,
            history_path=history_path,
            run_name=f"mark2_mnist_n{sample_size}_ebcm_eps{MARK2_EPSILON:g}",
            history_time_offset=0.0,
        )
        train_seconds = time.monotonic() - t0
        w2 = mnist_w2_errors(data, recon)
        rows.append(trial_row(
            experiment="mnist",
            sample_size=sample_size,
            method="ebcm",
            variant="entropic_displacement",
            epsilon=MARK2_EPSILON,
            embedding_seconds=embed_seconds,
            train_seconds=train_seconds,
            epochs_completed=metrics["epochs_completed"],
            iterations_completed=None,
            termination_reason=metrics["termination_reason"],
            mean_w2_squared=w2,
            embedded_recon_loss=metrics["train_recon_loss"],
            history_path=history_path,
            artifact=maps_path,
        ))

    if "heitz" in args.mnist_methods and not args.skip_heitz:
        image_dir = Path(data["image_dir"])
        heitz_records = sorted(data["records"], key=lambda record: record["filename"])
        target_measures = [image_measure_2d(Path(record["image_path"])) for record in heitz_records]
        shared_source = exp_dir / "heitz_external" / "WassersteinDictionaryLearning"
        shared_build = exp_dir / "heitz_build"
        for gamma in args.heitz_gammas:
            method_dir = exp_dir / f"heitz_gamma{label_float(gamma)}_sink{args.heitz_sinkhorn_iters}"
            summary, wrapper_elapsed = run_heitz_trial(
                input_dir=image_dir,
                run_dir=method_dir,
                args=suite_args,
                gamma=gamma,
                sinkhorn_iters=args.heitz_sinkhorn_iters,
                source_dir=shared_source,
                build_dir=shared_build,
                log_dir=log_dir,
            )
            w2, eval_status, eval_warning = safe_evaluate_heitz_outputs(
                run_dir=method_dir,
                target_measures=target_measures,
                kind="2d",
                strict=args.strict_heitz_eval,
            )
            rows.append(trial_row(
                experiment="mnist",
                sample_size=sample_size,
                method="heitz",
                variant="wasserstein_dictionary_learning",
                gamma=gamma,
                sinkhorn_iters=args.heitz_sinkhorn_iters,
                embedding_seconds=0.0,
                train_seconds=summary.get("termination_elapsed_seconds", wrapper_elapsed),
                epochs_completed=None,
                iterations_completed=summary.get("termination_iteration"),
                termination_reason=summary.get("termination_reason"),
                mean_w2_squared=w2,
                embedded_recon_loss=None,
                history_path=method_dir / "history.jsonl",
                artifact=method_dir,
                eval_status=eval_status,
                eval_warning=eval_warning,
            ))
    return rows


def read_loss_history(row: dict[str, Any]) -> list[dict[str, Any]]:
    path_value = row.get("history_path")
    if not path_value:
        return []
    path = Path(path_value)
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            event = json.loads(line)
            event_name = event.get("event")
            if event_name == "epoch" and event.get("train_loss") is not None:
                out.append({
                    "experiment": row["experiment"],
                    "sample_size": row["sample_size"],
                    "method": row["method"],
                    "gamma": row.get("gamma"),
                    "epsilon": row.get("epsilon"),
                    "elapsed_seconds": event.get("elapsed_seconds"),
                    "step": event.get("epoch"),
                    "loss": event.get("train_loss"),
                })
            elif event_name == "loss_eval" and event.get("loss") is not None:
                out.append({
                    "experiment": row["experiment"],
                    "sample_size": row["sample_size"],
                    "method": row["method"],
                    "gamma": row.get("gamma"),
                    "epsilon": row.get("epsilon"),
                    "elapsed_seconds": event.get("elapsed_seconds"),
                    "step": event.get("eval_index"),
                    "loss": event.get("loss"),
                })
    return out


def make_loss_label(row: pd.Series) -> str:
    method = row["method"]
    if method == "heitz":
        return f"n={row['sample_size']} heitz gamma={row['gamma']:g}"
    if method == "ebcm":
        return f"n={row['sample_size']} EBCM eps={row['epsilon']:g}"
    return f"n={row['sample_size']} {method}"


def write_loss_plots(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, str]:
    histories = []
    for row in rows:
        histories.extend(read_loss_history(row))
    history_path = out_dir / "loss_history_all.csv"
    if histories:
        history_df = pd.DataFrame(histories)
        history_df.to_csv(history_path, index=False)
    else:
        history_df = pd.DataFrame(columns=["experiment", "sample_size", "method", "elapsed_seconds", "loss"])
        history_df.to_csv(history_path, index=False)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        warning_path = out_dir / "loss_plot_warning.txt"
        warning_path.write_text(f"Could not import matplotlib: {exc}\n", encoding="utf-8")
        return {"history_csv": str(history_path), "warning": str(warning_path)}

    outputs: dict[str, str] = {"history_csv": str(history_path)}
    if history_df.empty:
        return outputs

    for experiment in sorted(history_df["experiment"].unique()):
        fig, ax = plt.subplots(figsize=(11, 6))
        exp_df = history_df[history_df["experiment"] == experiment]
        for _, group in exp_df.groupby(["sample_size", "method", "gamma", "epsilon"], dropna=False):
            group = group.sort_values("elapsed_seconds")
            label = make_loss_label(group.iloc[0])
            ax.plot(group["elapsed_seconds"], group["loss"], linewidth=1.8, label=label)
        ax.set_title(f"Mark 2 {experiment} loss over time")
        ax.set_xlabel("elapsed seconds")
        ax.set_ylabel("training/loss-eval loss")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        plot_path = out_dir / f"loss_curves_{experiment}.png"
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        outputs[f"{experiment}_png"] = str(plot_path)
    return outputs


def write_current_outputs(rows: list[dict[str, Any]], run_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    table_paths = save_table(rows, run_dir, "timing_table_mark2")
    plot_paths = write_loss_plots(rows, run_dir)
    write_json(run_dir / "summary.json", {
        "run_dir": str(run_dir),
        "num_rows": len(rows),
        "tables": table_paths,
        "plots": plot_paths,
        "partial": True,
    })
    return table_paths, plot_paths


def cleanup_results_only(run_dir: Path, cache_dir: Path) -> None:
    """Keep consolidated metrics/loss outputs and remove bulky intermediates."""
    for child in ("pavia1d", "mnist"):
        shutil.rmtree(run_dir / child, ignore_errors=True)
    if cache_dir == run_dir / "_cache" or run_dir in cache_dir.parents:
        shutil.rmtree(cache_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mark 2 fixed-duration timing suite.")
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "experiments" / "results" / f"timing_suite_mark2_{timestamp()}")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "experiments" / "results" / "timing_cache_mark2")
    parser.add_argument("--experiment", choices=["all", "pavia1d", "mnist"], default="all")
    parser.add_argument("--sample-sizes", nargs="*", default=None)
    parser.add_argument("--duration-seconds", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--requested-timing-run", action="store_true",
                        help="Use the requested 1000-sample, 2000-second, 100000-iteration Pavia+MNIST settings.")
    parser.add_argument("--results-only", action="store_true",
                        help="After writing consolidated tables/loss histories, remove caches and bulky run artifacts.")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--force-outputs", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--skip-heitz", action="store_true")
    parser.add_argument("--strict-heitz-eval", action="store_true",
                        help="Abort when a Heitz run does not produce every finalFitting PNG")

    parser.add_argument("--atoms", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--lista-steps", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sparsity-coeff", type=float, default=HSI_LEGACY_SAE["sparsity_coeff"])
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--grid-side", type=int, default=64)
    parser.add_argument("--history-every", type=int, default=1)

    parser.add_argument("--pavia-cube-path", type=Path, default=REPO_ROOT / "datasets" / "hsi_data" / "pavia" / "data" / "pavia_cube.pt")
    parser.add_argument("--pavia-support-size", type=int, default=102)
    parser.add_argument("--hsi-architecture", choices=["JumpReLU_monotone", "TopKAE_monotone"], default="JumpReLU_monotone")
    parser.add_argument("--base-supp-size", type=int, default=400)

    parser.add_argument("--pavia-methods", nargs="*", choices=DEFAULT_PAVIA_METHODS, default=None,
                        help="Mark 2 Pavia methods to run.")
    parser.add_argument("--mnist-methods", nargs="*", choices=DEFAULT_MNIST_METHODS, default=None,
                        help="Mark 2 MNIST methods to run.")
    parser.add_argument("--heitz-gammas", nargs="*", default=None)
    parser.add_argument("--heitz-sinkhorn-iters", type=int, default=5)
    parser.add_argument("--heitz-max-optim-iter", type=int, default=1_000_000)
    parser.add_argument("--heitz-loss-type", type=int, default=2)
    parser.add_argument("--heitz-scale-dict-factor", type=float, default=100.0)
    parser.add_argument("--heitz-avx", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--heitz-with-openmp", action="store_true")
    args = parser.parse_args()

    args.sample_sizes = parse_int_list(args.sample_sizes, DEFAULT_SAMPLE_SIZES)
    args.pavia_methods = list(DEFAULT_PAVIA_METHODS if args.pavia_methods is None else args.pavia_methods)
    args.mnist_methods = list(DEFAULT_MNIST_METHODS if args.mnist_methods is None else args.mnist_methods)
    args.heitz_gammas = parse_float_list(args.heitz_gammas, HEITZ_GAMMAS)
    if args.requested_timing_run:
        args.experiment = "all"
        args.sample_sizes = [1000]
        args.duration_seconds = 2000.0
        args.epochs = 100_000
        args.heitz_max_optim_iter = 100_000
        args.pavia_methods = ["transport_map", "heitz"]
        args.mnist_methods = ["ebcm", "heitz"]
        args.results_only = True
        args.cache_dir = args.run_dir / "_cache"
    return args


def main() -> None:
    args = parse_args()
    if not args.skip_heitz and shutil.which("cmake") is None:
        raise RuntimeError("Heitz trials require cmake on PATH. Install cmake or pass --skip-heitz.")
    set_seeds(args.seed)
    device = resolve_device(args.device)
    run_dir = args.run_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = vars(args).copy()
    manifest["device_resolved"] = str(device)
    manifest["epsilon"] = MARK2_EPSILON
    manifest["pavia_methods"] = args.pavia_methods
    manifest["mnist_methods"] = args.mnist_methods
    manifest["results_only"] = args.results_only
    manifest["stopping_rule"] = {
        "type": "fixed_training_duration",
        "duration_seconds": args.duration_seconds,
        "plateau_stopping": False,
        "max_epochs": args.epochs,
        "heitz_max_optim_iter": args.heitz_max_optim_iter,
    }
    write_json(run_dir / "manifest.json", manifest)

    rows: list[dict[str, Any]] = []
    if args.experiment in {"all", "pavia1d"}:
        for sample_size in args.sample_sizes:
            rows.extend(run_pavia_size(args, sample_size=sample_size, run_dir=run_dir, cache_dir=cache_dir, device=device))
            write_current_outputs(rows, run_dir)
    if args.experiment in {"all", "mnist"}:
        for sample_size in args.sample_sizes:
            rows.extend(run_mnist_size(args, sample_size=sample_size, run_dir=run_dir, cache_dir=cache_dir, device=device))
            write_current_outputs(rows, run_dir)

    table_paths, plot_paths = write_current_outputs(rows, run_dir)
    write_json(run_dir / "summary.json", {
        "run_dir": str(run_dir),
        "cache_dir": str(cache_dir),
        "num_rows": len(rows),
        "tables": table_paths,
        "plots": plot_paths,
        "partial": False,
    })
    if args.results_only:
        cleanup_results_only(run_dir, cache_dir)
    print(f"\nMark 2 timing suite complete: {run_dir}", flush=True)
    print(f"Table: {table_paths['csv']}", flush=True)
    if "history_csv" in plot_paths:
        print(f"Loss history: {plot_paths['history_csv']}", flush=True)


if __name__ == "__main__":
    main()
