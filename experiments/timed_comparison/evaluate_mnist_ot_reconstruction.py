from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import ot
import torch
from PIL import Image

from common import REPO_ROOT, write_json

MNIST_PIPELINE = REPO_ROOT / "mnist" / "pipeline"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MNIST_PIPELINE) not in sys.path:
    sys.path.insert(0, str(MNIST_PIPELINE))

from mnist_ot_data import load_transport_maps  # noqa: E402
from mnist_sae_models import DisplacementFieldSAE  # noqa: E402
from train_mnist_sae import sample_grid_from_data_mixture  # noqa: E402


def image_histogram_measure(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    h, w = image.shape
    rows, cols = np.nonzero(image > 0)
    if len(rows) == 0:
        rows, cols = np.nonzero(np.ones_like(image, dtype=bool))
        weights = np.ones(len(rows), dtype=np.float64)
    else:
        weights = image[rows, cols].astype(np.float64)

    denom_h = max(h - 1, 1)
    denom_w = max(w - 1, 1)
    points = np.stack([rows / denom_h, cols / denom_w], axis=1).astype(np.float64)
    masses = weights / weights.sum()
    return points, masses


def uniform_point_measure(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    masses = np.full(points.shape[0], 1.0 / points.shape[0], dtype=np.float64)
    return points, masses


def ot_reconstruction_error(
    target_points: np.ndarray,
    target_masses: np.ndarray,
    recon_points: np.ndarray,
    recon_masses: np.ndarray,
) -> dict[str, float]:
    cost = ot.dist(target_points, recon_points, metric="sqeuclidean")
    w2_squared = float(ot.emd2(target_masses, recon_masses, cost))
    return {
        "w2_squared": w2_squared,
        "w2": math.sqrt(max(w2_squared, 0.0)),
    }


def sorted_input_records(metadata_path: Path) -> list[dict]:
    metadata = json.loads(metadata_path.read_text())
    records = sorted(metadata["selected"], key=lambda item: item["filename"])
    image_dir = Path(metadata["image_dir"])
    for index, record in enumerate(records):
        record["shared_index"] = index
        record["image_path"] = str(image_dir / record["filename"])
    return records


def evaluate_heitz(run_dir: Path, records: list[dict], method_label: str = "heitz_wdl") -> tuple[list[dict], dict]:
    output_dir = run_dir / "outputs"
    rows: list[dict] = []
    for record in records:
        idx = record["shared_index"]
        fitting_path = output_dir / f"finalFitting_{idx:03d}.png"
        if not fitting_path.exists():
            raise FileNotFoundError(f"Missing Heitz fitting image: {fitting_path}")

        target_points, target_masses = image_histogram_measure(Path(record["image_path"]))
        recon_points, recon_masses = image_histogram_measure(fitting_path)
        metrics = ot_reconstruction_error(target_points, target_masses, recon_points, recon_masses)
        rows.append({
            "method": method_label,
            "shared_index": idx,
            "digit": record["digit"],
            "sample_index": record["sample_index"],
            "mnist_train_index": record["mnist_train_index"],
            "target_path": record["image_path"],
            "reconstruction_path": str(fitting_path),
            **metrics,
        })
    return rows, summarize(rows)


def build_mnist_sae_model(config: dict, checkpoint_path: Path, device: torch.device):
    X, maps = load_transport_maps(config["data_dir"], digits=config.get("digits"))
    X = X.float()
    maps = maps.float()

    grid_points = None
    if config.get("grid_mode", "uniform") == "data_mixture":
        grid_support_size = config.get("grid_support_size") or config["grid_side"] ** 2
        grid_points = sample_grid_from_data_mixture(
            n_images=config.get("grid_n_images", 500),
            support_size=grid_support_size,
            seed=config["seed"],
        )

    eps = float(config["epsilons"][0])
    model = DisplacementFieldSAE(
        X,
        m=int(config["m"]),
        eps=eps,
        grid_side=int(config["grid_side"]),
        lista_steps=int(config["lista_steps"]),
        grid_points=grid_points,
        normalize_atoms=True,
        per_atom_gain=True,
        lateral_init="damped_identity",
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, X, maps


def evaluate_ours(
    run_dir: Path,
    records: list[dict],
    device_str: str,
    method_label: str = "mnist_ot_sae",
) -> tuple[list[dict], dict]:
    output_dir = run_dir / "outputs"
    config = json.loads((output_dir / "config.json").read_text())
    eps = config["epsilons"][0]
    c = config["sparsity_coeffs"][0]
    checkpoint_path = output_dir / f"displacement_eps{eps}_c{c}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing SAE checkpoint: {checkpoint_path}")

    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable for shared evaluator; using CPU.", flush=True)
        device = torch.device("cpu")
    elif device_str == "mps" and not torch.backends.mps.is_available():
        print("MPS unavailable for shared evaluator; using CPU.", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    model, _, maps = build_mnist_sae_model(config, checkpoint_path, device)
    with torch.no_grad():
        t_hat, _ = model(maps.to(device).float())
    recon_maps = t_hat.cpu().numpy()

    if len(records) != recon_maps.shape[0]:
        raise ValueError(
            f"Record count ({len(records)}) does not match reconstructed maps "
            f"({recon_maps.shape[0]})."
        )

    rows: list[dict] = []
    for record, recon_points_raw in zip(records, recon_maps):
        target_points, target_masses = image_histogram_measure(Path(record["image_path"]))
        recon_points, recon_masses = uniform_point_measure(recon_points_raw)
        metrics = ot_reconstruction_error(target_points, target_masses, recon_points, recon_masses)
        rows.append({
            "method": method_label,
            "shared_index": record["shared_index"],
            "digit": record["digit"],
            "sample_index": record["sample_index"],
            "mnist_train_index": record["mnist_train_index"],
            "target_path": record["image_path"],
            "reconstruction": "sae_pushforward_uniform_base",
            **metrics,
        })
    return rows, summarize(rows)


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"num_samples": 0}
    w2_squared = np.array([row["w2_squared"] for row in rows], dtype=np.float64)
    w2 = np.array([row["w2"] for row in rows], dtype=np.float64)
    return {
        "num_samples": len(rows),
        "mean_w2_squared": float(w2_squared.mean()),
        "median_w2_squared": float(np.median(w2_squared)),
        "mean_w2": float(w2.mean()),
        "median_w2": float(np.median(w2)),
        "min_w2_squared": float(w2_squared.min()),
        "max_w2_squared": float(w2_squared.max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shared MNIST OT reconstruction evaluator.")
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--methods", nargs="+", choices=["heitz", "ours"], default=["heitz", "ours"])
    parser.add_argument("--metadata-path", type=Path, default=None,
                        help="Input PNG metadata. Defaults to <comparison-dir>/heitz_wdl/data/mnist_png/metadata.json.")
    parser.add_argument("--heitz-run-dir", type=Path, default=None,
                        help="Heitz run directory. Defaults to <comparison-dir>/heitz_wdl.")
    parser.add_argument("--ours-run-dir", type=Path, default=None,
                        help="OT-SAE run directory. Defaults to <comparison-dir>/mnist_ot_sae.")
    parser.add_argument("--heitz-label", default="heitz_wdl")
    parser.add_argument("--ours-label", default="mnist_ot_sae")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison_dir = args.comparison_dir.resolve()
    output_path = args.output or (comparison_dir / "shared_ot_reconstruction.json")
    metadata_path = args.metadata_path or (
        comparison_dir / "heitz_wdl" / "data" / "mnist_png" / "metadata.json"
    )
    records = sorted_input_records(metadata_path.resolve())
    heitz_run_dir = (args.heitz_run_dir or (comparison_dir / "heitz_wdl")).resolve()
    ours_run_dir = (args.ours_run_dir or (comparison_dir / "mnist_ot_sae")).resolve()

    per_method: dict[str, dict] = {}
    per_sample: list[dict] = []

    if "heitz" in args.methods and (heitz_run_dir / "summary.json").exists():
        rows, summary = evaluate_heitz(heitz_run_dir, records, args.heitz_label)
        per_sample.extend(rows)
        per_method[args.heitz_label] = summary

    if "ours" in args.methods and (ours_run_dir / "summary.json").exists():
        rows, summary = evaluate_ours(ours_run_dir, records, args.device, args.ours_label)
        per_sample.extend(rows)
        per_method[args.ours_label] = summary

    payload = {
        "metric": {
            "name": "mnist_ot_reconstruction_error",
            "definition": "POT emd2 between original MNIST image measure and reconstructed measure",
            "cost": "squared Euclidean distance on [0,1]^2 pixel/support coordinates",
            "w2_squared": "reported emd2 value",
            "w2": "sqrt(w2_squared)",
            "notes": [
                "Heitz reconstructions are read from finalFitting PNG files, so they include PNG quantization.",
                "MNIST OT-SAE reconstructions are uniform pushforwards of the learned reconstructed transport maps.",
            ],
        },
        "comparison_dir": str(comparison_dir),
        "metadata_path": str(metadata_path.resolve()),
        "per_method": per_method,
        "per_sample": per_sample,
    }
    write_json(output_path, payload)
    print(f"Shared OT reconstruction metrics written to {output_path}")


if __name__ == "__main__":
    main()
