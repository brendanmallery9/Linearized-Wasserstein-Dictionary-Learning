from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import ot
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets

from common import REPO_ROOT, stream_command, timestamp, write_json

MNIST_PIPELINE = REPO_ROOT / "mnist" / "pipeline"
HSI_PIPELINE = REPO_ROOT / "hsi" / "pipeline"
for path in (REPO_ROOT, MNIST_PIPELINE, HSI_PIPELINE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import SAE  # noqa: E402
from OT_utils import image_to_empirical  # noqa: E402
from brenier_embedding_functions import brenier_potential, wass_map_1D  # noqa: E402
from mnist_sae_models import DisplacementFieldSAE  # noqa: E402
from train_mnist_sae import train_one_model  # noqa: E402


PRESETS = {
    "smoke": {
        "atoms": 3,
        "top_k": 2,
        "lista_steps": 2,
        "epochs": 1,
        "batch_size": 8,
        "base_supp_size": 32,
        "pavia_samples": 8,
        "mnist_max_per_digit": 1,
        "gaussian_measures": 6,
        "gaussian_support_size": 32,
        "heitz_max_optim_iter": 1,
        "heitz_sinkhorn_iters": 1,
    },
    "local": {
        "atoms": 30,
        "top_k": 3,
        "lista_steps": 20,
        "epochs": 100,
        "batch_size": 128,
        "base_supp_size": 400,
        "pavia_samples": 100,
        "mnist_max_per_digit": 10,
        "gaussian_measures": 100,
        "gaussian_support_size": 400,
        "heitz_max_optim_iter": 500,
        "heitz_sinkhorn_iters": 5,
    },
}

PAVIA_EPSILONS = [0.25, 0.025, 0.005]
MNIST_EPSILONS = [0.25, 0.125, 0.025]
GAUSSIAN_DIMS = [1, 5, 10, 20, 40]
HEITZ_GAMMAS = [0.5, 2.0]


class TensorOnlyDataset(Dataset):
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __len__(self) -> int:
        return int(self.tensor.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.tensor[index]


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def label_float(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU", flush=True)
        return torch.device("cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU", flush=True)
        return torch.device("cpu")
    return torch.device(device)


def choose(args: argparse.Namespace, name: str):
    value = getattr(args, name)
    return PRESETS[args.preset][name] if value is None else value


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


def normalize_masses(values: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    masses = np.asarray(values, dtype=np.float64).reshape(-1)
    masses = np.nan_to_num(masses, nan=0.0, posinf=0.0, neginf=0.0)
    masses = np.maximum(masses, 0.0)
    total = float(masses.sum())
    if total <= eps:
        masses = np.ones_like(masses, dtype=np.float64)
        total = float(masses.sum())
    return masses / total


def as_points(array: np.ndarray | torch.Tensor) -> np.ndarray:
    if torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    points = np.asarray(array, dtype=np.float64)
    if points.ndim == 1:
        points = points[:, None]
    return points


def uniform_masses(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n, dtype=np.float64)


def barycentric_ot_map(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    source_masses: np.ndarray | None = None,
    target_masses: np.ndarray | None = None,
    method: str = "entropic",
    eps: float = 0.025,
    num_iter: int = 10_000,
) -> np.ndarray:
    source_points = as_points(source_points)
    target_points = as_points(target_points)
    a = uniform_masses(source_points.shape[0]) if source_masses is None else normalize_masses(source_masses)
    b = uniform_masses(target_points.shape[0]) if target_masses is None else normalize_masses(target_masses)

    cost = ot.dist(source_points, target_points, metric="sqeuclidean").astype(np.float64)
    scale = float(cost.max())
    if scale <= 0 or not np.isfinite(scale):
        scale = 1.0
    scaled_cost = cost / scale

    if method == "emd":
        plan = ot.emd(a, b, scaled_cost, numItermax=10_000_000)
    elif method == "entropic":
        plan = ot.bregman.sinkhorn_stabilized(
            a,
            b,
            scaled_cost,
            reg=float(eps),
            numItermax=num_iter,
            stopThr=1e-9,
            warn=False,
        )
    else:
        raise ValueError(f"Unknown OT method: {method}")

    row_sums = plan.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-300)
    mapped = (plan / row_sums) @ target_points
    return mapped.astype(np.float32)


def compute_potential(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    source_masses: np.ndarray | None = None,
    target_masses: np.ndarray | None = None,
    method: str = "emd",
    eps_reg: float | None = None,
) -> torch.Tensor:
    potential = brenier_potential(
        as_points(source_points),
        source_masses,
        as_points(target_points),
        target_masses,
        method=method,
        eps_reg=eps_reg,
    ).float()
    return potential - potential.mean()


def w2_squared(
    target_points: np.ndarray,
    target_masses: np.ndarray,
    recon_points: np.ndarray,
    recon_masses: np.ndarray | None = None,
) -> float:
    target_points = as_points(target_points)
    recon_points = as_points(recon_points)
    target_masses = normalize_masses(target_masses)
    if recon_masses is None:
        recon_masses = uniform_masses(recon_points.shape[0])
    else:
        recon_masses = normalize_masses(recon_masses)
    cost = ot.dist(target_points, recon_points, metric="sqeuclidean")
    return float(ot.emd2(target_masses, recon_masses, cost))


def make_loaders(
    tensor: torch.Tensor,
    *,
    batch_size: int,
    test_fraction: float,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset = TensorOnlyDataset(tensor)
    n = len(dataset)
    if n <= 1:
        n_test = 0
    else:
        n_test = max(1, int(round(n * test_fraction)))
        n_test = min(n_test, n - 1)
    n_train = n - n_test
    generator = torch.Generator().manual_seed(seed)
    if n_test:
        train_set, test_set = random_split(dataset, [n_train, n_test], generator=generator)
    else:
        train_set = dataset
        test_set = dataset
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, all_loader


def reconstruct_tensor(model: torch.nn.Module, data: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    model.eval()
    outs = []
    loader = DataLoader(TensorOnlyDataset(data), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            xhat, _ = model(batch.to(device).float())
            outs.append(xhat.detach().cpu())
    model.train()
    return torch.cat(outs, dim=0)


def build_generic_sae(
    architecture: str,
    *,
    input_dim: int,
    hidden_dim: int,
    top_k: int,
    lista_steps: int,
) -> torch.nn.Module:
    if architecture == "TopKAE":
        return SAE.TopKAE(input_dim, hidden_dim, top_k=top_k)
    if architecture == "JumpReLU":
        return SAE.JumpReLUAE(input_dim, hidden_dim)
    if architecture == "ReLUAE":
        return SAE.ReLUAE(input_dim, hidden_dim)
    if architecture == "LISTAAE":
        return SAE.LISTAAE(
            input_dim,
            hidden_dim,
            lista_steps=lista_steps,
            activation_type="relu",
            per_atom_gain=True,
            lateral_init="damped_identity",
        )
    if architecture == "JumpReLU_monotone":
        return SAE.JumpReLUAE_monotone(input_dim, hidden_dim)
    if architecture == "TopKAE_monotone":
        return SAE.TopKAE_monotone(input_dim, hidden_dim, top_k=top_k)
    raise ValueError(f"Unsupported architecture: {architecture}")


def evaluate_generic_recon(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    sparsity_coeff: float,
) -> dict[str, float]:
    total_recon = 0.0
    total_loss = 0.0
    total_l1 = 0.0
    total_active = 0.0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device).float()
            xhat, z = model(batch)
            recon_per_sample = ((xhat - batch) ** 2).mean(dim=1)
            l1_per_sample = z.abs().sum(dim=1)
            active_per_sample = (z.abs() > 0).float().sum(dim=1)
            total_recon += float(recon_per_sample.sum().item())
            total_l1 += float(l1_per_sample.sum().item())
            total_active += float(active_per_sample.sum().item())
            total_loss += float((recon_per_sample + sparsity_coeff * l1_per_sample).sum().item())
            total_samples += int(batch.shape[0])
    model.train()
    total_samples = max(total_samples, 1)
    return {
        "recon_mse": total_recon / total_samples,
        "total_loss": total_loss / total_samples,
        "mean_l1": total_l1 / total_samples,
        "mean_active": total_active / total_samples,
    }


def train_generic_sae(
    data: torch.Tensor,
    *,
    architecture: str,
    hidden_dim: int,
    top_k: int,
    lista_steps: int,
    batch_size: int,
    epochs: int,
    lr: float,
    sparsity_coeff: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    history_path: Path | None,
    run_name: str,
    history_time_offset: float = 0.0,
    max_elapsed_seconds: float | None = None,
    plateau_window: int = 0,
    plateau_min_delta: float = 0.0,
    test_fraction: float = 0.1,
) -> tuple[torch.nn.Module, dict[str, Any], torch.Tensor]:
    train_loader, test_loader, all_loader = make_loaders(
        data,
        batch_size=batch_size,
        test_fraction=test_fraction,
        seed=seed,
    )
    model = build_generic_sae(
        architecture,
        input_dim=int(data.shape[1]),
        hidden_dim=hidden_dim,
        top_k=top_k,
        lista_steps=lista_steps,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if history_path is not None:
        history_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    epoch_losses: list[float] = []
    best_plateau_average = None
    termination_reason = "max_epochs"
    epochs_completed = 0
    termination_elapsed = history_time_offset

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        steps = 0
        for batch in train_loader:
            batch = batch.to(device).float()
            xhat, z = model(batch)
            recon = F.mse_loss(xhat, batch, reduction="mean")
            sparsity = z.abs().sum(dim=1).mean()
            loss = recon + sparsity_coeff * sparsity
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            steps += 1

        avg_loss = epoch_loss / max(steps, 1)
        epoch_losses.append(avg_loss)
        epochs_completed = epoch + 1
        elapsed = time.monotonic() - start
        termination_elapsed = history_time_offset + elapsed
        if history_path is not None:
            with history_path.open("a") as f:
                f.write(json.dumps({
                    "event": "epoch",
                    "run_name": run_name,
                    "epoch": epochs_completed,
                    "epochs": epochs,
                    "elapsed_seconds": termination_elapsed,
                    "train_loss": avg_loss,
                }) + "\n")
        if epoch == 0 or epochs_completed % 10 == 0 or epochs_completed == epochs:
            print(f"[{run_name}] Epoch {epochs_completed:4d}/{epochs} loss={avg_loss:.6f}", flush=True)

        stop_reason = None
        if max_elapsed_seconds is not None and termination_elapsed >= max_elapsed_seconds:
            stop_reason = "max_elapsed_seconds"
        elif plateau_window > 0 and len(epoch_losses) >= plateau_window:
            window_average = sum(epoch_losses[-plateau_window:]) / plateau_window
            if best_plateau_average is None:
                best_plateau_average = window_average
            elif best_plateau_average - window_average > plateau_min_delta:
                best_plateau_average = window_average
            else:
                stop_reason = "loss_plateau"

        if stop_reason is not None:
            termination_reason = stop_reason
            break

    if history_path is not None:
        with history_path.open("a") as f:
            f.write(json.dumps({
                "event": "termination",
                "run_name": run_name,
                "reason": termination_reason,
                "epoch": epochs_completed,
                "epochs": epochs,
                "elapsed_seconds": termination_elapsed,
                "train_loss": epoch_losses[-1] if epoch_losses else None,
            }) + "\n")

    train_metrics = evaluate_generic_recon(
        model,
        train_loader,
        device=device,
        sparsity_coeff=sparsity_coeff,
    )
    test_metrics = evaluate_generic_recon(
        model,
        test_loader,
        device=device,
        sparsity_coeff=sparsity_coeff,
    )
    all_metrics = evaluate_generic_recon(
        model,
        all_loader,
        device=device,
        sparsity_coeff=sparsity_coeff,
    )
    recon = reconstruct_tensor(model, data, device, batch_size)
    metrics = {
        "epochs_completed": epochs_completed,
        "requested_epochs": epochs,
        "termination_reason": termination_reason,
        "termination_elapsed_seconds": termination_elapsed,
        "final_train_loss": epoch_losses[-1] if epoch_losses else None,
        "train_recon_mse": train_metrics["recon_mse"],
        "test_recon_mse": test_metrics["recon_mse"],
        "all_recon_mse": all_metrics["recon_mse"],
        "train_total_loss": train_metrics["total_loss"],
        "test_total_loss": test_metrics["total_loss"],
        "all_total_loss": all_metrics["total_loss"],
        "mean_active": all_metrics["mean_active"],
    }
    return model, metrics, recon


def train_ebcm(
    source_points: np.ndarray,
    maps: torch.Tensor,
    *,
    eps: float,
    grid_points: np.ndarray | torch.Tensor | None,
    args: argparse.Namespace,
    device: torch.device,
    history_path: Path | None,
    run_name: str,
    history_time_offset: float,
) -> tuple[DisplacementFieldSAE, dict[str, Any], torch.Tensor]:
    source = torch.as_tensor(source_points, dtype=torch.float32)
    maps = maps.float()
    train_loader, test_loader, _ = make_loaders(
        maps,
        batch_size=args.batch_size,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    grid_tensor = None
    if grid_points is not None:
        grid_tensor = torch.as_tensor(grid_points, dtype=torch.float32)

    model = DisplacementFieldSAE(
        source,
        m=args.atoms,
        eps=eps,
        grid_side=args.grid_side,
        lista_steps=args.lista_steps,
        grid_points=grid_tensor,
        normalize_atoms=True,
        per_atom_gain=True,
        lateral_init="damped_identity",
    ).to(device)

    config = {
        "name": run_name,
        "sparsity_coeff": args.sparsity_coeff,
        "optimizer": "adamw",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": "none",
        "epochs": args.epochs,
        "history_path": str(history_path) if history_path else None,
        "history_every": args.history_every,
        "history_time_offset": history_time_offset,
        "max_elapsed_seconds": args.max_elapsed_seconds,
        "plateau_window": args.plateau_window,
        "plateau_min_delta": args.plateau_min_delta,
    }
    metrics = train_one_model(model, train_loader, test_loader, config, device)
    recon = reconstruct_tensor(model, maps, device, args.batch_size)
    return model, metrics, recon


def save_table(rows: list[dict[str, Any]], out_dir: Path, name: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    json_path = out_dir / f"{name}.json"
    md_path = out_dir / f"{name}.md"
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    write_json(json_path, rows)
    md_path.write_text(df.to_string(index=False) + "\n" if rows else "_No rows_\n")
    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "markdown": str(md_path),
    }


def load_pavia_cube(cube_path: Path) -> torch.Tensor:
    if not cube_path.exists():
        try:
            from download_hyperspec_data import restore_pavia_cube_from_chunks
        except Exception:
            restore_pavia_cube_from_chunks = None
        if restore_pavia_cube_from_chunks is not None:
            restore_pavia_cube_from_chunks(cube_path.parents[1])
    if not cube_path.exists():
        raise FileNotFoundError(
            f"Missing Pavia cube at {cube_path}. Run hsi/pipeline/download_hyperspec_data.py for pavia first."
        )
    cube = torch_load(cube_path)
    if isinstance(cube, dict):
        cube = cube.get("data", cube.get("cube"))
    if not torch.is_tensor(cube):
        raise TypeError(f"Expected tensor-like Pavia cube at {cube_path}, got {type(cube)}")
    return cube.float()


def prepare_pavia_subset(args: argparse.Namespace, cache_dir: Path) -> dict[str, Any]:
    subset_dir = cache_dir / "pavia1d" / (
        f"samples{args.pavia_samples}_support{args.pavia_support_size}_seed{args.seed}"
    )
    payload_path = subset_dir / "subset.pt"
    if payload_path.exists() and not args.force_cache:
        return torch_load(payload_path)

    subset_dir.mkdir(parents=True, exist_ok=True)
    cube = load_pavia_cube(args.pavia_cube_path)
    flat = cube.reshape(-1, cube.shape[-1])
    spectra = flat.clamp_min(0)
    valid = torch.isfinite(spectra).all(dim=1) & (spectra.sum(dim=1) > 0)
    valid_indices = valid.nonzero(as_tuple=False).reshape(-1)
    if len(valid_indices) < args.pavia_samples:
        raise RuntimeError(f"Only {len(valid_indices)} valid Pavia spectra available")

    generator = torch.Generator().manual_seed(args.seed)
    chosen_offsets = torch.randperm(len(valid_indices), generator=generator)[:args.pavia_samples]
    chosen_indices = valid_indices[chosen_offsets]
    chosen = spectra[chosen_indices]
    masses = chosen / chosen.sum(dim=1, keepdim=True).clamp_min(1e-12)

    target_grid = torch.linspace(0.0, 1.0, cube.shape[-1]).reshape(-1, 1)
    source_grid = torch.linspace(0.0, 1.0, args.pavia_support_size).reshape(-1, 1)
    records = [
        {
            "sample_id": f"pavia_{i:06d}",
            "flat_index": int(chosen_indices[i]),
        }
        for i in range(args.pavia_samples)
    ]
    payload = {
        "source_points": source_grid,
        "target_points": target_grid,
        "target_masses": masses.float(),
        "records": records,
        "parameters": {
            "cube_path": str(args.pavia_cube_path),
            "pavia_samples": args.pavia_samples,
            "pavia_support_size": args.pavia_support_size,
            "seed": args.seed,
        },
    }
    torch.save(payload, payload_path)
    write_json(subset_dir / "metadata.json", payload["parameters"])
    return payload


def pavia_target_arrays(data: dict[str, Any], index: int) -> tuple[np.ndarray, np.ndarray]:
    target_points = as_points(data["target_points"])
    masses = np.asarray(data["target_masses"][index], dtype=np.float64)
    return target_points, normalize_masses(masses)


def prepare_pavia_maps(
    data: dict[str, Any],
    cache_dir: Path,
    *,
    method: str,
    eps: float | None,
    force: bool,
    progress_every: int,
) -> tuple[torch.Tensor, float, Path]:
    eps_label = "exact" if eps is None else label_float(eps)
    out_dir = cache_dir / "pavia1d" / "embeddings" / f"{method}_eps{eps_label}"
    maps_path = out_dir / "maps.pt"
    meta_path = out_dir / "metadata.json"
    params = {
        "method": method,
        "eps": eps,
        "num_samples": len(data["records"]),
        "source_shape": tuple(data["source_points"].shape),
    }
    if maps_path.exists() and meta_path.exists() and not force:
        metadata = json.loads(meta_path.read_text())
        return torch_load(maps_path).float(), float(metadata.get("elapsed_seconds", 0.0)), maps_path

    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    source = as_points(data["source_points"])
    maps = []
    for index in range(len(data["records"])):
        if progress_every and (index == 0 or index % progress_every == 0):
            print(f"[pavia:{method}] {index}/{len(data['records'])}", flush=True)
        target_points, target_masses = pavia_target_arrays(data, index)
        if method == "hsi_1d":
            mapping = wass_map_1D(
                source,
                None,
                target_points,
                target_masses,
                dtype=torch.float32,
            ).cpu().numpy()
        elif method == "ebcm":
            mapping = barycentric_ot_map(
                source,
                target_points,
                target_masses=target_masses,
                method="entropic",
                eps=float(eps),
            )
        else:
            raise ValueError(method)
        maps.append(torch.as_tensor(mapping, dtype=torch.float32))
    tensor = torch.stack(maps, dim=0)
    elapsed = time.monotonic() - start
    torch.save(tensor, maps_path)
    write_json(meta_path, {**params, "elapsed_seconds": elapsed})
    return tensor, elapsed, maps_path


def prepare_potentials(
    source_points: np.ndarray | torch.Tensor,
    targets: list[tuple[np.ndarray, np.ndarray | None]],
    out_dir: Path,
    *,
    force: bool,
    progress_label: str,
    progress_every: int,
    method: str = "emd",
    eps_reg: float | None = None,
) -> tuple[torch.Tensor, float, Path]:
    potentials_path = out_dir / "potentials.pt"
    meta_path = out_dir / "metadata.json"
    if potentials_path.exists() and meta_path.exists() and not force:
        metadata = json.loads(meta_path.read_text())
        return torch_load(potentials_path).float(), float(metadata.get("elapsed_seconds", 0.0)), potentials_path
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    source = as_points(source_points)
    potentials = []
    for index, (target_points, target_masses) in enumerate(targets):
        if progress_every and (index == 0 or index % progress_every == 0):
            print(f"[{progress_label}:potential] {index}/{len(targets)}", flush=True)
        potentials.append(
            compute_potential(
                source,
                target_points,
                target_masses=target_masses,
                method=method,
                eps_reg=eps_reg,
            )
        )
    tensor = torch.stack(potentials, dim=0).float()
    elapsed = time.monotonic() - start
    torch.save(tensor, potentials_path)
    write_json(meta_path, {
        "method": method,
        "eps_reg": eps_reg,
        "num_samples": len(targets),
        "source_shape": tuple(source.shape),
        "elapsed_seconds": elapsed,
    })
    return tensor, elapsed, potentials_path


def write_pavia_pngs(data: dict[str, Any], out_dir: Path, *, force: bool) -> tuple[Path, list[Path]]:
    image_dir = out_dir / "pavia_png" / "all"
    if image_dir.exists() and force:
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, record in enumerate(data["records"]):
        path = image_dir / f"{record['sample_id']}.png"
        if not path.exists() or force:
            masses = np.asarray(data["target_masses"][index], dtype=np.float64)
            masses = normalize_masses(masses)
            scaled = masses / max(float(masses.max()), 1e-12)
            pixels = np.round(255.0 * scaled).clip(0, 255).astype(np.uint8)
            Image.fromarray(np.tile(pixels.reshape(1, -1), (2, 1)), mode="L").save(path)
        paths.append(path)
    write_json(image_dir.parent / "metadata.json", {
        "format": "pavia_1d_png_for_heitz",
        "image_dir": str(image_dir),
        "num_images": len(paths),
        "support_size": int(data["target_points"].shape[0]),
    })
    return image_dir, paths


def image_measure_1d(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"Expected grayscale image at {path}")
    weights = image.sum(axis=0)
    points = np.linspace(0.0, 1.0, image.shape[1]).reshape(-1, 1)
    return points, normalize_masses(weights)


def image_measure_2d(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    rows, cols = np.nonzero(image > 0)
    if len(rows) == 0:
        rows, cols = np.nonzero(np.ones_like(image, dtype=bool))
        weights = np.ones(len(rows), dtype=np.float64)
    else:
        weights = image[rows, cols].astype(np.float64)
    denom_h = max(image.shape[0] - 1, 1)
    denom_w = max(image.shape[1] - 1, 1)
    points = np.stack([rows / denom_h, cols / denom_w], axis=1)
    return points.astype(np.float64), normalize_masses(weights)


def run_heitz_trial(
    *,
    input_dir: Path,
    run_dir: Path,
    args: argparse.Namespace,
    gamma: float,
    sinkhorn_iters: int,
    source_dir: Path,
    build_dir: Path,
    log_dir: Path,
) -> tuple[dict[str, Any], float]:
    cmd = [
        sys.executable,
        REPO_ROOT / "experiments" / "timed_comparison" / "run_heitz_wdl.py",
        "--run-dir", run_dir,
        "--source-dir", source_dir,
        "--build-dir", build_dir,
        "--input-dir", input_dir,
        "--k", args.atoms,
        "--loss-type", args.heitz_loss_type,
        "--sinkhorn-iters", sinkhorn_iters,
        "--max-optim-iter", args.heitz_max_optim_iter,
        "--gamma", gamma,
        "--scale-dict-factor", args.heitz_scale_dict_factor,
        "--avx", args.heitz_avx,
        "--max-elapsed-seconds", args.max_elapsed_seconds if args.max_elapsed_seconds is not None else 0,
        "--plateau-window", args.plateau_window,
        "--plateau-min-delta", args.plateau_min_delta,
        "--deterministic",
    ]
    if args.max_elapsed_seconds is None:
        idx = cmd.index("--max-elapsed-seconds")
        del cmd[idx:idx + 2]
    if args.heitz_with_openmp:
        cmd.append("--with-openmp")
    returncode, elapsed = stream_command(
        cmd,
        cwd=REPO_ROOT,
        log_path=log_dir / f"{run_dir.name}.log",
    )
    if returncode != 0:
        raise RuntimeError(f"Heitz trial failed: {run_dir}")
    summary = json.loads((run_dir / "summary.json").read_text())
    return summary, elapsed


def evaluate_heitz_outputs(
    *,
    run_dir: Path,
    target_measures: list[tuple[np.ndarray, np.ndarray]],
    kind: str,
) -> float:
    rows = []
    for index, (target_points, target_masses) in enumerate(target_measures):
        fitting_path = run_dir / "outputs" / f"finalFitting_{index:03d}.png"
        if not fitting_path.exists():
            raise FileNotFoundError(f"Missing Heitz output: {fitting_path}")
        if kind == "1d":
            recon_points, recon_masses = image_measure_1d(fitting_path)
        elif kind == "2d":
            recon_points, recon_masses = image_measure_2d(fitting_path)
        else:
            raise ValueError(kind)
        rows.append(w2_squared(target_points, target_masses, recon_points, recon_masses))
    return float(np.mean(rows)) if rows else float("nan")


def prepare_mnist_subset(args: argparse.Namespace, cache_dir: Path) -> dict[str, Any]:
    subset_dir = cache_dir / "mnist" / (
        f"digits{'-'.join(map(str, args.mnist_digits))}_max{args.mnist_max_per_digit}"
        f"_support{args.base_supp_size}_seed{args.seed}"
    )
    payload_path = subset_dir / "subset.pt"
    if payload_path.exists() and not args.force_cache:
        return torch_load(payload_path)

    subset_dir.mkdir(parents=True, exist_ok=True)
    image_dir = subset_dir / "png" / "all"
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    mnist = datasets.MNIST(root=str(REPO_ROOT / "mnist_raw"), train=True, download=True)
    rng = np.random.RandomState(args.seed)
    source_points = rng.rand(args.base_supp_size, 2).astype(np.float32)

    counts = {digit: 0 for digit in args.mnist_digits}
    records = []
    target_points: list[torch.Tensor] = []
    target_masses: list[torch.Tensor] = []
    for dataset_index, (image, label) in enumerate(mnist):
        label = int(label)
        if label not in counts:
            continue
        if counts[label] >= args.mnist_max_per_digit:
            continue
        sample_index = counts[label]
        filename = f"digit{label}_sample{sample_index:05d}_mnist{dataset_index:05d}.png"
        image.save(image_dir / filename)
        measure = image_to_empirical(np.asarray(image))
        target_points.append(torch.as_tensor(measure.points, dtype=torch.float32))
        target_masses.append(torch.as_tensor(measure.masses, dtype=torch.float64))
        records.append({
            "digit": label,
            "sample_index": sample_index,
            "mnist_train_index": int(dataset_index),
            "filename": filename,
            "image_path": str(image_dir / filename),
        })
        counts[label] += 1
        if all(counts[d] >= args.mnist_max_per_digit for d in args.mnist_digits):
            break

    missing = {digit: args.mnist_max_per_digit - count for digit, count in counts.items() if count < args.mnist_max_per_digit}
    if missing:
        raise RuntimeError(f"Could not collect requested MNIST digits: {missing}")

    payload = {
        "source_points": torch.as_tensor(source_points),
        "target_points": target_points,
        "target_masses": target_masses,
        "records": records,
        "image_dir": str(image_dir),
        "parameters": {
            "digits": args.mnist_digits,
            "max_per_digit": args.mnist_max_per_digit,
            "base_supp_size": args.base_supp_size,
            "seed": args.seed,
        },
    }
    torch.save(payload, payload_path)
    write_json(subset_dir / "metadata.json", {
        **payload["parameters"],
        "image_dir": str(image_dir),
        "num_images": len(records),
        "selected": records,
    })
    return payload


def mnist_targets(data: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (as_points(points), normalize_masses(np.asarray(masses)))
        for points, masses in zip(data["target_points"], data["target_masses"])
    ]


def prepare_mnist_maps(
    data: dict[str, Any],
    cache_dir: Path,
    *,
    eps: float,
    force: bool,
    progress_every: int,
) -> tuple[torch.Tensor, float, Path]:
    out_dir = cache_dir / "mnist" / "embeddings" / (
        f"digits{'-'.join(map(str, sorted({r['digit'] for r in data['records']})))}"
        f"_n{len(data['records'])}_support{data['source_points'].shape[0]}_eps{label_float(eps)}"
    )
    maps_path = out_dir / "maps.pt"
    meta_path = out_dir / "metadata.json"
    if maps_path.exists() and meta_path.exists() and not force:
        metadata = json.loads(meta_path.read_text())
        return torch_load(maps_path).float(), float(metadata.get("elapsed_seconds", 0.0)), maps_path

    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    source = as_points(data["source_points"])
    maps = []
    for index, (target_points, target_masses) in enumerate(mnist_targets(data)):
        if progress_every and (index == 0 or index % progress_every == 0):
            print(f"[mnist:ebcm eps={eps}] {index}/{len(data['records'])}", flush=True)
        maps.append(torch.as_tensor(
            barycentric_ot_map(
                source,
                target_points,
                target_masses=target_masses,
                method="entropic",
                eps=eps,
            ),
            dtype=torch.float32,
        ))
    tensor = torch.stack(maps, dim=0)
    elapsed = time.monotonic() - start
    torch.save(tensor, maps_path)
    write_json(meta_path, {
        "method": "entropic_barycentric_maps",
        "eps": eps,
        "num_samples": len(data["records"]),
        "source_shape": tuple(data["source_points"].shape),
        "elapsed_seconds": elapsed,
    })
    return tensor, elapsed, maps_path


def generate_gaussian_covariance(rng: np.random.Generator, dim: int) -> np.ndarray:
    if dim == 1:
        variance = float(np.exp(rng.normal(loc=0.0, scale=0.45)))
        return np.array([[variance]], dtype=np.float64)
    matrix = rng.normal(size=(dim, dim))
    cov = matrix @ matrix.T / dim
    cov += 0.15 * np.eye(dim)
    cov *= dim / np.trace(cov)
    return cov.astype(np.float64)


def prepare_gaussian_dataset(args: argparse.Namespace, cache_dir: Path, dim: int) -> dict[str, Any]:
    data_dir = cache_dir / "gaussian" / (
        f"dim{dim}_measures{args.gaussian_measures}_support{args.gaussian_support_size}_seed{args.seed}"
    )
    payload_path = data_dir / "dataset.pt"
    if payload_path.exists() and not args.force_cache:
        return torch_load(payload_path)

    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed + 10_000 * dim)
    source = rng.normal(size=(args.gaussian_support_size, dim)).astype(np.float32)
    targets = []
    covariances = []
    for _ in range(args.gaussian_measures):
        cov = generate_gaussian_covariance(rng, dim)
        target = rng.multivariate_normal(
            mean=np.zeros(dim, dtype=np.float64),
            cov=cov,
            size=args.gaussian_support_size,
        ).astype(np.float32)
        targets.append(torch.as_tensor(target))
        covariances.append(torch.as_tensor(cov, dtype=torch.float32))
    target_tensor = torch.stack(targets, dim=0)

    flat_targets = target_tensor.reshape(-1, dim)
    generator = torch.Generator().manual_seed(args.seed + dim)
    grid_count = min(args.gaussian_support_size, flat_targets.shape[0])
    grid_idx = torch.randperm(flat_targets.shape[0], generator=generator)[:grid_count]
    grid_points = flat_targets[grid_idx].float()

    payload = {
        "source_points": torch.as_tensor(source),
        "target_points": target_tensor.float(),
        "grid_points": grid_points,
        "covariances": torch.stack(covariances, dim=0),
        "records": [{"sample_id": f"gaussian_d{dim}_{i:05d}", "dimension": dim} for i in range(args.gaussian_measures)],
        "parameters": {
            "dim": dim,
            "num_measures": args.gaussian_measures,
            "support_size": args.gaussian_support_size,
            "seed": args.seed,
            "centered": True,
            "covariance_trace": dim,
        },
    }
    torch.save(payload, payload_path)
    write_json(data_dir / "metadata.json", payload["parameters"])
    return payload


def prepare_gaussian_maps(
    data: dict[str, Any],
    cache_dir: Path,
    *,
    dim: int,
    eps: float,
    force: bool,
    progress_every: int,
) -> tuple[torch.Tensor, float, Path]:
    out_dir = cache_dir / "gaussian" / "embeddings" / (
        f"dim{dim}_n{len(data['records'])}_support{data['source_points'].shape[0]}_eps{label_float(eps)}"
    )
    maps_path = out_dir / "maps.pt"
    meta_path = out_dir / "metadata.json"
    if maps_path.exists() and meta_path.exists() and not force:
        metadata = json.loads(meta_path.read_text())
        return torch_load(maps_path).float(), float(metadata.get("elapsed_seconds", 0.0)), maps_path

    out_dir.mkdir(parents=True, exist_ok=True)
    source = as_points(data["source_points"])
    start = time.monotonic()
    maps = []
    for index, target in enumerate(data["target_points"]):
        if progress_every and (index == 0 or index % progress_every == 0):
            print(f"[gaussian d={dim}:ebcm eps={eps}] {index}/{len(data['records'])}", flush=True)
        maps.append(torch.as_tensor(
            barycentric_ot_map(source, as_points(target), method="entropic", eps=eps),
            dtype=torch.float32,
        ))
    tensor = torch.stack(maps, dim=0)
    elapsed = time.monotonic() - start
    torch.save(tensor, maps_path)
    write_json(meta_path, {
        "method": "entropic_barycentric_maps",
        "dim": dim,
        "eps": eps,
        "elapsed_seconds": elapsed,
        "num_samples": len(data["records"]),
    })
    return tensor, elapsed, maps_path


def pavia_w2_errors(data: dict[str, Any], recon_maps: torch.Tensor) -> float:
    errors = []
    for index, recon in enumerate(recon_maps):
        target_points, target_masses = pavia_target_arrays(data, index)
        errors.append(w2_squared(target_points, target_masses, as_points(recon)))
    return float(np.mean(errors)) if errors else float("nan")


def mnist_w2_errors(data: dict[str, Any], recon_maps: torch.Tensor) -> float:
    errors = []
    for (target_points, target_masses), recon in zip(mnist_targets(data), recon_maps):
        errors.append(w2_squared(target_points, target_masses, as_points(recon)))
    return float(np.mean(errors)) if errors else float("nan")


def run_pavia1d(args: argparse.Namespace, run_dir: Path, cache_dir: Path, device: torch.device) -> list[dict[str, Any]]:
    print("\n=== Experiment 1: Pavia 1D ===", flush=True)
    exp_dir = run_dir / "pavia1d"
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_dir = exp_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    data = prepare_pavia_subset(args, cache_dir)
    rows: list[dict[str, Any]] = []

    hsi_maps, hsi_embed_seconds, hsi_maps_path = prepare_pavia_maps(
        data,
        cache_dir,
        method="hsi_1d",
        eps=None,
        force=args.force_cache,
        progress_every=args.progress_every,
    )
    hsi_history = exp_dir / "hsi_1d_history.jsonl"
    t0 = time.monotonic()
    _, hsi_metrics, hsi_recon = train_generic_sae(
        hsi_maps.reshape(hsi_maps.shape[0], hsi_maps.shape[1]),
        architecture=args.hsi_architecture,
        hidden_dim=args.atoms,
        top_k=args.top_k,
        lista_steps=args.lista_steps,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        sparsity_coeff=args.sparsity_coeff,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
        history_path=hsi_history,
        run_name="pavia_hsi_1d",
        history_time_offset=hsi_embed_seconds,
        max_elapsed_seconds=args.max_elapsed_seconds,
        plateau_window=args.plateau_window,
        plateau_min_delta=args.plateau_min_delta,
        test_fraction=args.test_fraction,
    )
    hsi_train_seconds = time.monotonic() - t0
    hsi_recon_maps = hsi_recon.reshape_as(hsi_maps)
    rows.append({
        "experiment": "pavia1d",
        "dimension": 1,
        "method": "hsi_1d_ot_maps",
        "variant": args.hsi_architecture,
        "epsilon": None,
        "gamma": None,
        "sinkhorn_iters": None,
        "embedding_seconds": hsi_embed_seconds,
        "train_seconds": hsi_train_seconds,
        "clock_time_seconds": hsi_embed_seconds + hsi_train_seconds,
        "epochs_completed": hsi_metrics["epochs_completed"],
        "termination_reason": hsi_metrics["termination_reason"],
        "mean_w2_squared": pavia_w2_errors(data, hsi_recon_maps),
        "embedded_recon_mse": hsi_metrics["all_recon_mse"],
        "primary_error": pavia_w2_errors(data, hsi_recon_maps),
        "primary_error_name": "mean_w2_squared",
        "artifact": str(hsi_maps_path),
    })

    for eps in args.pavia_eps:
        maps, embed_seconds, maps_path = prepare_pavia_maps(
            data,
            cache_dir,
            method="ebcm",
            eps=eps,
            force=args.force_cache,
            progress_every=args.progress_every,
        )
        history_path = exp_dir / f"ebcm_eps{label_float(eps)}_history.jsonl"
        t0 = time.monotonic()
        _, metrics, recon = train_ebcm(
            as_points(data["source_points"]),
            maps,
            eps=eps,
            grid_points=as_points(data["target_points"]),
            args=args,
            device=device,
            history_path=history_path,
            run_name=f"pavia_ebcm_eps{eps:g}",
            history_time_offset=embed_seconds,
        )
        train_seconds = time.monotonic() - t0
        w2 = pavia_w2_errors(data, recon)
        rows.append({
            "experiment": "pavia1d",
            "dimension": 1,
            "method": "ebcm",
            "variant": "entropic_displacement",
            "epsilon": eps,
            "gamma": None,
            "sinkhorn_iters": None,
            "embedding_seconds": embed_seconds,
            "train_seconds": train_seconds,
            "clock_time_seconds": embed_seconds + train_seconds,
            "epochs_completed": metrics["epochs_completed"],
            "termination_reason": metrics["termination_reason"],
            "mean_w2_squared": w2,
            "embedded_recon_mse": metrics["train_recon_loss"],
            "primary_error": w2,
            "primary_error_name": "mean_w2_squared",
            "artifact": str(maps_path),
        })

    targets = [pavia_target_arrays(data, i) for i in range(len(data["records"]))]
    potentials, potential_embed_seconds, potentials_path = prepare_potentials(
        data["source_points"],
        targets,
        cache_dir / "pavia1d" / "embeddings" / (
            f"potentials_n{len(data['records'])}_support{data['source_points'].shape[0]}"
        ),
        force=args.force_cache,
        progress_label="pavia",
        progress_every=args.progress_every,
        method=args.potential_ot_method,
        eps_reg=args.potential_eps_reg,
    )
    history_path = exp_dir / "potential_history.jsonl"
    t0 = time.monotonic()
    _, potential_metrics, _ = train_generic_sae(
        potentials,
        architecture=args.potential_architecture,
        hidden_dim=args.atoms,
        top_k=args.top_k,
        lista_steps=args.lista_steps,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        sparsity_coeff=args.sparsity_coeff,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
        history_path=history_path,
        run_name="pavia_potential",
        history_time_offset=potential_embed_seconds,
        max_elapsed_seconds=args.max_elapsed_seconds,
        plateau_window=args.plateau_window,
        plateau_min_delta=args.plateau_min_delta,
        test_fraction=args.test_fraction,
    )
    potential_train_seconds = time.monotonic() - t0
    rows.append({
        "experiment": "pavia1d",
        "dimension": 1,
        "method": "potential",
        "variant": args.potential_architecture,
        "epsilon": None,
        "gamma": None,
        "sinkhorn_iters": None,
        "embedding_seconds": potential_embed_seconds,
        "train_seconds": potential_train_seconds,
        "clock_time_seconds": potential_embed_seconds + potential_train_seconds,
        "epochs_completed": potential_metrics["epochs_completed"],
        "termination_reason": potential_metrics["termination_reason"],
        "mean_w2_squared": None,
        "embedded_recon_mse": potential_metrics["all_recon_mse"],
        "primary_error": potential_metrics["all_recon_mse"],
        "primary_error_name": "embedded_potential_recon_mse",
        "artifact": str(potentials_path),
    })

    if not args.skip_heitz:
        image_dir, image_paths = write_pavia_pngs(data, exp_dir / "heitz_input", force=args.force_outputs)
        target_measures = [image_measure_1d(path) for path in image_paths]
        shared_source = exp_dir / "heitz_external" / "WassersteinDictionaryLearning"
        shared_build = exp_dir / "heitz_build"
        for gamma in args.heitz_gammas:
            method_dir = exp_dir / f"heitz_gamma{label_float(gamma)}_sink{args.heitz_sinkhorn_iters}"
            summary, wrapper_elapsed = run_heitz_trial(
                input_dir=image_dir,
                run_dir=method_dir,
                args=args,
                gamma=gamma,
                sinkhorn_iters=args.heitz_sinkhorn_iters,
                source_dir=shared_source,
                build_dir=shared_build,
                log_dir=log_dir,
            )
            w2 = evaluate_heitz_outputs(run_dir=method_dir, target_measures=target_measures, kind="1d")
            rows.append({
                "experiment": "pavia1d",
                "dimension": 1,
                "method": "heitz",
                "variant": "wasserstein_dictionary_learning",
                "epsilon": None,
                "gamma": gamma,
                "sinkhorn_iters": args.heitz_sinkhorn_iters,
                "embedding_seconds": 0.0,
                "train_seconds": summary.get("termination_elapsed_seconds", wrapper_elapsed),
                "clock_time_seconds": summary.get("termination_elapsed_seconds", wrapper_elapsed),
                "epochs_completed": summary.get("termination_iteration"),
                "termination_reason": summary.get("termination_reason"),
                "mean_w2_squared": w2,
                "embedded_recon_mse": None,
                "primary_error": w2,
                "primary_error_name": "mean_w2_squared",
                "artifact": str(method_dir),
            })

    save_table(rows, exp_dir, "timing_table_pavia1d")
    return rows


def run_mnist(args: argparse.Namespace, run_dir: Path, cache_dir: Path, device: torch.device) -> list[dict[str, Any]]:
    print("\n=== Experiment 2: MNIST ===", flush=True)
    exp_dir = run_dir / "mnist"
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_dir = exp_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    data = prepare_mnist_subset(args, cache_dir)
    source = as_points(data["source_points"])
    rows: list[dict[str, Any]] = []

    for eps in args.mnist_eps:
        maps, embed_seconds, maps_path = prepare_mnist_maps(
            data,
            cache_dir,
            eps=eps,
            force=args.force_cache,
            progress_every=args.progress_every,
        )
        history_path = exp_dir / f"ebcm_eps{label_float(eps)}_history.jsonl"
        t0 = time.monotonic()
        _, metrics, recon = train_ebcm(
            source,
            maps,
            eps=eps,
            grid_points=None,
            args=args,
            device=device,
            history_path=history_path,
            run_name=f"mnist_ebcm_eps{eps:g}",
            history_time_offset=embed_seconds,
        )
        train_seconds = time.monotonic() - t0
        w2 = mnist_w2_errors(data, recon)
        rows.append({
            "experiment": "mnist",
            "dimension": 2,
            "method": "ebcm",
            "variant": "entropic_displacement",
            "epsilon": eps,
            "gamma": None,
            "sinkhorn_iters": None,
            "embedding_seconds": embed_seconds,
            "train_seconds": train_seconds,
            "clock_time_seconds": embed_seconds + train_seconds,
            "epochs_completed": metrics["epochs_completed"],
            "termination_reason": metrics["termination_reason"],
            "mean_w2_squared": w2,
            "embedded_recon_mse": metrics["train_recon_loss"],
            "primary_error": w2,
            "primary_error_name": "mean_w2_squared",
            "artifact": str(maps_path),
        })

    targets = mnist_targets(data)
    potentials, potential_embed_seconds, potentials_path = prepare_potentials(
        data["source_points"],
        targets,
        cache_dir / "mnist" / "embeddings" / (
            f"potentials_n{len(data['records'])}_support{data['source_points'].shape[0]}"
        ),
        force=args.force_cache,
        progress_label="mnist",
        progress_every=args.progress_every,
        method=args.potential_ot_method,
        eps_reg=args.potential_eps_reg,
    )
    history_path = exp_dir / "potential_history.jsonl"
    t0 = time.monotonic()
    _, potential_metrics, _ = train_generic_sae(
        potentials,
        architecture=args.potential_architecture,
        hidden_dim=args.atoms,
        top_k=args.top_k,
        lista_steps=args.lista_steps,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        sparsity_coeff=args.sparsity_coeff,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
        history_path=history_path,
        run_name="mnist_potential",
        history_time_offset=potential_embed_seconds,
        max_elapsed_seconds=args.max_elapsed_seconds,
        plateau_window=args.plateau_window,
        plateau_min_delta=args.plateau_min_delta,
        test_fraction=args.test_fraction,
    )
    potential_train_seconds = time.monotonic() - t0
    rows.append({
        "experiment": "mnist",
        "dimension": 2,
        "method": "potential",
        "variant": args.potential_architecture,
        "epsilon": None,
        "gamma": None,
        "sinkhorn_iters": None,
        "embedding_seconds": potential_embed_seconds,
        "train_seconds": potential_train_seconds,
        "clock_time_seconds": potential_embed_seconds + potential_train_seconds,
        "epochs_completed": potential_metrics["epochs_completed"],
        "termination_reason": potential_metrics["termination_reason"],
        "mean_w2_squared": None,
        "embedded_recon_mse": potential_metrics["all_recon_mse"],
        "primary_error": potential_metrics["all_recon_mse"],
        "primary_error_name": "embedded_potential_recon_mse",
        "artifact": str(potentials_path),
    })

    if not args.skip_heitz:
        image_dir = Path(data["image_dir"])
        target_measures = [image_measure_2d(Path(record["image_path"])) for record in data["records"]]
        shared_source = exp_dir / "heitz_external" / "WassersteinDictionaryLearning"
        shared_build = exp_dir / "heitz_build"
        for gamma in args.heitz_gammas:
            method_dir = exp_dir / f"heitz_gamma{label_float(gamma)}_sink{args.heitz_sinkhorn_iters}"
            summary, wrapper_elapsed = run_heitz_trial(
                input_dir=image_dir,
                run_dir=method_dir,
                args=args,
                gamma=gamma,
                sinkhorn_iters=args.heitz_sinkhorn_iters,
                source_dir=shared_source,
                build_dir=shared_build,
                log_dir=log_dir,
            )
            w2 = evaluate_heitz_outputs(run_dir=method_dir, target_measures=target_measures, kind="2d")
            rows.append({
                "experiment": "mnist",
                "dimension": 2,
                "method": "heitz",
                "variant": "wasserstein_dictionary_learning",
                "epsilon": None,
                "gamma": gamma,
                "sinkhorn_iters": args.heitz_sinkhorn_iters,
                "embedding_seconds": 0.0,
                "train_seconds": summary.get("termination_elapsed_seconds", wrapper_elapsed),
                "clock_time_seconds": summary.get("termination_elapsed_seconds", wrapper_elapsed),
                "epochs_completed": summary.get("termination_iteration"),
                "termination_reason": summary.get("termination_reason"),
                "mean_w2_squared": w2,
                "embedded_recon_mse": None,
                "primary_error": w2,
                "primary_error_name": "mean_w2_squared",
                "artifact": str(method_dir),
            })

    save_table(rows, exp_dir, "timing_table_mnist")
    return rows


def run_gaussian(args: argparse.Namespace, run_dir: Path, cache_dir: Path, device: torch.device) -> list[dict[str, Any]]:
    print("\n=== Experiment 3: Gaussian Dimension Sweep ===", flush=True)
    exp_dir = run_dir / "gaussian"
    exp_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for dim in args.gaussian_dims:
        data = prepare_gaussian_dataset(args, cache_dir, dim)
        source = as_points(data["source_points"])
        eps_values = [0.1, 0.05, 0.2 / dim]
        for eps in eps_values:
            maps, embed_seconds, maps_path = prepare_gaussian_maps(
                data,
                cache_dir,
                dim=dim,
                eps=eps,
                force=args.force_cache,
                progress_every=args.progress_every,
            )
            history_path = exp_dir / f"dim{dim}_ebcm_eps{label_float(eps)}_history.jsonl"
            t0 = time.monotonic()
            _, metrics, _ = train_ebcm(
                source,
                maps,
                eps=eps,
                grid_points=as_points(data["grid_points"]),
                args=args,
                device=device,
                history_path=history_path,
                run_name=f"gaussian_d{dim}_ebcm_eps{eps:g}",
                history_time_offset=embed_seconds,
            )
            train_seconds = time.monotonic() - t0
            rows.append({
                "experiment": "gaussian",
                "dimension": dim,
                "method": "ebcm",
                "variant": "entropic_displacement",
                "epsilon": eps,
                "gamma": None,
                "sinkhorn_iters": None,
                "embedding_seconds": embed_seconds,
                "train_seconds": train_seconds,
                "clock_time_seconds": embed_seconds + train_seconds,
                "epochs_completed": metrics["epochs_completed"],
                "termination_reason": metrics["termination_reason"],
                "mean_w2_squared": None,
                "embedded_recon_mse": metrics["train_recon_loss"],
                "primary_error": metrics["train_recon_loss"],
                "primary_error_name": "embedded_map_recon_loss",
                "artifact": str(maps_path),
            })

        targets = [(as_points(target), None) for target in data["target_points"]]
        potentials, embed_seconds, potentials_path = prepare_potentials(
            data["source_points"],
            targets,
            cache_dir / "gaussian" / "embeddings" / (
                f"potentials_dim{dim}_n{len(data['records'])}_support{data['source_points'].shape[0]}"
            ),
            force=args.force_cache,
            progress_label=f"gaussian d={dim}",
            progress_every=args.progress_every,
            method=args.potential_ot_method,
            eps_reg=args.potential_eps_reg,
        )
        history_path = exp_dir / f"dim{dim}_potential_history.jsonl"
        t0 = time.monotonic()
        _, metrics, _ = train_generic_sae(
            potentials,
            architecture=args.potential_architecture,
            hidden_dim=args.atoms,
            top_k=args.top_k,
            lista_steps=args.lista_steps,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            sparsity_coeff=args.sparsity_coeff,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=device,
            history_path=history_path,
            run_name=f"gaussian_d{dim}_potential",
            history_time_offset=embed_seconds,
            max_elapsed_seconds=args.max_elapsed_seconds,
            plateau_window=args.plateau_window,
            plateau_min_delta=args.plateau_min_delta,
            test_fraction=args.test_fraction,
        )
        train_seconds = time.monotonic() - t0
        rows.append({
            "experiment": "gaussian",
            "dimension": dim,
            "method": "potential",
            "variant": args.potential_architecture,
            "epsilon": None,
            "gamma": None,
            "sinkhorn_iters": None,
            "embedding_seconds": embed_seconds,
            "train_seconds": train_seconds,
            "clock_time_seconds": embed_seconds + train_seconds,
            "epochs_completed": metrics["epochs_completed"],
            "termination_reason": metrics["termination_reason"],
            "mean_w2_squared": None,
            "embedded_recon_mse": metrics["all_recon_mse"],
            "primary_error": metrics["all_recon_mse"],
            "primary_error_name": "embedded_potential_recon_mse",
            "artifact": str(potentials_path),
        })

    save_table(rows, exp_dir, "timing_table_gaussian")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run timing experiments for HSI/Pavia, MNIST, and Gaussian data.")
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "experiments" / "results" / f"timing_suite_{timestamp()}")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "experiments" / "results" / "timing_cache")
    parser.add_argument("--experiment", choices=["all", "pavia1d", "mnist", "gaussian"], default="all")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--force-outputs", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)

    parser.add_argument("--atoms", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--lista-steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sparsity-coeff", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--grid-side", type=int, default=64)
    parser.add_argument("--history-every", type=int, default=1)
    parser.add_argument("--max-elapsed-seconds", type=float, default=None)
    parser.add_argument("--plateau-window", type=int, default=0)
    parser.add_argument("--plateau-min-delta", type=float, default=0.0)

    parser.add_argument("--pavia-cube-path", type=Path, default=REPO_ROOT / "datasets" / "hsi_data" / "pavia" / "data" / "pavia_cube.pt")
    parser.add_argument("--pavia-samples", type=int, default=None)
    parser.add_argument("--pavia-support-size", type=int, default=102)
    parser.add_argument("--pavia-eps", nargs="*", default=None)
    parser.add_argument("--hsi-architecture", choices=["JumpReLU_monotone", "TopKAE_monotone"], default="JumpReLU_monotone")

    parser.add_argument("--mnist-max-per-digit", type=int, default=None)
    parser.add_argument("--mnist-digits", nargs="*", default=None)
    parser.add_argument("--base-supp-size", type=int, default=None)
    parser.add_argument("--mnist-eps", nargs="*", default=None)

    parser.add_argument("--gaussian-dims", nargs="*", default=None)
    parser.add_argument("--gaussian-measures", type=int, default=None)
    parser.add_argument("--gaussian-support-size", type=int, default=None)

    parser.add_argument("--potential-architecture", choices=["TopKAE", "JumpReLU", "ReLUAE", "LISTAAE"], default="TopKAE")
    parser.add_argument("--potential-ot-method", choices=["emd", "entropic"], default="emd")
    parser.add_argument("--potential-eps-reg", type=float, default=None)

    parser.add_argument("--skip-heitz", action="store_true")
    parser.add_argument("--heitz-gammas", nargs="*", default=None)
    parser.add_argument("--heitz-sinkhorn-iters", type=int, default=None)
    parser.add_argument("--heitz-max-optim-iter", type=int, default=None)
    parser.add_argument("--heitz-loss-type", type=int, default=2)
    parser.add_argument("--heitz-scale-dict-factor", type=float, default=100.0)
    parser.add_argument("--heitz-avx", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--heitz-with-openmp", action="store_true")
    args = parser.parse_args()

    args.atoms = choose(args, "atoms")
    args.top_k = choose(args, "top_k")
    args.lista_steps = choose(args, "lista_steps")
    args.epochs = choose(args, "epochs")
    args.batch_size = choose(args, "batch_size")
    args.base_supp_size = choose(args, "base_supp_size")
    args.pavia_samples = choose(args, "pavia_samples")
    args.mnist_max_per_digit = choose(args, "mnist_max_per_digit")
    args.gaussian_measures = choose(args, "gaussian_measures")
    args.gaussian_support_size = choose(args, "gaussian_support_size")
    args.heitz_max_optim_iter = choose(args, "heitz_max_optim_iter")
    args.heitz_sinkhorn_iters = choose(args, "heitz_sinkhorn_iters")
    args.pavia_eps = parse_float_list(args.pavia_eps, PAVIA_EPSILONS)
    args.mnist_eps = parse_float_list(args.mnist_eps, MNIST_EPSILONS)
    args.gaussian_dims = parse_int_list(args.gaussian_dims, GAUSSIAN_DIMS)
    args.heitz_gammas = parse_float_list(args.heitz_gammas, HEITZ_GAMMAS)
    args.mnist_digits = parse_int_list(args.mnist_digits, list(range(10)))
    return args


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    device = resolve_device(args.device)
    run_dir = args.run_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = vars(args).copy()
    manifest["device_resolved"] = str(device)
    write_json(run_dir / "manifest.json", manifest)

    rows: list[dict[str, Any]] = []
    if args.experiment in {"all", "pavia1d"}:
        rows.extend(run_pavia1d(args, run_dir, cache_dir, device))
    if args.experiment in {"all", "mnist"}:
        rows.extend(run_mnist(args, run_dir, cache_dir, device))
    if args.experiment in {"all", "gaussian"}:
        rows.extend(run_gaussian(args, run_dir, cache_dir, device))

    paths = save_table(rows, run_dir, "timing_table_all")
    write_json(run_dir / "summary.json", {
        "run_dir": str(run_dir),
        "cache_dir": str(cache_dir),
        "num_rows": len(rows),
        "tables": paths,
    })
    print(f"\nTiming suite complete: {run_dir}", flush=True)
    print(f"Combined CSV: {paths['csv']}", flush=True)


if __name__ == "__main__":
    main()
