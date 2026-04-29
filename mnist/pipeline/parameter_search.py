"""
LISTA-only hyperparameter search with staged pruning.

This script samples LISTA+ trials over two hyperparameters:
    - sparsity coefficient c in [1e-4, 1]
    - Gibbs epsilon eps in [2.5e-3, 2.5e-1]

Default search:
    - 40 random trials
    - stage budgets: 100, 200, 200, 2500 epochs
    - prune to the top 50% after each of the first three stages

Trials are scored on two metrics:
    1. reconstruction loss on the held-out test split
    2. digit classification accuracy from a frozen linear probe on learned codes

Pruning uses "good on both metrics" as the first filter: keep the intersection of
the top keep-fraction by reconstruction and by probe accuracy. If that
intersection is smaller than the target survivor count, backfill by a combined
rank that favors balanced performance across both metrics.
"""

import argparse
import json
import math
import os
import time
import traceback
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from mnist_sae_models import DisplacementFieldSAE
from train_mnist_sae import compute_losses


DEFAULT_STAGE_PREFIX_EPOCHS = [100, 200, 200]


def _fmt(x):
    """Compact string formatting for run names."""
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def load_with_labels(data_dir, digits=None):
    """
    Load base measure, transport maps, and digit labels.

    Returns:
        X: (n, 2)
        maps: (N, n, 2)
        labels: (N,)
    """
    data_dir = Path(data_dir)
    X = torch.load(data_dir / "base_measure.pt", map_location="cpu").float()

    if digits is None:
        digits = sorted(
            int(p.name.split("_")[1])
            for p in data_dir.iterdir()
            if p.is_dir() and p.name.startswith("digit_")
        )

    all_maps = []
    all_labels = []
    for d in digits:
        path = data_dir / f"digit_{d}" / "mappings.pt"
        if not path.exists():
            print(f"  Warning: {path} not found, skipping digit {d}")
            continue
        maps_d = torch.load(path, map_location="cpu").float()
        all_maps.append(maps_d)
        all_labels.append(torch.full((maps_d.shape[0],), d, dtype=torch.long))
        print(f"  Loaded digit {d}: {maps_d.shape[0]} maps")

    maps = torch.cat(all_maps, dim=0)
    labels = torch.cat(all_labels, dim=0)
    print(f"Total: {maps.shape[0]} maps with labels, support size n={maps.shape[1]}")
    return X, maps, labels


def make_stratified_split(maps, labels, test_fraction=0.1, seed=42):
    """
    Stratified train/test split so every digit appears in both splits whenever
    possible.
    """
    gen = torch.Generator().manual_seed(seed)
    train_idx = []
    test_idx = []

    for digit in sorted(torch.unique(labels).tolist()):
        idx = (labels == digit).nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(len(idx), generator=gen)]

        if len(perm) <= 1:
            n_test = 0
        else:
            n_test = int(round(len(perm) * test_fraction))
            n_test = min(max(n_test, 1), len(perm) - 1)

        test_idx.append(perm[:n_test])
        train_idx.append(perm[n_test:])

    train_idx = torch.cat(train_idx, dim=0)
    test_idx = torch.cat(test_idx, dim=0)

    train_idx = train_idx[torch.randperm(len(train_idx), generator=gen)]
    test_idx = test_idx[torch.randperm(len(test_idx), generator=gen)]

    split = {
        "train_maps": maps[train_idx],
        "train_labels": labels[train_idx],
        "test_maps": maps[test_idx],
        "test_labels": labels[test_idx],
    }
    print(f"Split: {split['train_maps'].shape[0]} train, {split['test_maps'].shape[0]} test")
    return split


def make_map_loader(maps, batch_size, shuffle):
    """Wrap raw maps in a DataLoader for the existing training utilities."""
    dataset = TensorDataset(maps)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _loader_to_maps_only(loader):
    """
    Convert a TensorDataset-backed loader yielding `(maps,)` tuples into a simple
    iterator interface expected by compute_losses.
    """
    for batch in loader:
        yield batch[0]


def _anneal_eps(epoch, total_epochs, eps_start, eps_end, fraction):
    """Geometric anneal from eps_start to eps_end over the first fraction."""
    anneal_epochs = max(int(total_epochs * fraction), 1)
    progress = min(epoch / anneal_epochs, 1.0)
    log_eps = (1.0 - progress) * math.log(eps_start) + progress * math.log(eps_end)
    return math.exp(log_eps)


def build_model(X, cfg, device):
    """Build the LISTA+ displacement model for one trial."""
    model = DisplacementFieldSAE(
        X,
        m=cfg["m"],
        eps=cfg["eps"],
        grid_side=cfg["grid_side"],
        lista_steps=cfg["lista_steps"],
        activation_type=cfg["activation_type"],
        normalize_atoms=cfg["normalize_atoms"],
        grid_points=None,
        atoms_type=cfg["atoms_type"],
        n_sinkhorn=cfg["n_sinkhorn"],
        topk_k=cfg["topk_k"],
        per_atom_gain=cfg["per_atom_gain"],
        lateral_init=cfg["lateral_init"],
    ).to(device)
    return model


def make_optimizer(model, cfg):
    """Construct the optimizer declared in the trial config."""
    if cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )
    return torch.optim.Adam(model.parameters(), lr=cfg["lr"])


def make_scheduler(optimizer, cfg):
    """Optional cosine scheduler over the full target training horizon."""
    if cfg.get("scheduler", "none") != "cosine":
        return None
    eta_min = cfg.get("lr_min") or 0.0
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["final_epochs"],
        eta_min=eta_min,
    )


def move_optimizer_state_to_device(optimizer, device):
    """Move any tensor-valued optimizer state to the model device."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def train_stage(model, train_loader, cfg, device, stage_epochs,
                epochs_completed=0, optimizer_state=None, scheduler_state=None):
    """
    Continue training a model for one search stage and return training state.
    """
    optimizer = make_optimizer(model, cfg)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        move_optimizer_state_to_device(optimizer, device)

    scheduler = make_scheduler(optimizer, cfg)
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    c = cfg["sparsity_coeff"]
    n = model.n
    anneal = bool(cfg.get("eps_anneal", False))
    eps_start = cfg.get("eps_start")
    eps_end = cfg.get("eps_end")
    anneal_fraction = cfg.get("eps_anneal_fraction", 0.5)
    sparsity_warmup = cfg.get("sparsity_warmup", 0)
    total_target_epochs = cfg["final_epochs"]

    model.train()
    for local_epoch in range(stage_epochs):
        epoch = epochs_completed + local_epoch

        if anneal and eps_start is not None and eps_end is not None:
            cur_eps = _anneal_eps(epoch, total_target_epochs, eps_start, eps_end, anneal_fraction)
            model.atoms_module.set_eps(cur_eps)
        else:
            cur_eps = float(model.atoms_module.eps)

        if sparsity_warmup > 0 and epoch < sparsity_warmup:
            c_eff = c * (epoch + 1) / sparsity_warmup
        else:
            c_eff = c

        epoch_loss = 0.0
        epoch_steps = 0
        for (T_batch,) in train_loader:
            T_batch = T_batch.to(device).float()
            T_hat, lam = model(T_batch)

            diff_sq = ((T_batch - T_hat) ** 2).sum(dim=(1, 2))
            recon = 0.5 * diff_sq.mean() / n
            sparsity = c_eff * lam.abs().sum(dim=1).mean()
            loss = recon + sparsity

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1

        if scheduler is not None:
            scheduler.step()

        epoch_num = epoch + 1
        if epoch_num == 1 or epoch_num % 10 == 0 or epoch_num == epochs_completed + stage_epochs:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"[{cfg['name']}] Epoch {epoch_num:4d}/{total_target_epochs}  "
                f"loss={epoch_loss / max(epoch_steps, 1):.6f}  "
                f"lr={lr_now:.2e}  eps={cur_eps:.4g}  c={c_eff:.4g}",
                flush=True,
            )

    return {
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epochs_completed": epochs_completed + stage_epochs,
    }


@torch.no_grad()
def encode_dataset(model, maps, batch_size, device):
    """Encode all maps into sparse codes."""
    model.eval()
    codes = []
    for i in range(0, maps.shape[0], batch_size):
        T = maps[i:i + batch_size].to(device).float()
        _, lam = model(T)
        codes.append(lam.cpu())
    return torch.cat(codes, dim=0)


def fit_linear_probe(train_codes, train_labels, test_codes, test_labels, ridge=1e-3):
    """
    Fit a simple ridge-regularized linear classifier on frozen codes.

    Returns train/test accuracy along with the digit classes used.
    """
    classes = sorted(torch.unique(train_labels).tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    y_train = torch.tensor([class_to_idx[int(y)] for y in train_labels], dtype=torch.long)
    y_test = torch.tensor([class_to_idx[int(y)] for y in test_labels], dtype=torch.long)
    n_classes = len(classes)

    mean = train_codes.mean(dim=0, keepdim=True)
    std = train_codes.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-6)
    x_train = (train_codes - mean) / std
    x_test = (test_codes - mean) / std

    x_train = torch.cat([x_train, torch.ones(x_train.shape[0], 1)], dim=1)
    x_test = torch.cat([x_test, torch.ones(x_test.shape[0], 1)], dim=1)

    y_onehot = F.one_hot(y_train, num_classes=n_classes).float()
    gram = x_train.T @ x_train
    reg = ridge * torch.eye(gram.shape[0], dtype=gram.dtype)
    reg[-1, -1] = 0.0
    rhs = x_train.T @ y_onehot

    try:
        weights = torch.linalg.solve(gram + reg, rhs)
    except RuntimeError:
        weights = torch.linalg.pinv(gram + reg) @ rhs

    train_logits = x_train @ weights
    test_logits = x_test @ weights
    train_acc = (train_logits.argmax(dim=1) == y_train).float().mean().item()
    test_acc = (test_logits.argmax(dim=1) == y_test).float().mean().item()
    return {
        "classes": classes,
        "probe_train_accuracy": train_acc,
        "probe_test_accuracy": test_acc,
    }


def evaluate_trial(model, split, cfg, device):
    """Compute reconstruction and code-classification metrics."""
    train_loader = make_map_loader(split["train_maps"], cfg["batch_size"], shuffle=False)
    test_loader = make_map_loader(split["test_maps"], cfg["batch_size"], shuffle=False)

    train_metrics = compute_losses(model, _loader_to_maps_only(train_loader), cfg["sparsity_coeff"], device)
    test_metrics = compute_losses(model, _loader_to_maps_only(test_loader), cfg["sparsity_coeff"], device)

    train_codes = encode_dataset(model, split["train_maps"], cfg["batch_size"], device)
    test_codes = encode_dataset(model, split["test_maps"], cfg["batch_size"], device)
    probe_metrics = fit_linear_probe(
        train_codes,
        split["train_labels"],
        test_codes,
        split["test_labels"],
        ridge=cfg["probe_ridge"],
    )

    return {
        "train_recon_loss": train_metrics["recon_loss"],
        "train_total_loss": train_metrics["total_loss"],
        "train_mean_l1": train_metrics["mean_l1"],
        "train_mean_active": train_metrics["mean_active"],
        "test_recon_loss": test_metrics["recon_loss"],
        "test_total_loss": test_metrics["total_loss"],
        "test_mean_l1": test_metrics["mean_l1"],
        "test_mean_active": test_metrics["mean_active"],
        **probe_metrics,
    }


def sample_trials(num_trials, seed, epochs_total):
    """Sample random LISTA+ trials with log-uniform c and eps."""
    rng = torch.Generator().manual_seed(seed)

    def sample_log_uniform(lo, hi):
        log_lo = math.log(lo)
        log_hi = math.log(hi)
        u = torch.rand(1, generator=rng).item()
        return math.exp(log_lo + u * (log_hi - log_lo))

    trials = []
    for idx in range(num_trials):
        eps = sample_log_uniform(0.0025, 0.25)
        c = sample_log_uniform(1e-4, 1.0)
        trial = dict(
            name=f"lista_search_{idx + 1:03d}",
            trial_index=idx + 1,
            atoms_type="gibbs",
            eps=eps,
            activation_type="relu",
            sparsity_coeff=c,
            normalize_atoms=True,
            per_atom_gain=True,
            lateral_init="damped_identity",
            topk_k=3,
            eps_anneal=False,
            method="displacement",
            m=30,
            grid_side=64,
            lista_steps=20,
            batch_size=128,
            lr=1e-3,
            optimizer="adamw",
            weight_decay=0.1,
            scheduler="none",
            lr_min=0.0,
            n_sinkhorn=30,
            final_epochs=epochs_total,
            probe_ridge=1e-3,
        )
        trials.append(trial)
    return trials


def resolve_stage_epochs(total_epochs, stage_epochs=None):
    """
    Resolve the staged budget from a total epoch budget.

    If stage_epochs is omitted, use the default multi-fidelity schedule
    [100, 200, 200, total_epochs - 500].

    If stage_epochs is provided and sums to less than total_epochs, the
    remainder is appended as the final stage. If it sums exactly, it is used
    as-is. Sums larger than total_epochs are rejected.
    """
    if total_epochs <= 0:
        raise ValueError("--epochs must be positive.")

    if stage_epochs is None:
        prefix = list(DEFAULT_STAGE_PREFIX_EPOCHS)
    else:
        prefix = list(stage_epochs)

    used = sum(prefix)
    if used > total_epochs:
        raise ValueError(
            f"Stage budgets sum to {used}, which exceeds total epochs {total_epochs}."
        )
    if used == total_epochs:
        return prefix

    remainder = total_epochs - used
    if remainder <= 0:
        raise ValueError("Final stage must have a positive epoch budget.")
    return prefix + [remainder]


def summarize_trials(trials):
    """Pretty-print the sampled trial table."""
    print(f"{'#':>3} {'name':<20} {'eps':>10} {'c':>10} {'lista':>6}")
    print("-" * 58)
    for trial in trials:
        print(
            f"{trial['trial_index']:>3} {trial['name']:<20} "
            f"{trial['eps']:>10.4g} {trial['sparsity_coeff']:>10.4g} {trial['lista_steps']:>6d}"
        )
    print("-" * 58, flush=True)


def rank_and_select(stage_results, keep_fraction):
    """
    Rank trials by reconstruction and probe accuracy, then keep a balanced top
    fraction.
    """
    successful = [r for r in stage_results if r.get("status") == "ok"]
    if not successful:
        return []

    n_keep = max(1, math.ceil(len(successful) * keep_fraction))

    by_recon = sorted(successful, key=lambda r: (r["test_recon_loss"], -r["probe_test_accuracy"]))
    by_acc = sorted(successful, key=lambda r: (-r["probe_test_accuracy"], r["test_recon_loss"]))

    recon_rank = {r["name"]: i for i, r in enumerate(by_recon)}
    acc_rank = {r["name"]: i for i, r in enumerate(by_acc)}
    top_recon = {r["name"] for r in by_recon[:n_keep]}
    top_acc = {r["name"] for r in by_acc[:n_keep]}

    combined = []
    for result in successful:
        name = result["name"]
        result = dict(result)
        result["recon_rank"] = recon_rank[name]
        result["probe_rank"] = acc_rank[name]
        result["in_top_recon"] = name in top_recon
        result["in_top_probe"] = name in top_acc
        result["balanced_rank"] = max(result["recon_rank"], result["probe_rank"])
        result["rank_sum"] = result["recon_rank"] + result["probe_rank"]
        combined.append(result)

    balanced_sorted = sorted(
        combined,
        key=lambda r: (
            not (r["in_top_recon"] and r["in_top_probe"]),
            r["balanced_rank"],
            r["rank_sum"],
            r["test_recon_loss"],
            -r["probe_test_accuracy"],
        ),
    )

    return balanced_sorted[:n_keep]


def resolve_worker_device(device_str, worker_id):
    """Resolve the requested device for one worker process."""
    if device_str == "auto":
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)
            return torch.device("cuda:0")
        return torch.device("cpu")

    if device_str == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        print("CUDA not available, falling back to CPU", flush=True)
        return torch.device("cpu")

    if device_str == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("MPS not available, falling back to CPU", flush=True)
        return torch.device("cpu")

    return torch.device("cpu")


def worker(worker_id, trial_cfg, stage_index, stage_epochs, data_dir, output_dir,
           digits, test_fraction, split_seed, result_path, device_str):
    """
    Train one trial for one stage and write a stage metrics JSON.
    """
    torch.manual_seed(split_seed + trial_cfg["trial_index"] + stage_index)
    device = resolve_worker_device(device_str, worker_id)

    trial_dir = Path(output_dir) / trial_cfg["name"]
    trial_dir.mkdir(parents=True, exist_ok=True)
    state_path = trial_dir / "latest_state.pt"
    latest_model_path = trial_dir / "model_latest.pt"

    print(
        f"[worker {worker_id}] stage {stage_index} starting {trial_cfg['name']} "
        f"(device={device}, +{stage_epochs} epochs)",
        flush=True,
    )

    try:
        X, maps, labels = load_with_labels(data_dir, digits=digits)
        split = make_stratified_split(maps, labels, test_fraction=test_fraction, seed=split_seed)
        train_loader = make_map_loader(split["train_maps"], trial_cfg["batch_size"], shuffle=True)

        model = build_model(X, trial_cfg, device)
        optimizer_state = None
        scheduler_state = None
        epochs_completed = 0

        if state_path.exists():
            checkpoint = torch.load(state_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state"])
            optimizer_state = checkpoint.get("optimizer_state")
            scheduler_state = checkpoint.get("scheduler_state")
            epochs_completed = int(checkpoint.get("epochs_completed", 0))

        t0 = time.time()
        train_state = train_stage(
            model,
            train_loader,
            trial_cfg,
            device,
            stage_epochs=stage_epochs,
            epochs_completed=epochs_completed,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
        )
        elapsed = time.time() - t0

        metrics = evaluate_trial(model, split, trial_cfg, device)
        state_payload = {
            "model_state": model.state_dict(),
            **train_state,
            "trial_cfg": trial_cfg,
        }
        torch.save(state_payload, state_path)
        torch.save(model.state_dict(), latest_model_path)
        torch.save(
            model.state_dict(),
            trial_dir / f"model_stage{stage_index}_epoch{train_state['epochs_completed']}.pt",
        )

        result = {
            "status": "ok",
            "name": trial_cfg["name"],
            "trial_index": trial_cfg["trial_index"],
            "worker_id": worker_id,
            "device": str(device),
            "stage_index": stage_index,
            "stage_epochs": stage_epochs,
            "epochs_completed": train_state["epochs_completed"],
            "elapsed_seconds": round(elapsed, 1),
            "eps": trial_cfg["eps"],
            "sparsity_coeff": trial_cfg["sparsity_coeff"],
            "normalize_atoms": trial_cfg["normalize_atoms"],
            "per_atom_gain": trial_cfg["per_atom_gain"],
            "lateral_init": trial_cfg["lateral_init"],
            **metrics,
        }
    except Exception as exc:
        result = {
            "status": "failed",
            "name": trial_cfg["name"],
            "trial_index": trial_cfg["trial_index"],
            "worker_id": worker_id,
            "device": str(device),
            "stage_index": stage_index,
            "stage_epochs": stage_epochs,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    if result["status"] == "ok":
        print(
            f"[worker {worker_id}] stage {stage_index} finished {trial_cfg['name']}  "
            f"recon={result['test_recon_loss']:.6f}  "
            f"probe_acc={result['probe_test_accuracy']:.4f}",
            flush=True,
        )
    else:
        print(
            f"[worker {worker_id}] stage {stage_index} failed {trial_cfg['name']}: {result['error']}",
            flush=True,
        )


def run_stage(stage_trials, stage_index, stage_epochs, data_dir, output_dir,
              gpu_ids, digits, test_fraction, split_seed, device):
    """Dispatch one pruning stage across the available worker slots."""
    out = Path(output_dir)
    results = []

    n_workers = len(gpu_ids)
    n_waves = (len(stage_trials) + n_workers - 1) // n_workers
    for wave_start in range(0, len(stage_trials), n_workers):
        wave = stage_trials[wave_start:wave_start + n_workers]
        print(f"\n{'=' * 72}")
        print(
            f"Stage {stage_index}: wave {wave_start // n_workers + 1}/{n_waves}  "
            f"launching {len(wave)} trial(s) on {device} worker slot(s) {gpu_ids[:len(wave)]}"
        )
        print(f"{'=' * 72}\n", flush=True)

        processes = []
        result_paths = []
        for i, trial in enumerate(wave):
            worker_id = gpu_ids[i]
            result_path = out / f"stage{stage_index}_{trial['name']}_metrics.json"
            result_paths.append(result_path)
            p = mp.Process(
                target=worker,
                args=(
                    worker_id,
                    trial,
                    stage_index,
                    stage_epochs,
                    data_dir,
                    output_dir,
                    digits,
                    test_fraction,
                    split_seed,
                    str(result_path),
                    device,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        for result_path in result_paths:
            if result_path.exists():
                with open(result_path, "r") as f:
                    results.append(json.load(f))
            else:
                print(f"  Warning: missing result file {result_path}", flush=True)

    return results


def run_lista_search(data_dir, output_dir, gpu_ids, digits=None, test_fraction=0.1,
                     seed=42, num_trials=40, epochs=3000, stage_epochs=None,
                     keep_fraction=0.5, dry_run=False, device="auto"):
    """Main staged LISTA search routine."""
    stage_epochs = resolve_stage_epochs(total_epochs=epochs, stage_epochs=stage_epochs)
    total_epochs = sum(stage_epochs)

    trials = sample_trials(num_trials=num_trials, seed=seed, epochs_total=total_epochs)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "trials.json", "w") as f:
        json.dump(trials, f, indent=2, default=str)

    print(
        f"\nLISTA search: {len(trials)} sampled trials, "
        f"stages={stage_epochs}, total_epochs={total_epochs}, keep_fraction={keep_fraction}"
    )
    summarize_trials(trials)

    if dry_run:
        print("Dry run: wrote trials.json, no workers launched.")
        return []

    trial_lookup = {trial["name"]: trial for trial in trials}
    active_trials = list(trials)
    all_stage_results = []

    for stage_index, stage_len in enumerate(stage_epochs, start=1):
        print(
            f"\nStarting stage {stage_index}/{len(stage_epochs)} "
            f"for {len(active_trials)} trial(s), +{stage_len} epochs each",
            flush=True,
        )
        stage_results = run_stage(
            active_trials,
            stage_index=stage_index,
            stage_epochs=stage_len,
            data_dir=data_dir,
            output_dir=output_dir,
            gpu_ids=gpu_ids,
            digits=digits,
            test_fraction=test_fraction,
            split_seed=seed,
            device=device,
        )
        all_stage_results.append(stage_results)

        with open(out / f"stage{stage_index}_results.json", "w") as f:
            json.dump(stage_results, f, indent=2, default=str)

        successful = [r for r in stage_results if r.get("status") == "ok"]
        if not successful:
            print(f"All trials failed in stage {stage_index}.", flush=True)
            break

        print(f"\nStage {stage_index} summary:")
        print(f"{'name':<20} {'recon':>12} {'probe_acc':>10} {'epochs':>8}")
        for result in sorted(successful, key=lambda r: (r["test_recon_loss"], -r["probe_test_accuracy"])):
            print(
                f"{result['name']:<20} {result['test_recon_loss']:>12.6f} "
                f"{result['probe_test_accuracy']:>10.4f} {result['epochs_completed']:>8d}"
            )

        if stage_index == len(stage_epochs):
            active_trials = [trial_lookup[result["name"]] for result in successful]
            break

        survivors = rank_and_select(stage_results, keep_fraction=keep_fraction)
        survivor_names = [result["name"] for result in survivors]
        active_trials = [trial_lookup[name] for name in survivor_names]

        with open(out / f"stage{stage_index}_survivors.json", "w") as f:
            json.dump(survivors, f, indent=2, default=str)

        print(
            f"\nStage {stage_index} survivors ({len(active_trials)} / {len(successful)}): "
            f"{', '.join(survivor_names)}",
            flush=True,
        )

    final_results = all_stage_results[-1] if all_stage_results else []
    successful_final = [r for r in final_results if r.get("status") == "ok"]
    successful_final = sorted(
        successful_final,
        key=lambda r: (-r["probe_test_accuracy"], r["test_recon_loss"]),
    )

    with open(out / "all_stage_results.json", "w") as f:
        json.dump(all_stage_results, f, indent=2, default=str)
    with open(out / "final_results.json", "w") as f:
        json.dump(successful_final, f, indent=2, default=str)

    if successful_final:
        best = successful_final[0]
        with open(out / "best_trial.json", "w") as f:
            json.dump(best, f, indent=2, default=str)
        print(
            f"\nBest final trial: {best['name']}  "
            f"probe_acc={best['probe_test_accuracy']:.4f}  "
            f"recon={best['test_recon_loss']:.6f}",
            flush=True,
        )

    return successful_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LISTA-only random hyperparameter search with staged pruning."
    )
    parser.add_argument("--data_dir", type=str, default="datasets/mnist_ot")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="2D_WDL_results/lista_search",
    )
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument(
        "--epochs",
        type=int,
        default=3000,
        help="Total epochs for surviving final trials, mirroring the long experiment CLI.",
    )
    parser.add_argument("--num_trials", type=int, default=40)
    parser.add_argument(
        "--stage_epochs",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional stage lengths. If omitted, uses 100 200 200 and assigns the "
            "remaining epochs to the final stage so the total matches --epochs."
        ),
    )
    parser.add_argument(
        "--keep_fraction",
        type=float,
        default=0.5,
        help="Fraction of successful trials kept after each non-final stage.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Worker device: auto uses CUDA if available, otherwise CPU.",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    run_lista_search(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        gpu_ids=args.gpu_ids,
        test_fraction=args.test_fraction,
        seed=args.seed,
        num_trials=args.num_trials,
        epochs=args.epochs,
        stage_epochs=args.stage_epochs,
        keep_fraction=args.keep_fraction,
        dry_run=args.dry_run,
        device=args.device,
    )
