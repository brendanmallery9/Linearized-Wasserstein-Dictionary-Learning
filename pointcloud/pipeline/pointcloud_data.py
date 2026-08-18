"""
Dataset utilities for 3D point-cloud transport-map SAEs.

Mirrors mnist/pipeline/mnist_ot_data.py but reads class folders named
class_<name>/mappings.pt instead of digit_<d>/mappings.pt.

Files produced by prepare_pointcloud_ot.py:
    <data_dir>/base_measure.pt          # (n, 3)
    <data_dir>/class_<name>/mappings.pt # (N_c, n, 3) per class
"""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split


class TransportMapDataset(Dataset):
    """Plain wrapper over a (N, n, d) tensor of stacked transport maps."""
    def __init__(self, maps, aux_maps=None):
        self.maps = maps
        self.aux_maps = aux_maps
        if aux_maps is not None and aux_maps.shape[0] != maps.shape[0]:
            raise ValueError(
                f"aux_maps must have same first dimension as maps: "
                f"{aux_maps.shape[0]} vs {maps.shape[0]}"
            )

    def __len__(self):
        return self.maps.shape[0]

    def __getitem__(self, idx):
        if self.aux_maps is not None:
            return self.maps[idx], self.aux_maps[idx]
        return self.maps[idx]


def list_classes(data_dir):
    """List all class names that have a mappings.pt file in `data_dir`."""
    data_dir = Path(data_dir)
    return sorted(
        p.name[len("class_"):]
        for p in data_dir.iterdir()
        if p.is_dir() and p.name.startswith("class_")
        and (p / "mappings.pt").exists()
    )


def load_transport_maps(data_dir, classes=None):
    """
    Load base measure and concatenated transport maps for the requested classes.

    Supports two layouts:
      Class-based:
          <data_dir>/base_measure.pt
          <data_dir>/class_<name>/mappings.pt   (one per class)
      Flat:
          <data_dir>/base_measure.pt
          <data_dir>/maps/maps.pt               (single (N, n, d) tensor)

    Returns:
        X:      (n, d) base measure support
        maps:   (N, n, d) all transport maps stacked
    """
    data_dir = Path(data_dir)
    X = torch.load(data_dir / "base_measure.pt", map_location="cpu")

    flat_path = data_dir / "maps" / "maps.pt"
    if flat_path.exists():
        maps = torch.load(flat_path, map_location="cpu")
        if classes is not None:
            print(f"  Note: flat layout at {flat_path}; ignoring classes={classes}")
        print(f"Total: {maps.shape[0]} maps, support size n={maps.shape[1]}, dim={maps.shape[2]}")
        return X, maps

    if classes is None:
        classes = list_classes(data_dir)

    all_maps = []
    for c in classes:
        path = data_dir / f"class_{c}" / "mappings.pt"
        if path.exists():
            m = torch.load(path, map_location="cpu")
            all_maps.append(m)
            print(f"  Loaded class {c!r}: {m.shape[0]} maps")
        else:
            print(f"  Warning: {path} not found, skipping class {c!r}")

    maps = torch.cat(all_maps, dim=0)
    print(f"Total: {maps.shape[0]} maps, support size n={maps.shape[1]}, dim={maps.shape[2]}")
    return X, maps


def load_with_labels(data_dir, classes=None):
    """
    Load base measure, transport maps, and integer class labels.

    Returns:
        X:        (n, d)
        maps:     (N, n, d)
        labels:   (N,) long, indexing into `classes`
        classes:  list[str], the class names corresponding to label indices
    """
    data_dir = Path(data_dir)
    X = torch.load(data_dir / "base_measure.pt", map_location="cpu").float()

    if classes is None:
        classes = list_classes(data_dir)

    all_maps = []
    all_labels = []
    for idx, c in enumerate(classes):
        path = data_dir / f"class_{c}" / "mappings.pt"
        if not path.exists():
            print(f"  Warning: {path} not found, skipping class {c!r}")
            continue
        m = torch.load(path, map_location="cpu").float()
        all_maps.append(m)
        all_labels.append(torch.full((m.shape[0],), idx, dtype=torch.long))
        print(f"  Loaded class {c!r} (label={idx}): {m.shape[0]} maps")

    maps = torch.cat(all_maps, dim=0)
    labels = torch.cat(all_labels, dim=0)
    print(f"Total: {maps.shape[0]} labeled maps, n={maps.shape[1]}, dim={maps.shape[2]}")
    return X, maps, labels, classes


def load_raw_clouds(data_dir, classes=None):
    """
    Load the raw target point clouds (mu_i) in the same order as load_with_labels.

    These are the pre-OT ground-truth measures sampled from each shape (saved as
    class_<name>/raw_clouds.pt by prepare_pointcloud_ot.py).  They are used as
    the common target for native Wasserstein / Chamfer reconstruction scoring.

    Returns:
        raw_clouds: (N, cloud_supp_size, 3) tensor, aligned row-for-row with the
            maps returned by load_with_labels(data_dir, classes); or None if any
            requested class is missing raw_clouds.pt (older datasets that predate
            raw-cloud persistence).
    """
    data_dir = Path(data_dir)
    if classes is None:
        classes = list_classes(data_dir)

    all_clouds = []
    for c in classes:
        path = data_dir / f"class_{c}" / "raw_clouds.pt"
        if not path.exists():
            print(f"  Note: {path} not found; raw clouds unavailable for this "
                  f"dataset (regenerate with the updated prepare script).")
            return None
        all_clouds.append(torch.load(path, map_location="cpu").float())

    if not all_clouds:
        return None
    raw_clouds = torch.cat(all_clouds, dim=0)
    print(f"Loaded raw target clouds: {tuple(raw_clouds.shape)}")
    return raw_clouds


def make_dataloaders(maps, batch_size=64, test_fraction=0.1, seed=42,
                     aux_maps=None):
    """Split maps into train/test loaders."""
    dataset = TransportMapDataset(maps, aux_maps=aux_maps)
    N = len(dataset)
    n_test = int(N * test_fraction)
    n_train = N - n_test

    gen = torch.Generator().manual_seed(seed)
    train_set, test_set = random_split(dataset, [n_train, n_test], generator=gen)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print(f"Split: {n_train} train, {n_test} test")
    return train_loader, test_loader


# ============================================================
# Deterministic, shareable train/test splits
# ============================================================

def make_split_indices(labels, test_fraction=0.1, seed=42, split_path=None,
                       stratified=True):
    """
    Build (or load) a fixed train/test index split so several methods can be
    compared on exactly the same data partition.

    Behavior:
      * If `split_path` exists, load `train_indices` / `test_indices` from it
        (the saved seed / test_fraction are informational; the indices win).
      * Otherwise create a fresh split.  A stratified split (per-class
        proportions preserved) is used when `stratified=True` and labels are
        available; otherwise a plain random permutation split.
      * When `split_path` is given and did not exist, the resulting split is
        written to disk as JSON with keys: seed, test_fraction, train_indices,
        test_indices (plus `stratified`).

    Args:
        labels: (N,) array-like of integer class labels (torch tensor, numpy
            array, or list).  May be None for a non-stratified split, in which
            case `n_samples` is inferred from... nothing -- so labels is
            required to know N.  Pass the label vector even for non-stratified
            splits.
        test_fraction: fraction of samples held out for test.
        seed: RNG seed controlling the split.
        split_path: optional path to load-from / save-to (JSON).
        stratified: prefer an sklearn stratified split.

    Returns:
        train_indices, test_indices: two lists of Python ints.
    """
    if split_path is not None:
        split_path = Path(split_path)
        if split_path.exists():
            with open(split_path) as f:
                payload = json.load(f)
            train_indices = list(map(int, payload["train_indices"]))
            test_indices = list(map(int, payload["test_indices"]))
            print(f"Loaded split from {split_path}: "
                  f"{len(train_indices)} train, {len(test_indices)} test")
            return train_indices, test_indices

    if labels is None:
        raise ValueError("labels is required to build a new split (need N and, "
                         "for stratified splits, the class of each sample)")

    if hasattr(labels, "tolist"):
        labels_list = [int(v) for v in labels.tolist()]
    else:
        labels_list = [int(v) for v in labels]
    n_samples = len(labels_list)
    all_indices = list(range(n_samples))

    stratify = labels_list if stratified else None
    try:
        from sklearn.model_selection import train_test_split
        train_indices, test_indices = train_test_split(
            all_indices, test_size=test_fraction, random_state=seed,
            shuffle=True, stratify=stratify,
        )
    except Exception as exc:  # sklearn missing, or stratification infeasible
        if stratified:
            print(f"  Stratified split unavailable ({exc}); "
                  f"falling back to random split")
        gen = torch.Generator().manual_seed(int(seed))
        perm = torch.randperm(n_samples, generator=gen).tolist()
        n_test = int(round(n_samples * test_fraction))
        test_indices = perm[:n_test]
        train_indices = perm[n_test:]

    train_indices = sorted(int(i) for i in train_indices)
    test_indices = sorted(int(i) for i in test_indices)
    print(f"Created split (seed={seed}, test_fraction={test_fraction}, "
          f"stratified={stratified}): {len(train_indices)} train, "
          f"{len(test_indices)} test")

    if split_path is not None:
        split_path = Path(split_path)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with open(split_path, "w") as f:
            json.dump({
                "seed": int(seed),
                "test_fraction": float(test_fraction),
                "stratified": bool(stratified),
                "train_indices": train_indices,
                "test_indices": test_indices,
            }, f)
        print(f"Saved split to {split_path}")

    return train_indices, test_indices


def make_dataloaders_from_indices(maps, train_indices, test_indices,
                                  batch_size=64, aux_maps=None):
    """
    Build train/test dataloaders from *fixed* indices (deterministic).

    Mirrors `make_dataloaders`, but instead of drawing a fresh random split it
    partitions `maps` (and optional `aux_maps`) using the supplied index lists.
    Train loader shuffles; test loader does not.

    Args:
        maps: (N, n, d) stacked transport maps.
        train_indices, test_indices: index lists (e.g. from make_split_indices).
        batch_size: dataloader batch size.
        aux_maps: optional (N, n, d) auxiliary tensor (e.g. whitened residuals).

    Returns:
        train_loader, test_loader
    """
    dataset = TransportMapDataset(maps, aux_maps=aux_maps)
    train_set = Subset(dataset, list(train_indices))
    test_set = Subset(dataset, list(test_indices))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print(f"Split (fixed indices): {len(train_set)} train, {len(test_set)} test")
    return train_loader, test_loader
