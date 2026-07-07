"""
Dataset utilities for 3D point-cloud transport-map SAEs.

Mirrors mnist/pipeline/mnist_ot_data.py but reads class folders named
class_<name>/mappings.pt instead of digit_<d>/mappings.pt.

Files produced by prepare_pointcloud_ot.py:
    <data_dir>/base_measure.pt          # (n, 3)
    <data_dir>/class_<name>/mappings.pt # (N_c, n, 3) per class
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split


class TransportMapDataset(Dataset):
    """Plain wrapper over a (N, n, d) tensor of stacked transport maps."""
    def __init__(self, maps):
        self.maps = maps

    def __len__(self):
        return self.maps.shape[0]

    def __getitem__(self, idx):
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


def make_dataloaders(maps, batch_size=64, test_fraction=0.1, seed=42):
    """Split maps into train/test loaders."""
    dataset = TransportMapDataset(maps)
    N = len(dataset)
    n_test = int(N * test_fraction)
    n_train = N - n_test

    gen = torch.Generator().manual_seed(seed)
    train_set, test_set = random_split(dataset, [n_train, n_test], generator=gen)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print(f"Split: {n_train} train, {n_test} test")
    return train_loader, test_loader
