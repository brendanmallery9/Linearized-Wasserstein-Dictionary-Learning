"""
Dataset utilities for transport map sparse autoencoders.

Loads data produced by prepare_mnist_ot.py, which saves:
    <output_dir>/base_measure.pt          -- shape (n, 2)
    <output_dir>/digit_<d>/mappings.pt    -- shape (N_d, n, 2) for each digit d

This module concatenates all digits into a single dataset of maps,
splits into train/test, and provides a simple PyTorch Dataset.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path


class TransportMapDataset(Dataset):
    """
    Dataset of transport maps T_{rho->mu} of shape (n, 2).

    Args:
        maps: tensor of shape (N, n, 2)
    """
    def __init__(self, maps):
        self.maps = maps  # (N, n, 2)

    def __len__(self):
        return self.maps.shape[0]

    def __getitem__(self, idx):
        return self.maps[idx]  # (n, 2)


def load_transport_maps(data_dir, digits=None):
    """
    Load base measure and transport maps from prepare_mnist_ot.py output.

    Args:
        data_dir: path to directory containing base_measure.pt and digit_* folders
        digits: list of digit labels to load (default: all available)

    Returns:
        X: tensor of shape (n, 2), the base measure support points
        maps: tensor of shape (N, n, 2), all transport maps concatenated
    """
    data_dir = Path(data_dir)
    X = torch.load(data_dir / "base_measure.pt", map_location="cpu")  # (n, 2)

    if digits is None:
        # Load all digit folders that exist
        digits = sorted(
            int(p.name.split("_")[1])
            for p in data_dir.iterdir()
            if p.is_dir() and p.name.startswith("digit_")
        )

    all_maps = []
    for d in digits:
        path = data_dir / f"digit_{d}" / "mappings.pt"
        if path.exists():
            m = torch.load(path, map_location="cpu")  # (N_d, n, 2)
            all_maps.append(m)
            print(f"  Loaded digit {d}: {m.shape[0]} maps")
        else:
            print(f"  Warning: {path} not found, skipping digit {d}")

    maps = torch.cat(all_maps, dim=0)  # (N, n, 2)
    print(f"Total: {maps.shape[0]} maps, support size n={maps.shape[1]}")
    return X, maps


def make_dataloaders(maps, batch_size=64, test_fraction=0.1, seed=42):
    """
    Split maps into train/test and return DataLoaders.

    Args:
        maps: tensor of shape (N, n, 2)
        batch_size: batch size for both loaders
        test_fraction: fraction of data for test set
        seed: random seed for reproducible split

    Returns:
        train_loader, test_loader
    """
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
