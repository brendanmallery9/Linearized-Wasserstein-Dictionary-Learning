"""
Point-cloud-specific centered displacement SAE.

This keeps the point-cloud runner self-contained without depending on MNIST WIP
model variants.  The model subtracts a fixed train-split mean displacement field
before encoding and adds it back during decoding.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
MNIST_PIPELINE = REPO_ROOT / "mnist" / "pipeline"
for p in (str(REPO_ROOT), str(MNIST_PIPELINE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from mnist_sae_models import (  # noqa: E402
    LISTAEncoder,
    _build_atoms_module,
    normalize_atoms_l2rho,
)


class CenteredDisplacementFieldSAE(nn.Module):
    """
    Displacement SAE with a fixed mean displacement baseline.

    Let V = T - X and mu = E_train[V].  The model encodes residual fields
    V - mu and reconstructs T_hat = X + mu + sum_j lambda_j (T_j - X - mu).
    """

    def __init__(self, X, displacement_center, m, eps, grid_side=32,
                 lista_steps=1, activation_type="relu", normalize_atoms=False,
                 grid_points=None, atoms_type="gibbs", n_sinkhorn=30,
                 topk_k=3, per_atom_gain=False, lateral_init="zeros"):
        super().__init__()
        self.atoms_module = _build_atoms_module(
            atoms_type, X, m, eps, grid_side, grid_points, n_sinkhorn,
        )
        self.encoder = LISTAEncoder(
            m, lista_steps=lista_steps, activation_type=activation_type,
            topk_k=topk_k, per_atom_gain=per_atom_gain,
            lateral_init=lateral_init,
        )
        self.n = X.shape[0]
        self.m = m
        self.normalize_atoms = normalize_atoms
        self.register_buffer("X", X)
        self.register_buffer("displacement_center", displacement_center)

    def encode(self, V_centered, V_atoms_enc):
        inner = torch.einsum("bnd, mnd -> bm", V_centered, V_atoms_enc) / self.n
        return self.encoder(inner)

    def decode(self, lam, V_atoms_centered):
        V_hat_centered = torch.einsum("bm, mnd -> bnd", lam, V_atoms_centered)
        return (
            self.X.unsqueeze(0)
            + self.displacement_center.unsqueeze(0)
            + V_hat_centered
        )

    def forward(self, T):
        atoms = self.atoms_module()
        center = self.displacement_center.unsqueeze(0)
        V_atoms = atoms - self.X.unsqueeze(0) - center
        if self.normalize_atoms:
            V_atoms_enc = normalize_atoms_l2rho(V_atoms, self.n)
        else:
            V_atoms_enc = V_atoms
        V = T - self.X.unsqueeze(0) - center
        lam = self.encode(V, V_atoms_enc)
        T_hat = self.decode(lam, V_atoms)
        return T_hat, lam
