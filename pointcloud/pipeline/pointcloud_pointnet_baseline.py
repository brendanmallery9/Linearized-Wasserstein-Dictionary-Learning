"""
PointNet-style autoencoder baseline for the minimal point-cloud ablation.

This is the "raw-cloud, no-OT" control: unlike PCA / sparse coding / LWDL-EOT,
which all operate in the linearized OT (displacement) space, this method works
directly on the raw target point clouds mu_i and never uses a transport map or a
point correspondence.

Architecture (encoder-heavy, decoder-light -- deliberately mirroring LWDL's
own bias rather than handing the baseline a heavy generative decoder):

    Encoder:  canonical PointNet -- a shared per-point MLP followed by a
              symmetric max-pool over points and a small head, giving a
              permutation-invariant code  z in R^m.

    Decoder:  a *linear m-atom dictionary*
                  T_hat = bias + sum_j z_j A_j,   A_j in R^{n_out x 3}
              so the code width IS the number of atoms (one coefficient per
              atom), exactly like LWDL's linear decoder -- but the atoms A_j are
              free clouds with no Wasserstein / entropic-map constraint and no X
              anchor.  This makes it a same-m, capacity-matched dictionary whose
              only differences from LWDL are the free atoms and the (decoupled)
              PointNet encoder.

Two variants share this module via the `l1` knob:
    dense (l1 == 0):  no sparsity penalty; all m atoms free to activate.
    L1   (l1  > 0):   L1 penalty on the code -> a tunable per-sample L0, the
                      cloud-space analog of the sparse-coding row.

Training uses Chamfer distance (fast, differentiable); the ablation reports
Wasserstein at eval time via the shared helpers (Chamfer-train / EMD-eval).

Nothing here touches an existing result directory.
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Device
# ============================================================

def resolve_device(device_str):
    if device_str in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device_str == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


# ============================================================
# Model
# ============================================================

class PointNetEncoder(nn.Module):
    """Canonical PointNet encoder: shared per-point MLP + max-pool + head."""

    def __init__(self, code_dim, widths=(64, 128, 128, 256)):
        super().__init__()
        layers = []
        in_dim = 3
        for w in widths:
            layers += [nn.Linear(in_dim, w), nn.ReLU(inplace=True)]
            in_dim = w
        self.point_mlp = nn.Sequential(*layers)   # applied per point
        self.head = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, code_dim),
        )

    def forward(self, clouds):
        # clouds: (B, n, 3)
        feats = self.point_mlp(clouds)            # (B, n, W)
        pooled = feats.max(dim=1).values          # (B, W) symmetric pool
        return self.head(pooled)                  # (B, code_dim)


class LinearAtomDecoder(nn.Module):
    """Linear dictionary decoder: T_hat = bias + sum_j z_j A_j."""

    def __init__(self, m, n_out, atom_scale=0.1):
        super().__init__()
        self.atoms = nn.Parameter(torch.randn(m, n_out, 3) * atom_scale)
        self.bias = nn.Parameter(torch.zeros(n_out, 3))

    def forward(self, z):
        # z: (B, m) -> (B, n_out, 3)
        return torch.einsum("bm,mnd->bnd", z, self.atoms) + self.bias.unsqueeze(0)


class PointNetLinearAtomAE(nn.Module):
    """PointNet encoder + linear m-atom decoder."""

    def __init__(self, m, n_out, code_nonneg=False):
        super().__init__()
        self.encoder = PointNetEncoder(m)
        self.decoder = LinearAtomDecoder(m, n_out)
        self.code_nonneg = code_nonneg

    def encode(self, clouds):
        z = self.encoder(clouds)
        if self.code_nonneg:
            z = F.relu(z)
        return z

    def forward(self, clouds):
        z = self.encode(clouds)
        return self.decoder(z), z


# ============================================================
# Chamfer distance (batched, differentiable)
# ============================================================

def chamfer_distance_batch(a, b):
    """
    Mean symmetric Chamfer distance over a batch.

    Args:
        a: (B, n, 3)
        b: (B, m, 3)
    Returns:
        scalar mean over the batch of (mean_a min_b ||a-b|| + mean_b min_a ...).
    """
    D = torch.cdist(a, b)                 # (B, n, m) Euclidean
    forward = D.min(dim=2).values.mean(dim=1)
    backward = D.min(dim=1).values.mean(dim=1)
    return (forward + backward).mean()


# ============================================================
# Train / encode
# ============================================================

def _iterate_batches(N, batch_size, shuffle, generator=None):
    idx = (torch.randperm(N, generator=generator) if shuffle
           else torch.arange(N))
    for start in range(0, N, batch_size):
        yield idx[start:start + batch_size]


def run_pointnet_baseline(train_clouds, test_clouds, m, l1=0.0, epochs=150,
                          batch_size=64, lr=1e-3, seed=42, device="auto",
                          code_nonneg=False, n_out=None, verbose=True):
    """
    Fit the PointNet linear-atom AE on the train clouds and encode both splits.

    Args:
        train_clouds: (Ntr, n, 3) raw input clouds (mu_i on the train split).
        test_clouds:  (Nte, n, 3) raw input clouds (test split).
        m:            dictionary width == number of atoms == code dim.
        l1:           L1 penalty coefficient on the code (0.0 -> dense variant).
        epochs, batch_size, lr: training hyperparameters.
        seed:         torch seed for init + batching.
        device:       "auto" | "cuda" | "mps" | "cpu".
        code_nonneg:  if True, ReLU the code (nonnegative, LWDL-like).
        n_out:        decoder output point count (default: input n).

    Returns:
        dict with train/test codes, train/test reconstructed clouds (cpu tensors),
        runtime, chosen l1, and the learned atoms.  Reconstructions are point
        clouds (no correspondence), so map-space L2 is not defined for this row.
    """
    dev = resolve_device(device)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    train_clouds = torch.as_tensor(np.asarray(train_clouds), dtype=torch.float32)
    test_clouds = torch.as_tensor(np.asarray(test_clouds), dtype=torch.float32)
    n_in = train_clouds.shape[1]
    if n_out is None:
        n_out = n_in

    model = PointNetLinearAtomAE(m, n_out, code_nonneg=code_nonneg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(int(seed))

    Ntr = train_clouds.shape[0]
    t0 = time.time()
    model.train()
    for epoch in range(epochs):
        epoch_recon = 0.0
        epoch_steps = 0
        for batch_idx in _iterate_batches(Ntr, batch_size, shuffle=True,
                                          generator=gen):
            clouds = train_clouds[batch_idx].to(dev)
            recon, z = model(clouds)
            recon_loss = chamfer_distance_batch(recon, clouds)
            loss = recon_loss
            if l1 > 0:
                loss = loss + l1 * z.abs().sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_recon += float(recon_loss.item())
            epoch_steps += 1
        if verbose and (epoch == 0 or (epoch + 1) % 25 == 0
                        or epoch == epochs - 1):
            print(f"    [pointnet l1={l1:g}] epoch {epoch+1:4d}/{epochs}  "
                  f"chamfer={epoch_recon / max(epoch_steps,1):.6f}", flush=True)

    # Encode / reconstruct both splits
    model.eval()

    def _encode_recon(clouds):
        codes, recons = [], []
        with torch.no_grad():
            for start in range(0, clouds.shape[0], batch_size):
                b = clouds[start:start + batch_size].to(dev)
                r, z = model(b)
                codes.append(z.cpu())
                recons.append(r.cpu())
        return torch.cat(codes, 0), torch.cat(recons, 0)

    train_codes, train_recon = _encode_recon(train_clouds)
    test_codes, test_recon = _encode_recon(test_clouds)
    runtime = time.time() - t0

    return {
        "train_codes": train_codes,
        "test_codes": test_codes,
        "train_recon_clouds": train_recon,
        "test_recon_clouds": test_recon,
        "atoms": model.decoder.atoms.detach().cpu(),
        "l1": float(l1),
        "n_out": int(n_out),
        "runtime_seconds": runtime,
        "n_components": int(m),
    }
