import argparse
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

from torch.optim.lr_scheduler import CosineAnnealingLR


class _NullMetricWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def flush(self):
        return None

    def close(self):
        return None


class JumpReLUAE(nn.Module):
    def __init__(self, m, hidden_dim):
        super().__init__()
        #m: input dimension
        self.enc = nn.Linear(m, hidden_dim, bias=True)
        self.threshold = nn.Parameter(torch.zeros(hidden_dim))
        
    def forward(self, x):
        pre_act = self.enc(x)
        z = F.relu(pre_act) * (pre_act > self.threshold).float()
        xhat = F.linear(z, self.enc.weight.t())
        return xhat, z


class ReLUAE(nn.Module):
    def __init__(self, m, hidden_dim):
        super().__init__()
        #m: input dimension
        #k: hidden dimension
        self.enc = nn.Linear(m, hidden_dim, bias=True)
        nn.init.kaiming_uniform_(self.enc.weight, nonlinearity="relu")
        nn.init.zeros_(self.enc.bias)

    def forward(self, x):
        z = F.relu(self.enc(x))                  # (B, k)
        xhat = F.linear(z, self.enc.weight.t())  # tied decoder
        return xhat, z


class GatedSAE(nn.Module):
    def __init__(self, m, hidden_dim):
        super().__init__()
        self.W_gate = nn.Linear(m, hidden_dim, bias=True)
        self.W_mag = nn.Linear(m, hidden_dim, bias=False)
        
        self.W_dec = nn.Parameter(torch.empty(m, hidden_dim))
        nn.init.kaiming_uniform_(self.W_dec, nonlinearity="linear")
        
        # Store dims for config extraction
        self.m = m
        self.hidden_dim= hidden_dim
        self.enc=self.W_mag
    def forward(self, x):
        gate = (self.W_gate(x) > 0).float()
        magnitude = F.relu(self.W_mag(x))
        z = gate * magnitude
        xhat = F.linear(z, self.W_dec)
        return xhat, z
    

def topk_activation(z, k):
    # z: (B, hidden_dim)
    k = min(k, z.size(-1))
    topk_vals, topk_idx = torch.topk(z, k, dim=-1)
    z_sparse = torch.zeros_like(z)
    z_sparse.scatter_(-1, topk_idx, topk_vals)
    return z_sparse

class TopKAE(nn.Module):
    def __init__(self, m: int, hidden_dim: int, top_k: int):
        super().__init__()
        self.m = m
        self.hidden_dim = hidden_dim
        self.top_k = top_k

        self.enc = nn.Linear(m, hidden_dim, bias=True)
        nn.init.kaiming_uniform_(self.enc.weight, nonlinearity="relu")
        nn.init.zeros_(self.enc.bias)

    def forward(self, x):
        # Encoder pre-activations
        pre = self.enc(x)                # (B, hidden_dim)

        # Nonnegativity + top-k sparsification
        z = F.relu(pre)                  # (B, hidden_dim)
        z = topk_activation(z, self.top_k)

        # Tied decoder (W_dec = W_enc^T)
        xhat = F.linear(z, self.enc.weight.t())  # (B, m)
        return xhat, z


class LISTAAE(nn.Module):
    """
    Tied-decoder sparse autoencoder with a multi-step LISTA encoder.

    With lista_steps=1 this is a one-step ReLU/JumpReLU encoder. Additional
    steps reuse the same input drive and add learned lateral interactions from
    the previous sparse code.
    """
    def __init__(
        self,
        m: int,
        hidden_dim: int,
        lista_steps: int = 1,
        activation_type: str = "relu",
        per_atom_gain: bool = False,
        lateral_init: str = "zeros",
    ):
        super().__init__()
        if lista_steps < 1:
            raise ValueError("lista_steps must be >= 1")
        if activation_type not in {"relu", "jumprelu"}:
            raise ValueError("activation_type must be 'relu' or 'jumprelu'")
        if lateral_init not in {"zeros", "damped_identity"}:
            raise ValueError("lateral_init must be 'zeros' or 'damped_identity'")

        self.m = m
        self.hidden_dim = hidden_dim
        self.lista_steps = lista_steps
        self.activation_type = activation_type
        self.per_atom_gain = per_atom_gain

        self.enc = nn.Linear(m, hidden_dim, bias=False)
        nn.init.kaiming_uniform_(self.enc.weight, nonlinearity="relu")

        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden_dim)) for _ in range(lista_steps)]
        )

        laterals = []
        for _ in range(lista_steps - 1):
            if lateral_init == "damped_identity":
                S = 0.5 * torch.eye(hidden_dim) + 0.01 * torch.randn(hidden_dim, hidden_dim)
            else:
                S = torch.zeros(hidden_dim, hidden_dim)
            laterals.append(nn.Parameter(S))
        self.laterals = nn.ParameterList(laterals)

        if per_atom_gain:
            self.gain = nn.Parameter(torch.ones(hidden_dim))
        else:
            self.register_parameter("gain", None)

        if activation_type == "jumprelu":
            self.thresholds = nn.ParameterList(
                [nn.Parameter(torch.zeros(hidden_dim)) for _ in range(lista_steps)]
            )
        else:
            self.thresholds = None

    def _activate(self, pre_act, step: int):
        if self.activation_type == "relu":
            return F.relu(pre_act)

        thresh = self.thresholds[step]
        gate = (pre_act > thresh).float()
        # Straight-through gate: hard threshold in the forward pass, identity
        # gradient through the gate value.
        gate_st = pre_act - (pre_act - gate).detach()
        return F.relu(pre_act) * gate_st

    def forward(self, x):
        inner = self.enc(x)
        if self.gain is not None:
            inner = inner * self.gain

        z = self._activate(inner + self.biases[0], step=0)
        for t in range(self.lista_steps - 1):
            pre_act = inner + z @ self.laterals[t].t() + self.biases[t + 1]
            z = self._activate(pre_act, step=t + 1)

        xhat = F.linear(z, self.enc.weight.t())
        return xhat, z

class ReLUAE_monotone(nn.Module):
    """
    Monotone atoms + linear synthesis, with scale stabilization.
    (No code mass normalization: z is only constrained to be >= 0.)
    """
    def __init__(self, m: int, hidden_dim: int, eps: float = 1e-8):
        super().__init__()
        self.m = m
        self.k = hidden_dim
        self.eps = eps

        self.enc = nn.Linear(m, hidden_dim, bias=True)
        nn.init.kaiming_uniform_(self.enc.weight, nonlinearity="relu")
        nn.init.zeros_(self.enc.bias)
        #testing scaling factor for codes
        self.code_scale = nn.Parameter(torch.tensor(0.1))

        # Decoder raw params (k, m)
        self.theta = nn.Parameter(torch.empty(hidden_dim, m))
        nn.init.normal_(self.theta, mean=-2.0, std=0.02)

    def atoms(self):
        D = F.softplus(self.theta) / self.m            # (k, m) increments, >=0, scaled
        A = torch.cumsum(D, dim=-1)                    # (k, m) monotone
        A = A / A[:, -1:].clamp_min(self.eps)          # (k, m) normalize to end at 1
        return A

    def forward(self, x):
        z = F.relu(self.enc(x))                        # (B, k) >= 0 (NOT normalized)
        z = self.code_scale * z
        A = self.atoms()                               # (k, m)
        xhat = z @ A                                   # (B, m)
        #Enforces the xhat[0]=0
        xhat = xhat - xhat[..., :1]
        return xhat, z


class TopKAE_monotone(nn.Module):
    """
    Top-K sparse nonnegative codes + monotone atoms + linear synthesis.
    (No code mass normalization.)
    """
    def __init__(self, m: int, hidden_dim: int, top_k: int, eps: float = 1e-8):
        super().__init__()
        self.m = m
        self.k = hidden_dim
        self.top_k = top_k
        self.eps = eps

        self.enc = nn.Linear(m, hidden_dim, bias=True)
        nn.init.kaiming_uniform_(self.enc.weight, nonlinearity="relu")
        nn.init.zeros_(self.enc.bias)
        #testing scaling factor for codes
        self.code_scale = nn.Parameter(torch.tensor(0.1))

        self.theta = nn.Parameter(torch.empty(hidden_dim, m))
        nn.init.normal_(self.theta, mean=-2.0, std=0.02)

    def atoms(self):
        D = F.softplus(self.theta) / self.m
        A = torch.cumsum(D, dim=-1)
        A = A / A[:, -1:].clamp_min(self.eps)
        return A

    def forward(self, x):
        pre = self.enc(x)                              # (B, k)
        z = F.relu(pre)                                # (B, k) >= 0
        z = topk_activation(z, self.top_k)             # (B, k) sparse, >=0
        z = self.code_scale * z

        A = self.atoms()                               # (k, m)
        xhat = z @ A
        #Enforces the xhat[0]=0
        xhat = xhat - xhat[..., :1]

        return xhat, z
    
class JumpReLUAE_nonneg(nn.Module):
    """
    JumpReLU nonnegative codes + nonnegative atoms (no monotonicity) + linear synthesis.
    """
    def __init__(self, m: int, hidden_dim: int, eps: float = 1e-8):
        super().__init__()
        self.m = m
        self.k = hidden_dim
        self.eps = eps

        self.enc = nn.Linear(m, hidden_dim, bias=True)
        nn.init.kaiming_uniform_(self.enc.weight, nonlinearity="relu")
        nn.init.zeros_(self.enc.bias)

        self.threshold = nn.Parameter(torch.zeros(hidden_dim))
        self.code_scale = nn.Parameter(torch.tensor(0.1))

        self.theta = nn.Parameter(torch.empty(hidden_dim, m))
        nn.init.normal_(self.theta, mean=-2.0, std=0.02)

    def atoms(self):
        # Nonneg via softplus, but no cumsum -> no monotonicity
        A = F.softplus(self.theta)
        return A

    def forward(self, x):
        pre = self.enc(x)
        z = F.relu(pre) * (pre > self.threshold).float()
        z = self.code_scale * z
        A = self.atoms()
        xhat = z @ A
        xhat = xhat - xhat[..., :1]
        return xhat, z


class JumpReLUAE_monotone(nn.Module):
    """
    JumpReLU nonnegative codes + monotone atoms + linear synthesis.
    (No code mass normalization.)
    """
    def __init__(self, m: int, hidden_dim: int, eps: float = 1e-8):
        super().__init__()
        self.m = m
        self.k = hidden_dim
        self.eps = eps

        self.enc = nn.Linear(m, hidden_dim, bias=True)
        nn.init.kaiming_uniform_(self.enc.weight, nonlinearity="relu")
        nn.init.zeros_(self.enc.bias)

        self.threshold = nn.Parameter(torch.zeros(hidden_dim))

        #testing scaling factor for codes
        self.code_scale = nn.Parameter(torch.tensor(0.1))

        self.theta = nn.Parameter(torch.empty(hidden_dim, m))
        nn.init.normal_(self.theta, mean=-2.0, std=0.02)

    def atoms(self):
        D = F.softplus(self.theta) / self.m
        A = torch.cumsum(D, dim=-1)
        A = A / A[:, -1:].clamp_min(self.eps)
        return A

    def forward(self, x):
        pre = self.enc(x)                              # (B, k)
        z = F.relu(pre) * (pre > self.threshold).float()  # (B, k) >= 0
        z = self.code_scale * z
        A = self.atoms()                               # (k, m)
        xhat = z @ A
        #Enforces the xhat[0]=0
        xhat = xhat - xhat[..., :1]
        return xhat, z


class TensorDirDataset(IterableDataset):
    """
    Streams rows from all .pt files in a directory.
    Each file must contain a tensor of shape [L, D].
    """
    def __init__(self, shard_dir: Path, shuffle_shards=True, shuffle_samples=True, seed=None):
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.shuffle_shards = shuffle_shards
        self.shuffle_samples = shuffle_samples
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        """Call this at the start of each epoch for different shuffling"""
        self.epoch = epoch

    def __iter__(self):
        files = sorted(self.shard_dir.glob("*.pt"))
        if len(files) == 0:
            raise ValueError(f"No .pt files found in {self.shard_dir}")

        # Create generator with epoch-dependent seed
        if self.seed is not None:
            seed = self.seed + self.epoch
        else:
            seed = torch.initial_seed() + self.epoch
        
        g = torch.Generator()
        g.manual_seed(seed)

        # Shuffle shards
        if self.shuffle_shards:
            perm = torch.randperm(len(files), generator=g).tolist()
            files = [files[i] for i in perm]

        # Support multi-worker DataLoader: split files across workers
        worker = torch.utils.data.get_worker_info()
        if worker is not None:
            # Each worker gets different seed
            worker_seed = seed + worker.id
            g.manual_seed(worker_seed)
            files = files[worker.id::worker.num_workers]

        for f in files:
            shard = torch.load(f, map_location="cpu")  # [L, D]
            if shard.ndim != 2:
                raise ValueError(f"{f} expected 2D [L, D], got {tuple(shard.shape)}")

            # Shuffle samples within shard
            if self.shuffle_samples:
                indices = torch.randperm(shard.shape[0], generator=g)
                shard = shard[indices]

            for i in range(shard.shape[0]):
                yield shard[i]  # [D]

def compute_mu_sigma_from_dir(data_dir, eps=1e-6):
    data_dir = Path(data_dir)
    paths = sorted(data_dir.glob("*.pt"))
    if not paths:
        raise ValueError(f"No .pt files found in {data_dir}")

    total_sum = None
    total_sq_sum = None
    total_count = 0

    for p in paths:
        X = torch.load(p, map_location="cpu").float()  # [Ni, D]

        s1 = X.sum(dim=0)
        s2 = (X * X).sum(dim=0)

        if total_sum is None:
            total_sum = s1
            total_sq_sum = s2
        else:
            total_sum += s1
            total_sq_sum += s2

        total_count += X.shape[0]

    mu = total_sum / total_count
    var = total_sq_sum / total_count - mu * mu
    var = var.clamp_min(0.0)
    sigma = torch.sqrt(var).clamp_min(eps)

    return mu.unsqueeze(0), sigma.unsqueeze(0)

def train_sparse_ae(
    data_path,
    hidden_dim,
    architecture,
    input_dim,
    batch_size=256,
    epochs=50,
    lr=1e-7,
    l1=1e-3,
    top_k=4,
    step_print_every=200,
    epoch_summary_every=1,
    device="cuda",
    log_dir="runs/sparse_ae",
    log_every=50,
    grad_clip=-1,
    normalize=False,
    seed=42,
    validation_path=None,  # New argument: Path or None
    val_frequency=0.2,     # Validate every 20% of epochs by default
    early_stop=False,      # Enable early stopping
    patience=200,          # Epochs to wait for improvement before stopping
    min_delta=1e-5,        # Minimum relative improvement to count as progress
    lista_steps=1,
    lista_activation="relu",
    per_atom_gain=False,
    lateral_init="zeros"):
    
    data_path = Path(data_path)
    mu, sigma = compute_mu_sigma_from_dir(data_path, eps=1e-6)
    mu = mu.to(device)
    sigma = sigma.to(device)

    dataset = TensorDirDataset(data_path, shuffle_shards=True, shuffle_samples=True, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
    )
    first_batch = next(iter(loader))          # [B, D]
    print(f"Loaded shards from: {data_path}")
    print(f"Example batch: {tuple(first_batch.shape)}")

    # Setup validation data if provided
    val_loader = None
    if validation_path is not None:
        validation_path = Path(validation_path)
        val_dataset = TensorDirDataset(validation_path, shuffle_shards=False, shuffle_samples=False, seed=seed)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
            persistent_workers=False,
        )
        print(f"Loaded validation shards from: {validation_path}")
    
    # Calculate validation epochs
    val_interval = max(1, int(epochs * val_frequency))
    val_epochs = set(range(val_interval - 1, epochs, val_interval))
    val_epochs.add(epochs - 1)  # Always validate on last epoch
    print(f"Will validate at epochs: {sorted(val_epochs)}")

    print(
        f"Run dir: {data_path}\n"
        f"batch_size={batch_size}, "
        f"epochs={epochs}, "
        f"lr={lr}, "
        f"l1={l1}, "
        f"device={device}, "
        f"normalize={normalize}, "
        f"lista_steps={lista_steps}, "
        f"lista_activation={lista_activation}, "
        f"per_atom_gain={per_atom_gain}, "
        f"lateral_init={lateral_init}"
    )
    if architecture == 'ReLUAE':
        model = ReLUAE(input_dim, hidden_dim).to(device)
    elif architecture == 'JumpReLU':
        model = JumpReLUAE(input_dim, hidden_dim).to(device)
    elif architecture == 'GatedSAE':
        model = GatedSAE(input_dim, hidden_dim).to(device)
    elif architecture == 'TopKAE':
        model=TopKAE(input_dim,hidden_dim,top_k=top_k).to(device)
    elif architecture == 'LISTAAE':
        model = LISTAAE(
            input_dim,
            hidden_dim,
            lista_steps=lista_steps,
            activation_type=lista_activation,
            per_atom_gain=per_atom_gain,
            lateral_init=lateral_init,
        ).to(device)
    elif architecture == 'ReLUAE_monotone':
        model = ReLUAE_monotone(input_dim, hidden_dim).to(device)
    elif architecture == 'JumpReLU_monotone':
        model = JumpReLUAE_monotone(input_dim, hidden_dim).to(device)
    elif architecture == 'JumpReLU_nonneg':
        model=JumpReLUAE_nonneg(input_dim,hidden_dim).to(device)
  #  elif architecture == 'GatedSAE_monotone':
  #      model = GatedSAE_monotone(input_dim, hidden_dim).to(device)
    elif architecture == 'TopKAE_monotone':
        model=TopKAE_monotone(input_dim,hidden_dim,top_k=top_k).to(device)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr/1000)

    writer = _NullMetricWriter()
    global_step = 0
    
    # Track best validation loss and checkpoints
    best_val_loss = float('inf')
    val_results = []  # Store (epoch, val_loss, checkpoint_path)

    # Early stopping state
    best_train_loss = float('inf')
    epochs_without_improvement = 0
    stopped_early = False

    try:
        for ep in range(epochs):
            dataset.set_epoch(ep)
            steps_in_epoch = 0

            epoch_total = 0.0
            epoch_recon = 0.0
            epoch_l1 = 0.0

            for step_in_ep, xb in enumerate(loader, start=1):
                steps_in_epoch += 1

                xb = xb.to(device, non_blocking=True).float()

                # global normalization
                if normalize:
                    #Don't normalize for monotone data
                    with torch.no_grad():
                        xb = (xb - mu) / sigma

                xhat, z = model(xb)
                recon = F.mse_loss(xhat, xb, reduction="mean")
                spars = z.abs().mean()
                loss = recon + l1 * spars

                opt.zero_grad(set_to_none=True)
                loss.backward()

                if grad_clip is not None and grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                else:
                    grad_norm = torch.tensor(float("nan"), device=device)

                opt.step()

                # accumulate epoch stats
                epoch_total += loss.item()
                epoch_recon += recon.item()
                epoch_l1 += spars.item()

                # Metric hook retained as a no-op.
                if (global_step % log_every) == 0:
                    writer.add_scalar("loss/total", loss.item(), global_step)
                    writer.add_scalar("loss/recon", recon.item(), global_step)
                    writer.add_scalar("loss/sparsity_mean_abs_z", spars.item(), global_step)
                    writer.add_scalar("opt/grad_norm", float(grad_norm), global_step)
                if step_print_every > 0 and (step_in_ep == 1 or step_in_ep % step_print_every == 0):
                    print(
                        f"[epoch {ep+1:3d}/{epochs}] "
                        f"step {step_in_ep:4d} | "
                        f"loss {loss.item():.6f} (recon {recon.item():.6f} + {l1:g}*{spars.item():.6f}) | "
                        f"grad {float(grad_norm):.3f}"
                    )
                global_step += 1

            avg_total = epoch_total / steps_in_epoch
            avg_recon = epoch_recon / steps_in_epoch
            avg_l1    = epoch_l1    / steps_in_epoch

            writer.add_scalar("epoch/loss_total", avg_total, ep + 1)
            writer.add_scalar("epoch/loss_recon", avg_recon, ep + 1)
            writer.add_scalar("epoch/sparsity_mean_abs_z", avg_l1, ep + 1)

            should_print_epoch_summary = (
                epoch_summary_every > 0
                and (((ep + 1) % epoch_summary_every) == 0 or ep == 0 or ep == epochs - 1)
            )
            if should_print_epoch_summary:
                print(
                    f"[epoch {ep+1:3d}/{epochs}] DONE | avg loss {avg_total:.6f} | "
                    f"avg recon {avg_recon:.6f} | avg |z| {avg_l1:.6f}"
                )

            # Early stopping check
            if early_stop:
                rel_improvement = (best_train_loss - avg_total) / (abs(best_train_loss) + 1e-12)
                if avg_total < best_train_loss and rel_improvement > min_delta:
                    best_train_loss = avg_total
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {ep+1}: no improvement > {min_delta} "
                          f"for {patience} epochs (best loss: {best_train_loss:.6f})")
                    stopped_early = True

            # Validation
            if val_loader is not None and ep in val_epochs:
                model.eval()
                val_total = 0.0
                val_recon = 0.0
                val_l1 = 0.0
                val_steps = 0
                
                with torch.no_grad():
                    for val_xb in val_loader:
                        val_xb = val_xb.to(device, non_blocking=True).float()
                        
                        if normalize:
                            val_xb = (val_xb - mu) / sigma
                        
                        val_xhat, val_z = model(val_xb)
                        val_recon_loss = F.mse_loss(val_xhat, val_xb, reduction="mean")
                        val_spars = val_z.abs().mean()
                        val_loss = val_recon_loss + l1 * val_spars
                        
                        val_total += val_loss.item()
                        val_recon += val_recon_loss.item()
                        val_l1 += val_spars.item()
                        val_steps += 1
                
                avg_val_total = val_total / val_steps
                avg_val_recon = val_recon / val_steps
                avg_val_l1 = val_l1 / val_steps
                
                # Log validation metrics
                writer.add_scalar("val/loss_total", avg_val_total, ep + 1)
                writer.add_scalar("val/loss_recon", avg_val_recon, ep + 1)
                writer.add_scalar("val/sparsity_mean_abs_z", avg_val_l1, ep + 1)
                
                print(f"[epoch {ep+1:3d}/{epochs}] VAL  | avg loss {avg_val_total:.6f} | avg recon {avg_val_recon:.6f} | avg |z| {avg_val_l1:.6f}")
                
                # Save checkpoint for this validation epoch
                log_dir_path = Path(log_dir)
                ckpt_path = log_dir_path / f"sparse_ae_epoch{ep+1}.pt"
                torch.save(model.state_dict(), ckpt_path)
                val_results.append((ep + 1, avg_val_total, ckpt_path))
                print(f"Saved validation checkpoint to {ckpt_path}")
                
                # Track best model
                if avg_val_total < best_val_loss:
                    best_val_loss = avg_val_total
                    best_ckpt_path = log_dir_path / "sparse_ae_best.pt"
                    torch.save(model.state_dict(), best_ckpt_path)
                    print(f"New best model! Saved to {best_ckpt_path}")
                
                model.train()
            scheduler.step()
            if stopped_early:
                break  

    finally:
        writer.flush()
        writer.close()
    
    # Print validation summary
    if val_results:
        print("\n" + "="*60)
        print("Validation Summary:")
        print("="*60)
        for epoch, val_loss, ckpt_path in val_results:
            marker = " *BEST*" if val_loss == best_val_loss else ""
            print(f"  Epoch {epoch:3d}: val_loss={val_loss:.6f} -> {ckpt_path}{marker}")
        print("="*60 + "\n")

    return model

def main():
    parser = argparse.ArgumentParser(description="Sparse Autoencoder (tied weights)")

    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to a .pt tensor file (n, m) or (m, n)")
    parser.add_argument("--hidden_dim", type=int, required=True,
                        help="Hidden layer size (number of atoms)")
    parser.add_argument("--architecture", type=str, required=True)
    parser.add_argument("--input_dim", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l1", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--step_print_every", type=int, default=200,
                        help="Print per-step training metrics every N steps (set <=0 to disable)")
    parser.add_argument("--epoch_summary_every", type=int, default=1,
                        help="Print epoch summary metrics every N epochs")
    parser.add_argument("--lista_steps", type=int, default=1,
                        help="Number of LISTA encoder refinement steps for architecture=LISTAAE")
    parser.add_argument("--lista_activation", type=str, default="relu",
                        choices=["relu", "jumprelu"],
                        help="Sparsifying activation for architecture=LISTAAE")
    parser.add_argument("--per_atom_gain", action="store_true", default=False,
                        help="Enable learnable per-atom gain for architecture=LISTAAE")
    parser.add_argument("--lateral_init", type=str, default="zeros",
                        choices=["zeros", "damped_identity"],
                        help="Initialization for LISTAAE lateral matrices")
    parser.add_argument("--device", type=str, default="cuda")

    # logging
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Base directory for checkpoints")
    parser.add_argument("--log_every", type=int, default=10,
                        help="Log scalars every N optimization steps")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Clip grad norm (set <=0 to disable)")

    # ---- normalize flags (fixed) ----
    parser.add_argument(
        "--normalize",
        action="store_true",
        dest="normalize",
        help="Enable normalization (z-score each input vector)"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_false",
        dest="normalize",
        help="Disable normalization (default)"
    )
    parser.set_defaults(normalize=False)

    # Validation arguments
    parser.add_argument("--validation_path", type=str, default=None,
                        help="Path to validation data directory (optional)")
    parser.add_argument("--val_frequency", type=float, default=0.2,
                        help="Fraction of epochs between validation runs (default: 0.2 = every 20%% of epochs)")

    # Early stopping arguments
    parser.add_argument("--early_stop", action="store_true", default=False,
                        help="Enable early stopping based on training loss plateau")
    parser.add_argument("--patience", type=int, default=200,
                        help="Epochs to wait for improvement before stopping (default: 200)")
    parser.add_argument("--min_delta", type=float, default=1e-5,
                        help="Minimum relative improvement to count as progress (default: 1e-5)")

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            print("CUDA not available, falling back to MPS")
            device = "mps"
        else:
            print("CUDA not available, falling back to CPU")
            device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"

    # ---- unique run directory ----
    base_dir = Path(args.log_dir)
    run_dir = base_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_path)

    def _train(dev):
        return train_sparse_ae(
            data_path=data_path,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            architecture=args.architecture,
            input_dim=args.input_dim,
            epochs=args.epochs,
            lr=args.lr,
            l1=args.l1,
            top_k=args.top_k,
            step_print_every=args.step_print_every,
            epoch_summary_every=args.epoch_summary_every,
            device=dev,
            log_dir=str(run_dir),
            log_every=args.log_every,
            grad_clip=args.grad_clip if args.grad_clip and args.grad_clip > 0 else None,
            normalize=args.normalize,
            seed=args.seed,
            validation_path=args.validation_path,
            val_frequency=args.val_frequency,
            early_stop=args.early_stop,
            patience=args.patience,
            min_delta=args.min_delta,
            lista_steps=args.lista_steps,
            lista_activation=args.lista_activation,
            per_atom_gain=args.per_atom_gain,
            lateral_init=args.lateral_init,
        )

    try:
        model = _train(device)
    except RuntimeError as e:
        if device == "mps":
            print(f"\nMPS training failed ({type(e).__name__}: {e})")
            print("Retrying on CPU...\n")
            device = "cpu"
            model = _train(device)
        else:
            raise

    # save model into the same run folder
    ckpt_path = run_dir / "sparse_ae.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved model to {ckpt_path}")

    # save args for reproducibility
    args_path = run_dir / "args.txt"
    with open(args_path, "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
    print(f"Saved args to {args_path}")

    print(f"CKPT_PATH={ckpt_path}")

if __name__ == "__main__":
    main()
