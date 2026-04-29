"""
Sparse autoencoder model for transport maps on [0,1]^2.

DisplacementFieldSAE works with displacement fields V = T - Id, with
dictionary atoms parameterized via Gibbs/softmax over a fixed 2D grid Y.

Dictionary atom parameterization
--------------------------------
Each atom j maps source points X = {x_l} to [0,1]^2 via:

    T_j(x_l) = sum_k  pi_{lk}^{(j)}  y_k

where pi is a softmax:

    pi_{lk}^{(j)} = softmax_k( (1/eps) * (-0.5 ||x_l - y_k||^2 + h_{j,k}) )

The trainable parameters are H_raw of shape (m, K), centered to H = H_raw - row_mean(H_raw)
so that sum_k h_{j,k} = 0 for each atom j.  <-- CENTERING CONSTRAINT enforced here.

Epsilon enters the atom construction through the 1/eps scaling of the softmax logits.
Smaller epsilon -> sharper (more deterministic) transport plans.

Encoding
--------
    [lambda(mu, theta)]_j = chi( <T_j, T_{rho->mu}>_{L^2(rho)} + b_j )

where chi = ReLU by default, and <.,.>_{L^2(rho)} is the discrete weighted inner product
over the support points X with uniform weights w_l = 1/n.

Reconstruction
--------------
    T_hat = sum_j lambda_j * T_j        (raw-map version)
    V_hat = sum_j lambda_j * V_j,  T_hat = Id + V_hat   (displacement version)

Nothing from SAE.py is directly reused as a base class, but the general style
(explicit forward pass, simple parameter layout, no deep class hierarchies) follows
that file's conventions.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_grid_2d(grid_side=32):
    """
    Create a uniform 2D grid on [0,1]^2.

    Returns:
        Y: tensor of shape (K, 2) where K = grid_side^2
    """
    t = torch.linspace(0, 1, grid_side)
    yy, xx = torch.meshgrid(t, t, indexing="ij")
    Y = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)  # (K, 2)
    return Y


class GibbsAtoms(nn.Module):
    """
    Parameterizes m dictionary atoms via Gibbs/softmax over a fixed grid.

    Parameters:
        X: (n, 2) base measure support points (registered as buffer)
        Y: (K, 2) target grid points (registered as buffer)
        H_raw: (m, K) unconstrained parameters

    The centering constraint sum_k h_{j,k} = 0 is enforced by subtracting
    the row mean in the forward pass: H = H_raw - mean(H_raw, dim=1, keepdim=True).

    Epsilon (eps) controls the sharpness of the softmax and enters as 1/eps
    scaling of the logits.
    """
    def __init__(self, X, m, eps, grid_side=32, grid_points=None):
        super().__init__()
        if grid_points is not None:
            Y = grid_points if isinstance(grid_points, torch.Tensor) else torch.tensor(grid_points, dtype=torch.float32)
        else:
            Y = make_grid_2d(grid_side)
        K = Y.shape[0]

        self.register_buffer("X", X)          # (n, 2)
        self.register_buffer("Y", Y)          # (K, 2)
        self.m = m
        self.eps = eps

        # Unconstrained parameters; centering applied in forward pass
        self.H_raw = nn.Parameter(torch.randn(m, K) * 0.01)

        # Precompute squared distances ||x_l - y_k||^2, shape (n, K)
        # This doesn't change, so register as buffer
        dist_sq = torch.cdist(X.unsqueeze(0), Y.unsqueeze(0)).squeeze(0) ** 2  # (n, K)
        self.register_buffer("cost", -0.5 * dist_sq)  # (n, K), the -0.5||x-y||^2 term

    def set_eps(self, eps):
        """Update the softmax temperature (used by epsilon annealing)."""
        self.eps = float(eps)

    def forward(self):
        """
        Compute all m atoms evaluated at the n source points.

        Returns:
            atoms: (m, n, 2) -- T_j(x_l) for each atom j and source point x_l
        """
        # Centering constraint: H = H_raw - row_mean(H_raw)
        H = self.H_raw - self.H_raw.mean(dim=1, keepdim=True)  # (m, K)

        # Logits: (1/eps) * (-0.5||x_l - y_k||^2 + h_{j,k})
        # cost is (n, K), H is (m, K)
        # logits[j, l, k] = (1/eps) * (cost[l, k] + H[j, k])
        logits = (self.cost.unsqueeze(0) + H.unsqueeze(1)) / self.eps  # (m, n, K)

        # Stable softmax over k (last dim)
        pi = F.softmax(logits, dim=2)  # (m, n, K)

        # Atoms: T_j(x_l) = sum_k pi_{lk}^{(j)} * y_k
        atoms = torch.einsum("mnk, kd -> mnd", pi, self.Y)  # (m, n, 2)
        return atoms


class SinkhornAtoms(nn.Module):
    """
    Dictionary atoms parameterized as entropic OT maps from rho (uniform on X)
    to a learned target measure nu_j on the grid Y.

    For atom j, the learnable parameter is psi_j (K,), and the target measure is
        b_j = softmax(psi_j)   (probability vector on Y)

    The entropic OT plan between rho = (1/n) * sum_l delta_{x_l} and nu_j is
        pi^{(j)}_{lk} = a_l * b_{j,k} * exp((f_{j,l} + g_{j,k} - C_{lk}) / eps)

    where C_{lk} = (1/2) ||x_l - y_k||^2 and (f_j, g_j) are dual potentials
    that enforce the marginals.  The potentials are NOT learned; they are
    computed by log-domain Sinkhorn iterations (warm-started across forward
    passes).

    The atom is the barycentric projection:
        T_j(x_l) = (1 / a_l) * sum_k pi^{(j)}_{lk} * y_k

    Differentiation: we run (n_sinkhorn - 1) iterations under no_grad (with
    log_b detached) and then one more Sinkhorn step with grad enabled.  This
    treats (f, g) as the converged response to the current log_b and gives a
    much lighter computation graph than unrolling all iterations.  Close in
    spirit to implicit differentiation at the fixed point.
    """
    def __init__(self, X, m, eps, grid_side=32, grid_points=None, n_sinkhorn=30):
        super().__init__()
        if grid_points is not None:
            Y = grid_points if isinstance(grid_points, torch.Tensor) else torch.tensor(grid_points, dtype=torch.float32)
        else:
            Y = make_grid_2d(grid_side)
        K = Y.shape[0]
        n = X.shape[0]

        self.register_buffer("X", X)
        self.register_buffer("Y", Y)
        self.m = m
        self.n = n
        self.K = K
        self.eps = eps
        self.n_sinkhorn = n_sinkhorn

        # Learnable per-atom target-marginal logits
        self.psi = nn.Parameter(torch.randn(m, K) * 0.01)

        # Quadratic cost matrix (n, K), C_{lk} = 0.5 ||x_l - y_k||^2
        dist_sq = torch.cdist(X.unsqueeze(0), Y.unsqueeze(0)).squeeze(0) ** 2  # (n, K)
        self.register_buffer("cost", 0.5 * dist_sq)  # (n, K)

        # Warm-start buffers for Sinkhorn potentials (updated each forward)
        self.register_buffer("f_cache", torch.zeros(m, n))
        self.register_buffer("g_cache", torch.zeros(m, K))

    def set_eps(self, eps):
        """Update the entropic regularization (used by epsilon annealing)."""
        self.eps = float(eps)
        # Stale potentials under new eps; reset warm-start.
        self.f_cache.zero_()
        self.g_cache.zero_()

    def forward(self):
        """
        Returns:
            atoms: (m, n, 2) -- entropic OT map of each atom evaluated at X
        """
        eps = self.eps
        n, K, m = self.n, self.K, self.m
        device = self.psi.device

        log_a = torch.full((n,), -math.log(n), device=device)       # (n,)
        log_b = F.log_softmax(self.psi, dim=1)                      # (m, K)
        C = self.cost                                                # (n, K)

        # Warm start from cache (detached)
        f = self.f_cache.detach().clone()
        g = self.g_cache.detach().clone()

        # No-grad Sinkhorn iterations with detached log_b
        log_b_det = log_b.detach()
        with torch.no_grad():
            for _ in range(max(self.n_sinkhorn - 1, 0)):
                # Update f: (m, n) = eps * (log_a - logsumexp_k[(g - C)/eps + log_b])
                inner_f = (g.unsqueeze(1) - C.unsqueeze(0)) / eps + log_b_det.unsqueeze(1)  # (m, n, K)
                f = eps * (log_a.unsqueeze(0) - torch.logsumexp(inner_f, dim=2))            # (m, n)
                # Update g: (m, K) = eps * (log_b - logsumexp_l[(f - C)/eps + log_a])
                inner_g = (f.unsqueeze(2) - C.unsqueeze(0)) / eps + log_a.view(1, n, 1)     # (m, n, K)
                g = eps * (log_b_det - torch.logsumexp(inner_g, dim=1))                     # (m, K)

        # Cache potentials for next forward pass
        self.f_cache.copy_(f.detach())
        self.g_cache.copy_(g.detach())

        # One final Sinkhorn step with grad w.r.t. log_b (i.e., psi)
        inner_f = (g.unsqueeze(1) - C.unsqueeze(0)) / eps + log_b.unsqueeze(1)
        f = eps * (log_a.unsqueeze(0) - torch.logsumexp(inner_f, dim=2))
        inner_g = (f.unsqueeze(2) - C.unsqueeze(0)) / eps + log_a.view(1, n, 1)
        g = eps * (log_b - torch.logsumexp(inner_g, dim=1))

        # Transport plan in log domain, then exponentiate
        log_pi = (log_a.view(1, n, 1)
                  + log_b.unsqueeze(1)
                  + (f.unsqueeze(2) + g.unsqueeze(1) - C.unsqueeze(0)) / eps)  # (m, n, K)
        pi = log_pi.exp()

        # Barycentric projection:  T_j(x_l) = (1/a_l) * sum_k pi_{lk} * y_k = n * sum_k pi * y
        atoms = n * torch.einsum("mnk, kd -> mnd", pi, self.Y)  # (m, n, 2)
        return atoms


class LISTAEncoder(nn.Module):
    """
    Multi-step LISTA encoder for sparse codes over L^2(rho) inner products.

    Step 0 (standard 1-step encoder):
        lam = chi( W @ input + b^{(0)} )

    where W @ input = (1/n) sum_l D_j(x_l) . input(x_l)  (inner products with atoms).

    Steps t = 1, ..., T-1 (LISTA unrolling):
        lam = chi( W @ input + S^{(t)} @ lam + b^{(t)} )

    where S^{(t)} is a learned (m, m) lateral inhibition matrix per step.

    With lista_steps=1 this reduces to the original single-step encoder.

    Activation options (activation_type):
        "relu"         -- ReLU(x) = max(0, x)
        "jumprelu"     -- JumpReLU: ReLU(x) * (x > threshold), straight-through gate.
        "topk"         -- keep only the top-k largest post-ReLU entries per sample.
        "topk_simplex" -- top-k then L1-renormalize so lam lives on the simplex.

    Args:
        m: number of dictionary atoms
        lista_steps: number of LISTA iterations (1 = original behavior)
        activation_type: one of "relu", "jumprelu", "topk", "topk_simplex"
        topk_k: k for topk / topk_simplex
        per_atom_gain: if True, multiply the inner-product input by a learnable
            per-atom gain before the activation -- decouples the scale of the
            encoder input from atom norms (helps when atoms aren't normalized).
        lateral_init: "zeros" (original) or "damped_identity" (S = 0.5*I + small
            noise).  The damped-identity init is the Gregor-LeCun style warm
            start for LISTA: without it, early steps are forced through zeros.
    """
    def __init__(self, m, lista_steps=1, activation_type="relu",
                 topk_k=3, per_atom_gain=False, lateral_init="zeros"):
        super().__init__()
        self.m = m
        self.lista_steps = lista_steps
        self.activation_type = activation_type
        self.topk_k = topk_k
        self.per_atom_gain = per_atom_gain

        # One bias per step
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(m))
                                        for _ in range(lista_steps)])

        # Lateral inhibition matrices for steps 1, 2, ...
        if lateral_init == "damped_identity":
            laterals = []
            for _ in range(lista_steps - 1):
                S = 0.5 * torch.eye(m) + 0.01 * torch.randn(m, m)
                laterals.append(nn.Parameter(S))
            self.laterals = nn.ParameterList(laterals)
        else:
            self.laterals = nn.ParameterList([nn.Parameter(torch.zeros(m, m))
                                              for _ in range(lista_steps - 1)])

        # Optional learnable per-atom gain on the inner-product input
        if per_atom_gain:
            self.gain = nn.Parameter(torch.ones(m))
        else:
            self.register_parameter("gain", None)

        # JumpReLU: one learned threshold vector per step
        if activation_type == "jumprelu":
            self.thresholds = nn.ParameterList([nn.Parameter(torch.zeros(m))
                                                for _ in range(lista_steps)])

    def _activate(self, pre_act, step):
        """Apply activation for a given LISTA step."""
        if self.activation_type == "relu":
            return F.relu(pre_act)
        elif self.activation_type == "jumprelu":
            thresh = self.thresholds[step]
            gate = (pre_act > thresh).float()
            # Straight-through estimator: forward = gate, backward = 1
            gate_st = pre_act - (pre_act - gate).detach()
            return F.relu(pre_act) * gate_st
        elif self.activation_type == "topk":
            return _topk_activation(pre_act, self.topk_k, renormalize=False)
        elif self.activation_type == "topk_simplex":
            return _topk_activation(pre_act, self.topk_k, renormalize=True)
        else:
            raise ValueError(f"Unknown activation_type: {self.activation_type}")

    def forward(self, inner):
        """
        Args:
            inner: (B, m) -- pre-computed L^2(rho) inner products with atoms

        Returns:
            lam: (B, m) -- sparse codes after lista_steps iterations
        """
        if self.gain is not None:
            inner = inner * self.gain  # (B, m)

        # Step 0
        lam = self._activate(inner + self.biases[0], step=0)  # (B, m)

        # Steps 1 ... T-1
        for t in range(self.lista_steps - 1):
            pre_act = inner + lam @ self.laterals[t].T + self.biases[t + 1]
            lam = self._activate(pre_act, step=t + 1)

        return lam


def _topk_activation(pre_act, k, renormalize=False):
    """
    Keep only the top-k largest post-ReLU entries per sample; zero the rest.

    If renormalize=True, the surviving entries are L1-normalized so lam lives
    on the probability simplex (per sample).  If the top-k values are all
    zero (can happen early in training), the output is left as zeros.

    Args:
        pre_act: (B, m)
        k: int, number of active atoms
        renormalize: bool, if True project onto the simplex

    Returns:
        lam: (B, m) with at most k nonzero entries per row.
    """
    x = F.relu(pre_act)                                    # (B, m)
    k = min(k, x.shape[1])
    vals, idx = torch.topk(x, k, dim=1)                    # (B, k)
    if renormalize:
        s = vals.sum(dim=1, keepdim=True)                  # (B, 1)
        vals = vals / s.clamp(min=1e-8)
    out = torch.zeros_like(x)
    out.scatter_(1, idx, vals)
    return out


def normalize_atoms_l2rho(atoms, n, eps=1e-8):
    """
    L^2(rho)-normalize atoms for use in the encoder.

    |A_j|_{L^2(rho)} = sqrt( (1/n) sum_l |A_j(x_l)|^2 )
    A_tilde_j = A_j / (|A_j|_{L^2(rho)} + eps)

    Args:
        atoms: (m, n, 2)
        n: number of support points (for the 1/n weight)
        eps: small constant for numerical stability

    Returns:
        atoms_normalized: (m, n, 2)
    """
    # |A_j|^2_{L^2(rho)} = (1/n) sum_l |A_j(x_l)|^2
    norm_sq = (atoms ** 2).sum(dim=(1, 2)) / n   # (m,)
    norm = norm_sq.sqrt()                          # (m,)
    return atoms / (norm[:, None, None] + eps)     # (m, n, 2)


def _build_atoms_module(atoms_type, X, m, eps, grid_side, grid_points, n_sinkhorn):
    """Dispatch between Gibbs (row-softmax) and Sinkhorn (entropic OT) atoms."""
    if atoms_type == "gibbs":
        return GibbsAtoms(X, m, eps, grid_side, grid_points=grid_points)
    elif atoms_type == "sinkhorn":
        return SinkhornAtoms(X, m, eps, grid_side=grid_side,
                             grid_points=grid_points, n_sinkhorn=n_sinkhorn)
    else:
        raise ValueError(f"Unknown atoms_type: {atoms_type}")


class DisplacementFieldSAE(nn.Module):
    """
    Sparse autoencoder operating on displacement fields V = T - Id.

    Encoding (multi-step LISTA):
        V_j = T_j - Id
        inner_j = <A_j, V_{rho->mu}>_{L^2(rho)}   where A_j = V_j (or normalized V_j)
        lam^{(0)} = chi(inner + b^{(0)})
        lam^{(t+1)} = chi(inner + S^{(t)} @ lam^{(t)} + b^{(t+1)})

    Reconstruction (always with unnormalized atoms):
        V_hat = sum_j lambda_j * V_j
        T_hat = Id + V_hat

    Args:
        X: (n, 2) base measure support
        m: number of dictionary atoms
        eps: Gibbs softmax temperature
        grid_side: side length of target grid
        lista_steps: number of LISTA iterations (1 = original single-step)
        activation_type: "relu" or "jumprelu"
        normalize_atoms: if True, use L^2(rho)-normalized atoms in the encoder
    """
    def __init__(self, X, m, eps, grid_side=32, lista_steps=1, activation_type="relu",
                 normalize_atoms=False, grid_points=None,
                 atoms_type="gibbs", n_sinkhorn=30,
                 topk_k=3, per_atom_gain=False, lateral_init="zeros"):
        super().__init__()
        self.atoms_module = _build_atoms_module(
            atoms_type, X, m, eps, grid_side, grid_points, n_sinkhorn,
        )
        self.encoder = LISTAEncoder(
            m, lista_steps=lista_steps, activation_type=activation_type,
            topk_k=topk_k, per_atom_gain=per_atom_gain, lateral_init=lateral_init,
        )
        self.n = X.shape[0]
        self.m = m
        self.normalize_atoms = normalize_atoms
        self.register_buffer("X", X)  # (n, 2)

    def encode(self, V, V_atoms_enc):
        """
        Args:
            V: (B, n, 2) batch of displacement fields
            V_atoms_enc: (m, n, 2) displacement atoms for encoding (possibly normalized)

        Returns:
            lam: (B, m)
        """
        inner = torch.einsum("bnd, mnd -> bm", V, V_atoms_enc) / self.n
        lam = self.encoder(inner)
        return lam

    def decode(self, lam, V_atoms):
        """
        Args:
            lam: (B, m)
            V_atoms: (m, n, 2) unnormalized displacement atoms

        Returns:
            T_hat: (B, n, 2)
        """
        V_hat = torch.einsum("bm, mnd -> bnd", lam, V_atoms)
        T_hat = self.X.unsqueeze(0) + V_hat  # (B, n, 2) = Id + V_hat
        return T_hat

    def forward(self, T):
        """
        Args:
            T: (B, n, 2) batch of transport maps

        Returns:
            T_hat: (B, n, 2) reconstructed maps
            lam: (B, m) encoding coefficients
        """
        atoms = self.atoms_module()                                          # (m, n, 2)
        V_atoms = atoms - self.X.unsqueeze(0)                                # (m, n, 2)
        V_atoms_enc = normalize_atoms_l2rho(V_atoms, self.n) if self.normalize_atoms else V_atoms
        V = T - self.X.unsqueeze(0)                                          # (B, n, 2)
        lam = self.encode(V, V_atoms_enc)                                    # (B, m)
        T_hat = self.decode(lam, V_atoms)                                    # (B, n, 2)
        return T_hat, lam
