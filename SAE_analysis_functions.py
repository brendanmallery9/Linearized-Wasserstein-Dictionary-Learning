import torch
from SAE import *
import matplotlib.pyplot as plt
from dataclasses import dataclass
import numpy as np



#Extract weights from a trained SAE
def extract_weights(model_dir,supp_size,no_atoms):
    # load model
    model = ReLUAE(supp_size, no_atoms)
    model.load_state_dict(torch.load(model_dir, map_location="cpu"))
    model.eval()

    # extract dictionary
    W = model.enc.weight.detach()   # (k, m)
    return W

ARCH_REGISTRY = {
    "TopKAE":              TopKAE,
    "topk":                TopKAE,
    "JumpReLU":            JumpReLUAE,
    "GatedSAE":            GatedSAE,
    "ReLUAE":              ReLUAE,
    "TopKAE_monotone":     TopKAE_monotone,
    "JumpReLUAE_monotone": JumpReLUAE_monotone,
    "JumpReLU_monotone": JumpReLUAE_monotone,
    "ReLUAE_monotone":     ReLUAE_monotone,
    "JumpReLUAE_nonneg": JumpReLUAE_nonneg,
    "JumpReLU_nonneg":JumpReLUAE_nonneg
}



def sae_encode_potential_batch(
    data_tensor,
    model_path,
    architecture,
    m,
    hidden_dim,
    top_k,
    device="cuda",
    batch_size=2048,
):
    """
    Encode a big tensor/array in batches using a saved SAE model.
    data_tensor: (N, m) torch.Tensor or np.ndarray
    m: input dim (support size)
    """

    arch_cls = ARCH_REGISTRY[architecture]

    # Build model first (so we can infer/confirm device)
    if arch_cls in (TopKAE, TopKAE_monotone):
        model = arch_cls(m, hidden_dim, top_k=top_k)
    else:
        model = arch_cls(m, hidden_dim)

    # Normalize device
    device = torch.device(device) if not isinstance(device, torch.device) else device
    model = model.to(device)

    # Make X a torch tensor on the right device
    if torch.is_tensor(data_tensor):
        X = data_tensor.to(device=device, dtype=torch.float32)
    else:
        # numpy -> torch, stay on device, force float32
        X = torch.as_tensor(data_tensor, device=device, dtype=torch.float32)

    # Load checkpoint — unwrap whichever nesting key the trainer used
    # Two sequential ifs (not elif) to handle double-nested dicts
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"Warning: State dict mismatch. Error: {e}")
        print("Attempting to load with strict=False...")
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    Z = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = X[i : i + batch_size]  # already on device
            _, z = model(xb)
            Z.append(z.detach().cpu())

    return torch.cat(Z, dim=0)
#EXAMPLE
'''
codes=sae_encode_potential_batch(augment_stacked_potentials, 
                                model_path, 
                                'JumpReLU',
                                  1130, 
                                  40,
                                  4, 
                                  device="cuda", 
                                  batch_size=1024)
'''


#EXAMPLE
'''

code_tensor_sae = sae_encode_batch(data_tensor, 
                                   "noised_kafka_SAE/l1_0.01/sparse_ae.pt",m=supp_size,k=no_atoms,
                                    device="cuda", batch_size=2048)

'''


def relu_threshold(x, lam):
    return torch.clamp(x - lam, min=0.0)

def power_iteration_spectral_norm(A, n_iter=50):
    # returns ||A||_2 (spectral norm) approx
    device = A.device
    v = torch.randn(A.shape[1], device=device)
    v = v / (v.norm() + 1e-12)
    for _ in range(n_iter):
        v = A.T @ (A @ v)
        v = v / (v.norm() + 1e-12)
    Av = A @ v
    return Av.norm().item()

#Extract coefficients with fista

def fista_lasso(A, b, lam, max_iter, tol, stepsize, verbose=False):
    """
    Solve:  min_x  0.5||A x - b||_2^2 + lam ||x||_1   (LASSO)
    A: (m, k), b: (m,) or (m,1)  -> x: (k,)
    """
    if b.ndim == 2 and b.shape[1] == 1:
        b = b[:, 0]
    assert A.ndim == 2 and b.ndim == 1 and A.shape[0] == b.shape[0]

    x = torch.zeros(A.shape[1], device=A.device, dtype=A.dtype)
    y = x.clone()
    t = 1.0

    for it in range(max_iter):
        x_old = x

        # grad of 0.5||Ay-b||^2 is A^T(Ay-b)
        grad = A.T @ (A @ y - b)
        x = relu_threshold(y - grad / stepsize, lam / stepsize)

        t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
        y = x + ((t - 1.0) / t_new) * (x - x_old)
        t = t_new

        if tol is not None:
            rel = (x - x_old).norm() / (x_old.norm() + 1e-12)
            if verbose and (it % 25 == 0 or it == max_iter - 1):
                obj = 0.5 * (A @ x - b).pow(2).sum() + lam * x.abs().sum()
                print(f"it {it:4d} | rel {rel:.2e} | obj {obj.item():.6g}")
            if rel.item() < tol:
                break
    return x


def fista_batch(data_tensor,model_dir,supp_size,no_atoms,l1,max_iter,tol):
    data_tensor=data_tensor.float()
    
    W=extract_weights(model_dir,supp_size,no_atoms).float()
    W=W.T
    s = power_iteration_spectral_norm(W)
    L = s * s  # Lipschitz const of grad = ||W^T W|| = ||W||_2^2
    code_tensor=[]
    counter=0
    for row in data_tensor:
        code=fista_lasso(W, row, l1, max_iter, tol, L, verbose=False)
        code_tensor.append(code)
        counter+=1
        print(counter)
    code_tensor=torch.stack(code_tensor)
    return code_tensor

#EXAMPLE

'''
code_tensor_a=fista_batch(data_tensor,
            "noised_kafka_SAE/40atoms1e-2/sparse_ae.pt",
            supp_size=991,
            no_atoms=40,
            l1=0.01,
            max_iter=200,
            tol=1e-6)

'''

class LabeledData:
    def __init__(self, idx: int, data: torch.Tensor, code: torch.Tensor, label: str):
        self.idx = idx
        self.data = data
        self.code = code
        self.label = label


import math
def plot_blocked_l2_histograms(
    C,
    row_labels,
    bins=50,
    fig_width=12,
    row_height=2.2,
    sharex=True,
    title="Row-wise ℓ2 norm histograms by label block",
):
    C = np.asarray(C.detach().cpu()) if hasattr(C, "detach") else np.asarray(C)
    row_labels = np.asarray(row_labels)
    N = C.shape[0]
    assert len(row_labels) == N

    blocks = []
    start = 0
    for i in range(1, N + 1):
        if i == N or row_labels[i] != row_labels[start]:
            blocks.append((row_labels[start], start, i - 1))
            start = i

    l2 = np.linalg.norm(C, axis=1)

    B = len(blocks)
    ncols = 3 if B >= 3 else B
    nrows = math.ceil(B / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_width, row_height * nrows),
        sharex=sharex
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, (name, a, b) in zip(axes, blocks):
        ax.hist(l2[a:b+1], bins=bins)
        ax.set_title(f"{name} (n={b-a+1})")
        ax.set_ylabel("count")
        ax.set_xlabel("||row||₂")

    for ax in axes[len(blocks):]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_coeff_matrix(
    C,
    row_labels,
    title="Coefficient matrix",
    vmax=None,
    vmin=None,
    cmap="viridis",
):
    C = np.asarray(C.detach().cpu()) if hasattr(C, "detach") else np.asarray(C)
    row_labels = np.asarray(row_labels)
    N = C.shape[0]
    assert len(row_labels) == N

    block_centers = []
    block_names = []
    start = 0
    for i in range(1, N + 1):
        if i == N or row_labels[i] != row_labels[start]:
            block_centers.append((start + i - 1) / 2)
            block_names.append(row_labels[start])
            start = i

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(C, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax)

    ax.set_xlabel("Atom index")
    ax.set_ylabel("Sample index")
    ax.set_title(title)
    ax.set_yticks(block_centers)
    ax.set_yticklabels(block_names)

    fig.tight_layout()
    return fig



def sae_encode_activation_batch(
    data_dir,
    model_path,
    architecture,
    hidden_dim,
    top_k,
    N_PER_DIR,
    avg_method,
    device="cuda",
    ext=".pt",
    chunk_size=100,):

    data_dir = Path(data_dir)
    pt_files = sorted(data_dir.rglob(f"*{ext}"))[:N_PER_DIR]

    arch_cls = ARCH_REGISTRY[architecture]

    state = torch.load(model_path, map_location="cpu")
    embed_dim=state['enc.weight'].shape[1]
    if arch_cls == TopKAE:
        model = arch_cls(embed_dim, hidden_dim, top_k=top_k).to(device)
    else:
        model = arch_cls(embed_dim, hidden_dim).to(device)

    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"Warning: State dict mismatch: {e}")
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    summaries = []
    # iterate in chunks over files
    for start in range(0, len(pt_files), chunk_size):
        batch_files = pt_files[start : start + chunk_size]

        # load tensors
        tensor_list = [torch.load(p)["acts"] for p in batch_files]  # each (ctx, m) presumably

        with torch.no_grad():
            for X in tensor_list:
                X = X.to(device).float() 
                _, z = model(X)        

                if avg_method == "mean":
                    # mean over context/tokens, keep hidden_dim
                    avg_code = z.mean(dim=0)            # (hidden_dim,)
                elif avg_method == "topk_mean":
                    k = min(top_k, z.shape[0])          # safety
                    row_norms = z.norm(p=2, dim=1)      # (num_rows,)
                    topk_idx = row_norms.topk(k).indices
                    avg_code = z[topk_idx].mean(dim=0)  # (hidden_dim,)
                else:
                    raise ValueError(f"Unknown avg_method: {avg_method}")
                summaries.append(avg_code.detach().cpu())
    # (num_files, hidden_dim)
    return torch.stack(summaries, dim=0)

#EXAMPLE
'''
batch=sae_encode_activation_batch(
    DATA_DIR,
    MODEL_PATH,
    'TopKAE',
    2048,
    16,
    955,
    'mean',
    device="cuda",
    ext=".pt",
    chunk_size=100,)
batch_list.append(batch)
labels.extend([DIR_NAME] * batch.shape[0])
'''