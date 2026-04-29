import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import SAE
from SAE_analysis_functions import *
import numpy as np
import torch
import matplotlib.pyplot as plt
import ot
try:
    import umap
except ImportError:
    umap = None
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def summarize_labels(
    labels,
    strip_numeric_prefix=True,
    collapse_swap=False,
    swap_label="swap",
):
    """
    labels: list/array of strings

    strip_numeric_prefix:
        '270_delete_words' -> 'delete_words'

    collapse_swap:
        any label containing 'swap' -> 'swap'
    """
    out = []

    for lab in labels:
        s = str(lab)

        if strip_numeric_prefix:
            s = re.sub(r"^\d+_", "", s)

        if collapse_swap and "swap" in s:
            s = swap_label

        out.append(s)

    return np.array(out)


summarize_corruption_labels = summarize_labels



#Potential Model Eval

def load_stacked_tensors_from_dirs(
    data_dir,
    n_per_dir,
    ext=".pt",
    tensor_key="object",
):
    data_dir = Path(data_dir)
    dir_names = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])

    name_to_id = {name: i for i, name in enumerate(dir_names)}
    id_to_name = {i: name for name, i in name_to_id.items()}

    all_tensors = []
    labels = []

    for name in dir_names:
        d = data_dir / name
        pt_files = sorted(d.rglob(f"*{ext}"))

        if len(pt_files) < n_per_dir:
            raise ValueError(f"{name}: only {len(pt_files)} files")

        for p in pt_files[:n_per_dir]:
            print(p)
            obj = torch.load(p)
            t = obj[tensor_key] if isinstance(obj, dict) else obj
            all_tensors.append(t if torch.is_tensor(t) else torch.tensor(t))
            labels.append(name_to_id[name])

    stacked_tensor = torch.stack(all_tensors, dim=0)
    int_labels = torch.tensor(labels, dtype=torch.long)
    str_labels = [id_to_name[i] for i in labels]

    return stacked_tensor, int_labels, str_labels, name_to_id, id_to_name

#EXAMPLE
'''
DATA_DIR = 'datasets/noised_luther/potentials/EleutherAI__pythia-410m-deduped'
DIR_NAMES = sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()])
N_PER_DIR=955
EXT= ".pt"
'''

def plot_pca_plotly(
    X,
    labels,
    title="PCA scatter (colored by label)",
    dim=3,                    # 2 or 3
    standardize=True,
    marker_size=None,
    opacity=0.6,
):
    X = np.asarray(X)
    labels = np.asarray(labels).astype(str)

    if X.ndim != 2 or X.shape[1] < dim:
        raise ValueError(f"X must be shape (N, >= {dim}). Got {X.shape}")
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")

    if standardize:
        Xp = StandardScaler().fit_transform(X)
    else:
        Xp = X

    P = PCA(n_components=dim, random_state=0).fit_transform(Xp)  # (N, dim)

    if marker_size is None:
        marker_size = 6 if dim == 2 else 2.5

    if dim == 2:
        fig = px.scatter(
            x=P[:, 0], y=P[:, 1],
            color=labels, title=title, opacity=opacity,
            labels={"x": "PC1", "y": "PC2", "color": "Label"},
        )
    else:
        fig = px.scatter_3d(
            x=P[:, 0], y=P[:, 1], z=P[:, 2],
            color=labels, title=title, opacity=opacity,
            labels={"x": "PC1", "y": "PC2", "z": "PC3", "color": "Label"},
        )

    fig.update_traces(marker=dict(size=marker_size))
    fig.update_layout(
        legend=dict(title="Label", x=1.02, y=1.0, xanchor="left", yanchor="top"),
        margin=dict(l=40, r=180, t=60, b=40),
    )
    fig.show()
    return fig, P


def plot_embedding_plotly(
    Z,
    labels,
    title="Embedding scatter (colored by label)",
    dim=None,
    marker_size=None,
    opacity=0.6,
):
    Z = np.asarray(Z)
    labels = np.asarray(labels)

    if Z.ndim != 2 or Z.shape[1] < 2:
        raise ValueError(f"Z must be shape (N, >=2). Got {Z.shape}")

    if dim is None:
        dim = 3 if Z.shape[1] >= 3 else 2
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")
    if dim == 3 and Z.shape[1] < 3:
        raise ValueError(f"Need Z with >=3 columns for 3D plot. Got {Z.shape}")

    if marker_size is None:
        marker_size = 6 if dim == 2 else 2.5

    labels = labels.astype(str)
    if dim == 2:
        fig = px.scatter(
            x=Z[:, 0], y=Z[:, 1],
            color=labels, title=title, opacity=opacity,
            labels={"x": "Dim 1", "y": "Dim 2", "color": "Label"},
        )
    else:
        fig = px.scatter_3d(
            x=Z[:, 0], y=Z[:, 1], z=Z[:, 2],
            color=labels, title=title, opacity=opacity,
            labels={"x": "Dim 1", "y": "Dim 2", "z": "Dim 3", "color": "Label"},
        )

    fig.update_traces(marker=dict(size=marker_size))
    fig.update_layout(
        legend=dict(title="Label", x=1.02, y=1.0, xanchor="left", yanchor="top"),
        margin=dict(l=40, r=180, t=60, b=40),
    )
    fig.show()
    return fig


def umap_profile_categorical_legend(
    X,
    labels,
    dim=None,                      # 2 or 3; if None, infer (prefer 3)
    renderer="browser",
    point_size=None,               # defaults: 2D=6, 3D=3
    opacity=0.2,
    max_labels_in_legend=None,     # e.g. 30 if you have tons of labels
    title=None,
    write_html=None,               # e.g. "umap.html" or None
    **umap_kwargs
):
    if umap is None:
        raise ImportError(
            "umap-learn is required for umap_profile_categorical_legend. "
            "Install it with `pip install umap-learn`."
        )

    pio.renderers.default = renderer
    X = np.asarray(X)
    labels = np.asarray(labels)

    if dim is None:
        dim = 3
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")

    uniq = np.unique(labels)
    print("X shape:", X.shape, "| #labels:", len(uniq), "| dim:", dim)

    # --- UMAP ---
    reducer = umap.UMAP(n_components=dim, **umap_kwargs)
    Y = reducer.fit_transform(X)

    # Optional: limit legend entries without changing coloring
    legend_set = set(uniq)
    if max_labels_in_legend is not None and len(uniq) > max_labels_in_legend:
        legend_set = set(uniq[:max_labels_in_legend])
        print(f"Showing only {max_labels_in_legend} labels in legend (still colors by label).")

    if point_size is None:
        point_size = 6 if dim == 2 else 3

    # --- Plotly categorical legend: one trace per label ---
    fig = go.Figure()

    if dim == 2:
        for lab in uniq:
            idx = labels == lab
            fig.add_trace(go.Scatter(
                x=Y[idx, 0], y=Y[idx, 1],
                mode="markers",
                name=str(lab),
                showlegend=(lab in legend_set),
                marker=dict(size=point_size, opacity=opacity),
                text=labels[idx].astype(str),
                hovertemplate="label=%{text}<extra></extra>",
            ))

        fig.update_layout(
            title=title or "UMAP 2D (categorical legend by label)",
            xaxis_title="UMAP1",
            yaxis_title="UMAP2",
            legend=dict(
                yanchor="top", y=1,
                xanchor="left", x=1.02,
                itemsizing="constant",
            ),
            margin=dict(l=40, r=200, t=60, b=40),
        )

    else:  # dim == 3
        for lab in uniq:
            idx = labels == lab
            fig.add_trace(go.Scatter3d(
                x=Y[idx, 0], y=Y[idx, 1], z=Y[idx, 2],
                mode="markers",
                name=str(lab),
                showlegend=(lab in legend_set),
                marker=dict(size=point_size, opacity=opacity),
                text=labels[idx].astype(str),
                hovertemplate="label=%{text}<extra></extra>",
            ))

        fig.update_layout(
            title=title or "UMAP 3D (categorical legend by label)",
            scene=dict(
                xaxis_title="UMAP1",
                yaxis_title="UMAP2",
                zaxis_title="UMAP3",
            ),
            legend=dict(
                yanchor="top", y=1,
                xanchor="left", x=1.02,
                itemsizing="constant",
            ),
            margin=dict(l=0, r=200, t=60, b=0),
        )

    if write_html is not None:
        fig.write_html(write_html, include_plotlyjs="cdn")

    fig.show()
    return Y, reducer, fig

#EX
'''
Y2, reducer2, fig2 = umap_profile_categorical_legend(X, labels, dim=2, n_neighbors=15, min_dist=0.1)
'''
