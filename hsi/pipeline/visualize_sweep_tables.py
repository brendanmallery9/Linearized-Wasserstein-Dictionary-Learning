#!/usr/bin/env python3
"""
Visualize sweep results as DataFrames (CSV) and PNG tables.

Produces per clustering method and a "best over methods" CSV + PNG.

Usage:
    python visualize_sweep_tables.py drop_contig_tables drop_contig.json
    python visualize_sweep_tables.py my_experiment sweep_results_drop_random.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd


def load_sweep(path):
    with open(path) as f:
        return json.load(f)


def parse_from_runs(data):
    runs = data.get("runs", [])
    if not runs:
        print("ERROR: No runs found in JSON.")
        sys.exit(1)

    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    for run in runs:
        k = run["k"]
        for section in ("aggregated_sae", "aggregated_nmf"):
            for method_key, agg_list in run.get(section, {}).items():
                parts = method_key.split("|", 1)
                if len(parts) == 2:
                    dataset, key_name = parts
                else:
                    dataset, key_name = "unknown", method_key
                for entry in agg_list:
                    clust_method = entry.get("method", "unknown")
                    acc = entry.get("accuracy_mean")
                    if acc is not None:
                        raw[dataset][key_name][clust_method][k].append(acc)
    return raw


def _k_sort_key(x):
    return (x is not None, x if x is not None else 0)


def _k_label(k):
    return "clean" if k is None else f"k={k}"


import re as _re

def _model_sort_key(key_name):
    """Sort model names so that _mon, _nonneg, and NMF (rank*) are grouped.

    Order: _mon models first, _nonneg models second, NMF/rank* last.
    Within each group, sort alphabetically.
    """
    if key_name.endswith("_mon"):
        return (0, key_name)
    if key_name.endswith("_nonneg"):
        return (1, key_name)
    if _re.match(r"^rank\d+$", key_name) or _re.match(r"^NMF", key_name):
        return (2, key_name)
    return (3, key_name)


def _model_display_name(key_name):
    """Human-readable label: rankN -> NMF (rankN), others unchanged."""
    m = _re.match(r"^rank(\d+)$", key_name)
    if m:
        return f"NMF (rank{m.group(1)})"
    return key_name


def _model_type_label(key_name):
    """Short type label for the 'type' column."""
    if key_name.endswith("_mon"):
        return "OT-SAE"
    if key_name.endswith("_nonneg"):
        return "N-SAE"
    if _re.match(r"^rank\d+$", key_name) or _re.match(r"^NMF", key_name):
        return "NMF"
    return ""


def _sorted_model_keys(dataset_dict):
    """Return model keys for one dataset, grouped and sorted."""
    return sorted(dataset_dict.keys(), key=_model_sort_key)


def build_dataframe(raw, clustering_method):
    """Build a DataFrame for one clustering method.

    Index: (dataset, key_name)
    Columns: one per k value, formatted as "mean ± std"
    """
    datasets = sorted(raw.keys())
    k_set = set()
    for dataset in datasets:
        for key_name in raw[dataset]:
            if clustering_method in raw[dataset][key_name]:
                k_set.update(raw[dataset][key_name][clustering_method].keys())
    k_values = sorted(k_set, key=_k_sort_key)

    records = []
    for dataset in datasets:
        for key_name in _sorted_model_keys(raw[dataset]):
            method_data = raw[dataset][key_name].get(clustering_method, {})
            row = {"dataset": dataset, "type": _model_type_label(key_name),
                   "key": _model_display_name(key_name)}
            for k in k_values:
                vals = method_data.get(k, [])
                if vals:
                    arr = np.array(vals)
                    row[f"k={k}_mean"] = float(np.mean(arr))
                    row[f"k={k}_std"] = float(np.std(arr))
                else:
                    row[f"k={k}_mean"] = np.nan
                    row[f"k={k}_std"] = np.nan
            records.append(row)

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.set_index(["dataset", "type", "key"])
    return df


def build_best_dataframe(raw):
    """Build a DataFrame taking the best accuracy across all clustering methods."""
    datasets = sorted(raw.keys())
    k_set = set()
    all_methods = set()
    for dataset in datasets:
        for key_name in raw[dataset]:
            for clust_method in raw[dataset][key_name]:
                all_methods.add(clust_method)
                k_set.update(raw[dataset][key_name][clust_method].keys())
    k_values = sorted(k_set, key=_k_sort_key)

    records = []
    for dataset in datasets:
        for key_name in _sorted_model_keys(raw[dataset]):
            row = {"dataset": dataset, "type": _model_type_label(key_name),
                   "key": _model_display_name(key_name)}
            for k in k_values:
                best_mean = -1.0
                best_std = 0.0
                best_method = ""
                for clust_method in all_methods:
                    vals = raw[dataset][key_name].get(clust_method, {}).get(k, [])
                    if vals:
                        m = float(np.mean(vals))
                        if m > best_mean:
                            best_mean = m
                            best_std = float(np.std(vals))
                            best_method = clust_method
                if best_mean >= 0:
                    row[f"k={k}_mean"] = best_mean
                    row[f"k={k}_std"] = best_std
                    row[f"k={k}_best_method"] = best_method
                else:
                    row[f"k={k}_mean"] = np.nan
                    row[f"k={k}_std"] = np.nan
                    row[f"k={k}_best_method"] = ""
            records.append(row)

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.set_index(["dataset", "type", "key"])
    return df


# ---------------------------------------------------------------------------
# PNG rendering
# ---------------------------------------------------------------------------

# Colour palette (dataset group header rows)
_DS_COLOURS = [
    "#dce6f1",  # soft blue
    "#e2efda",  # soft green
    "#fce4d6",  # soft orange
    "#ede7f6",  # soft purple
    "#fff9c4",  # soft yellow
    "#fce4ec",  # soft pink
]
_HEADER_COLOUR = "#2d3a4a"   # dark blue-grey for column headers
_BEST_FG      = "#1a5c1a"    # dark green text for best cell
_BEST_BG      = "#ffffcc"    # light yellow bg for best cell


_TYPE_COLOURS = {
    "OT-SAE": "#cfe2f3",   # blue tint
    "N-SAE":  "#d9ead3",   # green tint
    "NMF":    "#fce5cd",   # orange tint
    "":       "#f3f3f3",
}


def _render_df_png(raw, k_values, datasets, clustering_method, title, output_path):
    """Render a per-method (or BEST) PNG table.

    Columns (all as data columns, no rowLabels):
      0: dataset
      1: type  (OT-SAE / N-SAE / NMF)
      2: model name
      3+: k values

    The best value in each k-column *within each dataset group* is bolded.
    """
    # ------------------------------------------------------------------ #
    # 1. Collect display data                                              #
    # ------------------------------------------------------------------ #
    k_col_labels = [_k_label(k) for k in k_values]
    # col 0=dataset, 1=type, 2=model, 3+=k values
    col_labels = ["dataset", "type", "model"] + k_col_labels
    n_cols   = len(col_labels)
    n_k_cols = len(k_col_labels)

    # row_info: list of (dataset, display_name, type_label, {k: (mean, std)})
    row_info = []
    for dataset in datasets:
        for key_name in _sorted_model_keys(raw[dataset]):
            cell_vals = {}
            for k in k_values:
                if clustering_method == "BEST":
                    best_mean, best_std = float("nan"), float("nan")
                    for cm, kd in raw[dataset][key_name].items():
                        vals = kd.get(k, [])
                        if vals:
                            m = float(np.mean(vals))
                            if np.isnan(best_mean) or m > best_mean:
                                best_mean = m
                                best_std = float(np.std(vals))
                    cell_vals[k] = (best_mean, best_std)
                else:
                    vals = raw[dataset][key_name].get(clustering_method, {}).get(k, [])
                    if vals:
                        arr = np.array(vals)
                        cell_vals[k] = (float(np.mean(arr)), float(np.std(arr)))
                    else:
                        cell_vals[k] = (float("nan"), float("nan"))
            row_info.append((
                dataset,
                _model_display_name(key_name),
                _model_type_label(key_name),
                cell_vals,
            ))

    n_rows = len(row_info)
    if n_rows == 0:
        return

    # ------------------------------------------------------------------ #
    # 2. Build cell text + find best per (dataset, k-col index)           #
    # ------------------------------------------------------------------ #
    cell_text = []
    for (dataset, disp_name, type_label, cell_vals) in row_info:
        row_texts = [dataset, type_label, disp_name]
        for k in k_values:
            mean, std = cell_vals[k]
            if np.isnan(mean):
                row_texts.append("—")
            elif np.isnan(std) or std == 0.0:
                row_texts.append(f"{mean:.4f}")
            else:
                row_texts.append(f"{mean:.4f} ± {std:.4f}")
        cell_text.append(row_texts)

    # best mean per (dataset, k-col index)
    best_per = {}
    for ri, (dataset, _, _, cell_vals) in enumerate(row_info):
        for ki, k in enumerate(k_values):
            mean, _ = cell_vals[k]
            if not np.isnan(mean):
                prev = best_per.get((dataset, ki), float("-inf"))
                if mean > prev:
                    best_per[(dataset, ki)] = mean

    # bold_mask[ri][ki] — only for k-value columns
    bold_mask = []
    for ri, (dataset, _, _, cell_vals) in enumerate(row_info):
        row_bolds = []
        for ki, k in enumerate(k_values):
            mean, _ = cell_vals[k]
            is_best = (
                not np.isnan(mean)
                and (dataset, ki) in best_per
                and abs(mean - best_per[(dataset, ki)]) < 1e-9
            )
            row_bolds.append(is_best)
        bold_mask.append(row_bolds)

    # ------------------------------------------------------------------ #
    # 3. Figure sizing                                                     #
    # ------------------------------------------------------------------ #
    ds_col_w    = 1.4
    type_col_w  = 1.0
    model_col_w = 3.2
    k_col_w     = max(1.8, 10 / max(n_k_cols, 1))
    row_h       = 0.38
    fig_w = ds_col_w + type_col_w + model_col_w + n_k_cols * k_col_w
    fig_h = 1.0 + n_rows * row_h + 0.6

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    # ------------------------------------------------------------------ #
    # 4. Draw table (no rowLabels — everything is a data column)          #
    # ------------------------------------------------------------------ #
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Set column widths via the header cells
    col_widths = [ds_col_w, type_col_w, model_col_w] + [k_col_w] * n_k_cols
    total_w = sum(col_widths)
    for j, w in enumerate(col_widths):
        table[0, j].set_width(w / total_w)

    # Column headers
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor(_HEADER_COLOUR)
        cell.set_text_props(color="white", fontweight="bold", fontsize=8)

    # Dataset group colours
    ds_colour_map = {
        ds: _DS_COLOURS[i % len(_DS_COLOURS)]
        for i, ds in enumerate(datasets)
    }

    for ri, (dataset, disp_name, type_label, _) in enumerate(row_info):
        group_colour = ds_colour_map[dataset]
        type_colour  = _TYPE_COLOURS.get(type_label, group_colour)

        # Set widths for data rows too
        for j, w in enumerate(col_widths):
            table[ri + 1, j].set_width(w / total_w)

        # Dataset column (col 0)
        table[ri + 1, 0].set_facecolor(group_colour)
        table[ri + 1, 0].set_text_props(fontsize=7, fontstyle="italic")

        # Type column (col 1)
        table[ri + 1, 1].set_facecolor(group_colour)
        table[ri + 1, 1].set_text_props(fontweight="normal", fontsize=8)
        # Model name column (col 2)
        table[ri + 1, 2].set_facecolor(group_colour)
        table[ri + 1, 2].set_text_props(fontsize=7)

        # k-value columns (cols 3+)
        for ki in range(n_k_cols):
            cell = table[ri + 1, ki + 3]
            if bold_mask[ri][ki]:
                cell.set_facecolor(_BEST_BG)
                cell.set_text_props(fontweight="bold", color=_BEST_FG)
            else:
                cell.set_facecolor(group_colour)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved PNG: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize sweep results as DataFrames (CSV) and PNG tables")
    parser.add_argument("name", help="Name for the output directory (created if it doesn't exist) and used as file prefix")
    parser.add_argument("json_path", help="Path to sweep_results JSON file")
    args = parser.parse_args()

    output_dir = args.name
    prefix = os.path.basename(args.name)  # use just the leaf name as file prefix
    os.makedirs(output_dir, exist_ok=True)

    data = load_sweep(args.json_path)
    corruption_type = data.get("sweep_config", {}).get("corruption_type", "unknown")

    print(f"Corruption type: {corruption_type}")
    print(f"Output directory: {output_dir}")
    raw = parse_from_runs(data)

    datasets = sorted(raw.keys())
    all_methods = set()
    k_set = set()
    for dataset in raw:
        for key_name in raw[dataset]:
            for cm in raw[dataset][key_name]:
                all_methods.add(cm)
                k_set.update(raw[dataset][key_name][cm].keys())
    all_methods = sorted(all_methods)
    k_values = sorted(k_set, key=_k_sort_key)

    print(f"Datasets: {datasets}")
    print(f"Clustering methods: {all_methods}")

    all_dfs = {}

    for method in all_methods:
        df = build_dataframe(raw, method)
        safe_method = method.replace(" ", "_")
        out_csv = os.path.join(output_dir, f"df_{prefix}_{safe_method}.csv")
        df.to_csv(out_csv)
        all_dfs[method] = df
        print(f"  Saved {out_csv}  ({df.shape})")

        out_png = os.path.join(output_dir, f"table_{prefix}_{safe_method}.png")
        _render_df_png(
            raw, k_values, datasets,
            clustering_method=method,
            title=f"Accuracy ({method}) — corruption: {corruption_type}",
            output_path=out_png,
        )

    # Best over methods
    df_best = build_best_dataframe(raw)
    out_csv = os.path.join(output_dir, f"df_{prefix}_BEST.csv")
    df_best.to_csv(out_csv)
    all_dfs["BEST"] = df_best
    print(f"  Saved {out_csv}  ({df_best.shape})")

    out_png = os.path.join(output_dir, f"table_{prefix}_BEST.png")
    _render_df_png(
        raw, k_values, datasets,
        clustering_method="BEST",
        title=f"Accuracy (best over methods) — corruption: {corruption_type}",
        output_path=out_png,
    )

    print(f"\nDone! {len(all_methods) + 1} CSVs and {len(all_methods) + 1} PNGs written to {output_dir}/")


if __name__ == "__main__":
    main()

# Example usage:
# python visualize_sweep_tables.py log_diffeo log_warp.json
# -> creates my_experiment/ directory with files like:
#    my_experiment/df_my_experiment_BEST.csv
#    my_experiment/table_my_experiment_BEST.png
#    my_experiment/sweep_my_experiment.xlsx
