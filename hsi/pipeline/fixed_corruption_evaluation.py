#!/usr/bin/env python3
"""
Fixed-set corruption robustness evaluation in a single command.

Runs `clustering_eval.py` (via the corruption
wrapper) across the selected corruption types × every severity, then for each
corruption type produces one CSV + one PNG that compares all methods (NMF +
potentials SAE + linear SAE) side by side.

Each per-corruption-type table:
  Rows  : grouped into blocks by dataset (Pavia / Botswana / Salinas_A / …).
          Within each block: NMF, OT-SAE rows (potentials), N-SAE rows (linear).
  Columns:
    type    OT-SAE / N-SAE / NMF
    m       hidden_dim for SAE, rank for NMF
    c       l1 regularization (— for NMF)
    clean   accuracy with no corruption
    k=0.1, k=0.2, …   accuracy at each severity
  Cells = best clustering accuracy across {gmm, kmeans, spectral}.

Usage:
    python hsi/pipeline/fixed_corruption_evaluation.py \
        --root datasets/hsi_data \
        --output_dir hsi/results/fixed_corruption_evaluation \
        --key_pairs \
            potentials JUMPRELUAE_15_1e-1_mon \
            potentials JUMPRELUAE_10_5e-1_mon \
            linear     JUMPRELUAE_15_1e-3_nonneg \
            linear     JUMPRELUAE_10_1e-3_nonneg

If the per-corruption JSONs already exist in --output_dir, pass --skip_sweeps
to just rebuild the tables without re-running clustering.
"""

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse parsing helpers from the existing single-corruption visualizer
sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualize_sweep_tables import (  # noqa: E402
    parse_from_runs,
    _k_sort_key,
    _DS_COLOURS,
    _HEADER_COLOUR,
    _BEST_FG,
    _BEST_BG,
    _TYPE_COLOURS,
)

DEFAULT_CORRUPTIONS = [
    "drop_random",
    "drop_contiguous",
    "log_warp",
]


# ---------------------------------------------------------------------------
# Model-key parsing:  JUMPRELUAE_<m>_<c>_(mon|nonneg)  or  rank<m>
# ---------------------------------------------------------------------------
_RE_SAE = re.compile(r"^JUMPRELUAE_(\d+)_([^_]+)_(mon|nonneg)$")
_RE_NMF = re.compile(r"^rank(\d+)$")


def parse_model_key(key_name):
    """Return (type_label, m, c).  type_label in {OT-SAE, N-SAE, NMF}.

    For NMF, c is None (no l1 regularization).
    """
    m = _RE_NMF.match(key_name)
    if m:
        return "NMF", int(m.group(1)), None
    m = _RE_SAE.match(key_name)
    if m:
        type_label = "OT-SAE" if m.group(3) == "mon" else "N-SAE"
        return type_label, int(m.group(1)), m.group(2)
    # Unknown — surface raw name so it doesn't silently disappear
    return "?", 0, key_name


def _row_sort_key(parsed):
    """Sort within a dataset block: NMF first, then OT-SAE, then N-SAE."""
    type_label, m, c = parsed
    type_order = {"NMF": 0, "OT-SAE": 1, "N-SAE": 2}.get(type_label, 99)
    c_str = "" if c is None else str(c)
    return (type_order, m, c_str)


# ---------------------------------------------------------------------------
# 1. Run sweeps
# ---------------------------------------------------------------------------
def run_sweep_for_type(wrapper, root, ctype, k_values, cseeds, seeds,
                      key_pairs, output_json):
    cmd = [
        sys.executable, "-u", str(wrapper),
        "--root", str(root),
        "--corruption_type", ctype,
        "--k_values", *[str(k) for k in k_values],
        "--corruption_seeds", *[str(s) for s in cseeds],
        "--seeds", *[str(s) for s in seeds],
        "--key_pairs", *key_pairs,
        "--output", str(output_json),
    ]
    print(f"\n{'=' * 70}")
    print(f"[SWEEP] corruption_type={ctype}  →  {output_json}")
    print(f"{'=' * 70}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[FAIL] {ctype} sweep exited {result.returncode}")
        return None
    return output_json


# ---------------------------------------------------------------------------
# 2. Build per-corruption-type DataFrames
# ---------------------------------------------------------------------------
def best_across_methods(raw_dataset_key, k):
    """Return (best_mean, best_std, best_method) across clustering methods."""
    best_mean, best_std, best_method = float("nan"), float("nan"), ""
    for clust_method, kd in raw_dataset_key.items():
        vals = kd.get(k, [])
        if vals:
            m = float(np.mean(vals))
            if np.isnan(best_mean) or m > best_mean:
                best_mean = m
                best_std = float(np.std(vals))
                best_method = clust_method
    return best_mean, best_std, best_method


def build_corruption_table(raw, ctype):
    """Build the DataFrame for one corruption type.

    Columns: dataset, type, m, c, clean, k=v1, k=v2, …
    Each cell value is "mean ± std" (or "—") of the BEST clustering method.
    Also returns:
        cell_meta : dict[(dataset, key_name)][col] = (mean, std, best_method)
        k_cols    : list of column-label strings for the k values (sorted)
        rows      : ordered list of (dataset, key_name) tuples
    """
    # discover k values and (dataset, key) pairs
    k_set = set()
    for ds in raw:
        for key in raw[ds]:
            for cm in raw[ds][key]:
                k_set.update(raw[ds][key][cm].keys())
    has_clean = None in k_set
    k_sorted = [k for k in sorted(k_set, key=_k_sort_key) if k is not None]
    k_cols = [f"k={k}" for k in k_sorted]

    # Sort rows: by dataset name, then NMF → OT-SAE → N-SAE within each dataset
    rows = []
    for ds in sorted(raw.keys()):
        keys_in_ds = list(raw[ds].keys())
        keys_in_ds.sort(key=lambda k: _row_sort_key(parse_model_key(k)))
        for key in keys_in_ds:
            rows.append((ds, key))

    cell_meta = defaultdict(dict)
    records = []
    for (ds, key) in rows:
        type_label, m, c = parse_model_key(key)
        rec = {
            "dataset": ds,
            "type": type_label,
            "m": m,
            "c": "—" if c is None else c,
        }

        if has_clean:
            mean, std, bm = best_across_methods(raw[ds][key], None)
            if not np.isnan(mean):
                rec["clean"] = f"{mean:.4f} ± {std:.4f}"
                cell_meta[(ds, key)]["clean"] = (mean, std, bm)
            else:
                rec["clean"] = "—"

        for kval, kcol in zip(k_sorted, k_cols):
            mean, std, bm = best_across_methods(raw[ds][key], kval)
            if not np.isnan(mean):
                rec[kcol] = f"{mean:.4f} ± {std:.4f}"
                cell_meta[(ds, key)][kcol] = (mean, std, bm)
            else:
                rec[kcol] = "—"

        records.append(rec)

    df = pd.DataFrame(records)
    return df, cell_meta, k_cols, has_clean, rows


# ---------------------------------------------------------------------------
# 3. Render per-corruption-type PNG
# ---------------------------------------------------------------------------
def render_corruption_png(rows, cell_meta, k_cols, has_clean, ctype, output_path):
    """Render one PNG with a titled subtable for each dataset."""
    if not rows:
        print(f"[render] no rows for {ctype}, skipping")
        return

    # Preserve dataset order from the sorted `rows` list.
    datasets = []
    rows_by_dataset = defaultdict(list)
    for ds, key in rows:
        if ds not in rows_by_dataset:
            datasets.append(ds)
        rows_by_dataset[ds].append(key)

    data_cols = (["clean"] if has_clean else []) + k_cols
    n_data_cols = len(data_cols)

    # ----- best per (dataset, data_col) for highlighting -----
    best_per = {}
    for (ds, key), col_dict in cell_meta.items():
        for col, (mean, _, _) in col_dict.items():
            if not np.isnan(mean):
                cur = best_per.get((ds, col), float("-inf"))
                if mean > cur:
                    best_per[(ds, col)] = mean

    # ----- figure sizing -----
    type_w = 1.15
    m_w = 0.55
    c_w = 1.15
    data_w = 1.75
    row_h = 0.44
    title_h = 0.55
    fig_w = type_w + m_w + c_w + n_data_cols * data_w + 0.6
    height_ratios = [
        title_h + row_h * (len(rows_by_dataset[ds]) + 1.0)
        for ds in datasets
    ]
    fig_h = 0.8 + sum(height_ratios) + 0.25 * max(len(datasets) - 1, 0)

    fig, axes = plt.subplots(
        nrows=len(datasets),
        ncols=1,
        figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": height_ratios},
    )
    if len(datasets) == 1:
        axes = [axes]
    fig.suptitle(
        f"Corruption: {ctype}  (best clustering accuracy)",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )

    col_labels = ["type", "m", "c"] + data_cols
    n_cols = len(col_labels)
    col_widths = [type_w, m_w, c_w] + [data_w] * n_data_cols
    total_w = sum(col_widths)

    ds_colour_map = {
        ds: _DS_COLOURS[i % len(_DS_COLOURS)]
        for i, ds in enumerate(datasets)
    }

    for ax, ds in zip(axes, datasets):
        ax.axis("off")
        ax.set_title(
            f"Dataset: {ds}",
            loc="left",
            fontsize=10.5,
            fontweight="bold",
            pad=8,
        )

        cell_text = []
        for key in rows_by_dataset[ds]:
            type_label, m_val, c_val = parse_model_key(key)
            c_str = "—" if c_val is None else c_val
            row_text = [type_label, str(m_val), c_str]
            for col in data_cols:
                meta = cell_meta.get((ds, key), {}).get(col)
                if meta is None:
                    row_text.append("—")
                else:
                    mean, std, _ = meta
                    if std == 0.0 or np.isnan(std):
                        row_text.append(f"{mean:.4f}")
                    else:
                        row_text.append(f"{mean:.4f} ± {std:.4f}")
            cell_text.append(row_text)

        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.45)

        # Header styling
        for j in range(n_cols):
            cell = table[0, j]
            cell.set_width(col_widths[j] / total_w)
            cell.set_facecolor(_HEADER_COLOUR)
            cell.set_text_props(color="white", fontweight="bold", fontsize=8.5)

        gc = ds_colour_map[ds]
        for ri, key in enumerate(rows_by_dataset[ds]):
            for j in range(n_cols):
                table[ri + 1, j].set_width(col_widths[j] / total_w)

            type_label = parse_model_key(key)[0]
            type_colour = _TYPE_COLOURS.get(type_label, gc)

            # type / m / c columns (faint dataset tint, slight type accent)
            table[ri + 1, 0].set_facecolor(type_colour)
            table[ri + 1, 0].set_text_props(fontsize=8.5, fontweight="bold")
            table[ri + 1, 1].set_facecolor(gc)
            table[ri + 1, 1].set_text_props(fontsize=8.5)
            table[ri + 1, 2].set_facecolor(gc)
            table[ri + 1, 2].set_text_props(fontsize=8)

            # data columns: highlight best per (dataset, col)
            for ki, col in enumerate(data_cols):
                cell = table[ri + 1, 3 + ki]
                meta = cell_meta.get((ds, key), {}).get(col)
                if meta is not None and not np.isnan(meta[0]):
                    mean = meta[0]
                    best = best_per.get((ds, col), float("-inf"))
                    if abs(mean - best) < 1e-9:
                        cell.set_facecolor(_BEST_BG)
                        cell.set_text_props(fontweight="bold",
                                            color=_BEST_FG, fontsize=8.5)
                    else:
                        cell.set_facecolor(gc)
                        cell.set_text_props(fontsize=8.5)
                else:
                    cell.set_facecolor(gc)
                    cell.set_text_props(fontsize=8.5)

    fig.subplots_adjust(top=0.86, hspace=0.65)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved PNG: {output_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run all corruption sweeps and emit one table per corruption type."
    )
    parser.add_argument("--root", default="datasets/hsi_data",
                        help="Hyperspectral data root (passed to inner script)")
    parser.add_argument("--output_dir",
                        default=str(Path(__file__).resolve().parents[1] / "results" / "fixed_corruption_evaluation"),
                        help="Directory to write per-corruption JSONs and tables")
    parser.add_argument("--corruption_types", nargs="+",
                        default=DEFAULT_CORRUPTIONS)
    parser.add_argument("--k_values", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--corruption_seeds", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4])
    parser.add_argument("--key_pairs", nargs="+", required=False, default=None,
                        help="Pairs MODE KEY MODE KEY ... Required unless --skip_sweeps.")
    parser.add_argument("--skip_sweeps", action="store_true",
                        help="Skip running sweeps; rebuild tables from existing JSONs")
    args = parser.parse_args()

    if not args.skip_sweeps and not args.key_pairs:
        parser.error("--key_pairs is required unless --skip_sweeps is set")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wrapper = Path(__file__).resolve().parent / "corruption_sweep_wrapper.py"

    # ------------------------------------------------------------------
    # Run sweeps (sequentially)
    # ------------------------------------------------------------------
    json_paths = {}
    for ctype in args.corruption_types:
        json_path = output_dir / f"{ctype}.json"
        if args.skip_sweeps:
            json_paths[ctype] = json_path if json_path.exists() else None
            continue
        json_paths[ctype] = run_sweep_for_type(
            wrapper=wrapper, root=args.root, ctype=ctype,
            k_values=args.k_values, cseeds=args.corruption_seeds,
            seeds=args.seeds, key_pairs=args.key_pairs,
            output_json=json_path,
        )

    # ------------------------------------------------------------------
    # Build & render one table per corruption type
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("Building per-corruption tables")
    print(f"{'=' * 70}", flush=True)

    for ctype, jp in json_paths.items():
        if jp is None or not Path(jp).exists():
            print(f"  [skip] {ctype}: no JSON")
            continue
        with open(jp) as f:
            data = json.load(f)
        if not data.get("runs"):
            print(f"  [skip] {ctype}: empty runs")
            continue
        raw = parse_from_runs(data)

        df, cell_meta, k_cols, has_clean, rows = build_corruption_table(raw, ctype)

        csv_path = output_dir / f"table_{ctype}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  [{ctype}] CSV: {csv_path}  shape={df.shape}")

        png_path = output_dir / f"table_{ctype}.png"
        render_corruption_png(rows, cell_meta, k_cols, has_clean, ctype, png_path)

    print(f"\nDone. Outputs in {output_dir}/")


if __name__ == "__main__":
    main()
