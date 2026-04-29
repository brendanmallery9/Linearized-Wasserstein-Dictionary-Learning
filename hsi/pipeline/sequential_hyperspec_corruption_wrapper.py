#!/usr/bin/env python3
"""
Sequential wrapper that calls the multitrial clustering script across a grid of
corruption seeds x corruption magnitudes (k), then merges all per-run
JSON outputs into one consolidated file for downstream visualization.

Now supports mixed SAE modes (transport_maps / linear) via --key_pairs.

Speed optimizations (v2):
  - Baseline (clean) run is done once per mode; results are reused for every
    corruption seed via caching in the inner script.
  - All corruption seeds for a given (k, mode) are processed in a **single**
    subprocess invocation via --corruption_seeds, so data loading and clean-OT
    computation happen only once.

Usage example:
nohup python -u hsi/pipeline/sequential_hyperspec_corruption_wrapper.py \
    --script hsi/analysis/unsupervised_hyperspectral_clustering_multitrial.py \
    --root datasets/hsi_data \
    --corruption_type drop_contiguous \
    --k_values .1 .2 .3 .4 .5  \
    --corruption_seeds 0 1 2 3 4 \
    --seeds 0 1 2 3 4 \
    --key_pairs \
        transport_maps JUMPRELUAE_15_1e-1_mon \
        transport_maps JUMPRELUAE_15_5e-1_mon \
        transport_maps JUMPRELUAE_10_5e-1_mon \
        transport_maps JUMPRELUAE_10_1e-2_mon \
        transport_maps JUMPRELUAE_17_1e-5_mon \
        transport_maps JUMPRELUAE_17_1e-3_mon \
        transport_maps JUMPRELUAE_7_1e-5_mon \
        transport_maps JUMPRELUAE_7_1e-3_mon \
        linear JUMPRELUAE_15_1e-5_nonneg \
        linear JUMPRELUAE_15_1e-3_nonneg \
        linear JUMPRELUAE_10_1e-3_nonneg \
        linear JUMPRELUAE_10_1e-4_nonneg \
        linear JUMPRELUAE_17_1e-3_nonneg \
        linear JUMPRELUAE_17_1e-5_nonneg \
        linear JUMPRELUAE_7_1e-3_nonneg \
        linear JUMPRELUAE_7_1e-5_nonneg \
    --output drop_contig.json \
>& drop_contig.log &

"""

import argparse
import json
import subprocess
import sys
import tempfile
import os
import numpy as np
from pathlib import Path
from collections import defaultdict


def run_one(script, root, corruption_type, k, corruption_seeds, seeds,
            sae_mode="transport_maps", keys=None, batch_size=1024, extra_args=None):
    """Run the inner script once and return its JSON output.

    If k is None, no corruption is applied (baseline run).
    *corruption_seeds* is a list of ints - the inner script will loop over
    them in a single process, sharing data loading and clean OT maps.
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, dir=".") as tmp:
        tmp_json = tmp.name
    tmp_png = tmp_json.replace(".json", ".png")

    cmd = [
        sys.executable, "-u", script,
        "--root", str(root),
        "--seeds", *[str(s) for s in seeds],
        "--batch_size", str(batch_size),
        "--sae_mode", sae_mode,
        "--output_json", tmp_json,
        "--output_png", tmp_png,
    ]
    # Only add corruption args if k is not None
    if k is not None:
        cmd += ["--corruption", corruption_type, str(k),
                "--corruption_seeds", *[str(cs) for cs in corruption_seeds]]

    if keys:
        cmd += ["--keys", *keys]
    if extra_args:
        cmd += extra_args

    tag = (f"sae_mode={sae_mode}, k={k}, cseeds={corruption_seeds}"
           if k is not None
           else f"sae_mode={sae_mode}, baseline (no corruption)")
    print(f"\n[START] {tag}", flush=True)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[FAIL]  {tag} (exit code {result.returncode})", flush=True)
        return None

    try:
        with open(tmp_json) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[FAIL]  {tag}: could not read output JSON: {e}", flush=True)
        data = None
    finally:
        for p in (tmp_json, tmp_png):
            if os.path.exists(p):
                os.remove(p)

    if data is not None:
        print(f"[DONE]  {tag}", flush=True)

    return data


def _unpack_inner_output(raw_output, corruption_type, k, sae_mode, corruption_seeds):
    """Unpack inner script output into a list of per-corruption-seed run dicts.

    The inner script may return either:
      (a) Single-seed format (backward compat): top-level aggregated_sae, etc.
      (b) Multi-seed format: per_corruption_seed_outputs dict keyed by cseed.

    Returns a list of run dicts, each with corruption_type, k, corruption_seed,
    sae_mode, aggregated_sae, aggregated_nmf, per_seed_results.
    """
    runs = []

    if "per_corruption_seed_outputs" in raw_output:
        # Multi-seed format
        for cseed_str, block in raw_output["per_corruption_seed_outputs"].items():
            cseed = int(cseed_str) if cseed_str != "None" else None
            run = {
                "corruption_type": corruption_type if k is not None else "none",
                "k": k,
                "corruption_seed": cseed,
                "sae_mode": sae_mode,
                "aggregated_sae": block.get("aggregated_sae", {}),
                "aggregated_nmf": block.get("aggregated_nmf", {}),
                "per_seed_results": block.get("per_seed_results", []),
            }
            runs.append(run)
    else:
        # Single-seed format (baseline or old-style single cseed)
        cseed = raw_output.get("config", {}).get("corruption_seed", 0)
        run = {
            "corruption_type": corruption_type if k is not None else "none",
            "k": k,
            "corruption_seed": cseed if k is not None else None,
            "sae_mode": sae_mode,
            "aggregated_sae": raw_output.get("aggregated_sae", {}),
            "aggregated_nmf": raw_output.get("aggregated_nmf", {}),
            "per_seed_results": raw_output.get("per_seed_results", []),
        }
        runs.append(run)

    return runs


def build_summary(runs):
    """
    Aggregate metrics across corruption seeds for each (dataset|key, k) pair.

    The inner script's aggregated_sae / aggregated_nmf values are *lists* of
    dicts (one per clustering method), each with keys like:
        {"method": ..., "embedding": ..., "accuracy_mean": ..., "accuracy_std": ...}

    We collect the numeric metrics across corruption seeds and compute
    mean/std over them.
    """
    # summary_raw[method_key][k][metric] = [values across corruption seeds]
    summary_raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for run in runs:
        if run is None:
            continue
        k = run["k"]
        sae_mode = run.get("sae_mode", "transport_maps")

        # Process both SAE and NMF aggregated results
        for section in ("aggregated_sae", "aggregated_nmf"):
            for method_key, agg_list in run.get(section, {}).items():
                # Prefix with sae_mode so transport_maps and linear results
                # are tracked separately
                prefixed_key = f"{sae_mode}|{method_key}"
                for entry in agg_list:
                    for metric, value in entry.items():
                        # Skip non-numeric fields and std/var (we recompute)
                        if not isinstance(value, (int, float)):
                            continue
                        if metric.endswith("_std") or metric.endswith("_var"):
                            continue
                        summary_raw[prefixed_key][k][metric].append(value)

    # Compute mean/std across corruption seeds
    summary = {}
    for method_key, k_dict in summary_raw.items():
        summary[method_key] = {}
        for k, metric_dict in k_dict.items():
            summary[method_key][str(k)] = {}
            for metric, vals in metric_dict.items():
                arr = np.array(vals, dtype=float)
                summary[method_key][str(k)][metric] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "n": len(vals),
                    "per_corruption_seed": [float(v) for v in vals],
                }
    return summary


def _sort_k(k_str):
    return float("inf") if k_str == "None" else float(k_str)


def main():

    parser = argparse.ArgumentParser(
        description="Sweep corruption seeds x k values (sequential)"
    )
    parser.add_argument("--script", default=None,
                        help="Path to the multitrial clustering script")
    parser.add_argument("--root", required=True,
                        help="Root directory (passed through to inner script)")
    parser.add_argument("--corruption_type", required=True,
                        help="Corruption type (e.g. drop_random, shift_global)")
    parser.add_argument("--k_values", type=float, nargs="+", required=True,
                        help="List of corruption magnitudes k to sweep")
    parser.add_argument("--corruption_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Corruption RNG seeds to sweep (default: 0 1 2 3 4)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="SAE training seeds (passed through to inner script)")
    parser.add_argument("--key_pairs", type=str, nargs="+", default=None,
                        help="Pairs of (sae_mode, key), e.g.: "
                             "transport_maps JUMPRELUAE_15_1e-1_mon "
                             "linear JUMPRELUAE_15_1e-1_nonneg")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--output", default="sweep_results.json",
                        help="Path for consolidated output JSON")
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER, default=None,
                        help="Extra args forwarded to inner script")
    args = parser.parse_args()
    if args.script is None:
        args.script = str(Path(__file__).resolve().parents[1] / "analysis" /
                          "unsupervised_hyperspectral_clustering_multitrial.py")
    else:
        script_path = Path(args.script)
        if not script_path.is_absolute() and not script_path.exists():
            moved_script = (Path(__file__).resolve().parents[1] / "analysis" /
                            args.script)
            if moved_script.exists():
                args.script = str(moved_script)

    # ------------------------------------------------------------------
    # Parse key_pairs into grouped dict: {sae_mode: [keys]}
    # ------------------------------------------------------------------
    mode_keys = defaultdict(list)
    if args.key_pairs:
        if len(args.key_pairs) % 2 != 0:
            parser.error("--key_pairs requires pairs: MODE KEY MODE KEY ...")
        for i in range(0, len(args.key_pairs), 2):
            mode = args.key_pairs[i]
            key = args.key_pairs[i + 1]
            if mode == "potentials":
                mode = "transport_maps"
            if mode not in ("linear", "transport_maps"):
                parser.error(f"Invalid sae_mode '{mode}'. Must be 'linear' or 'transport_maps'.")
            mode_keys[mode].append(key)
    else:
        parser.error("--key_pairs is required.")

    # ------------------------------------------------------------------
    # Build job list: one job per (k, mode) - all cseeds are batched
    # Old: n_k * n_cseeds * n_modes jobs
    # New: (1 + n_k) * n_modes jobs  (1 baseline + n_k corrupted, each
    #       handling all cseeds internally)
    # ------------------------------------------------------------------
    jobs = []
    # Baseline (clean) run first - no corruption, one per mode
    for mode, keys in mode_keys.items():
        jobs.append((None, [], mode, keys))
    # Corrupted runs: one per (k, mode), batching all corruption seeds
    for k in args.k_values:
        for mode, keys in mode_keys.items():
            jobs.append((k, args.corruption_seeds, mode, keys))

    # Count equivalent old-style jobs for reporting
    old_total = len(mode_keys) + len(args.k_values) * len(args.corruption_seeds) * len(mode_keys)
    total = len(jobs)
    print(f"Sweep: {total} subprocess calls "
          f"(equivalent to {old_total} in the old per-cseed scheme)")
    print(f"  corruption_type={args.corruption_type}")
    print(f"  k_values={args.k_values}")
    print(f"  corruption_seeds={args.corruption_seeds}")
    print(f"  mode_keys:")
    for mode, keys in sorted(mode_keys.items()):
        print(f"    {mode}: {keys}")
    print(flush=True)

    all_runs = []
    failed = 0

    for i, (k, cseeds, mode, keys) in enumerate(jobs):
        print(f"\n{'=' * 70}")
        if k is None:
            print(f"[{i+1}/{total}] sae_mode={mode}, baseline (no corruption)")
        else:
            print(f"[{i+1}/{total}] sae_mode={mode}, k={k}, "
                  f"corruption_seeds={cseeds}")
        print(f"{'=' * 70}", flush=True)

        raw_output = run_one(
            script=args.script,
            root=args.root,
            corruption_type=args.corruption_type,
            k=k,
            corruption_seeds=cseeds,
            seeds=args.seeds,
            sae_mode=mode,
            keys=keys,
            batch_size=args.batch_size,
            extra_args=args.extra_args,
        )

        if raw_output is not None:
            unpacked = _unpack_inner_output(
                raw_output, args.corruption_type, k, mode, cseeds,
            )
            all_runs.extend(unpacked)
        else:
            # Count each cseed as a failure
            n_failed = max(len(cseeds), 1)
            failed += n_failed

    # Sort runs for deterministic output
    all_runs.sort(key=lambda r: (
        r.get("sae_mode", "transport_maps"),
        r["k"] is not None,
        r["k"] or 0,
        r["corruption_seed"] or 0,
    ))

    summary = build_summary(all_runs)

    output = {
        "sweep_config": {
            "script": args.script,
            "root": args.root,
            "corruption_type": args.corruption_type,
            "k_values": args.k_values,
            "corruption_seeds": args.corruption_seeds,
            "sae_seeds": args.seeds,
            "mode_keys": dict(mode_keys),
            "total_runs": old_total,
            "successful_runs": len(all_runs),
            "failed_runs": failed,
        },
        "runs": all_runs,
        "summary": summary,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Sweep complete: {len(all_runs)}/{old_total} runs succeeded, {failed} failed")
    print(f"Results written to {args.output}")
    print(f"{'=' * 70}")

    if summary:
        print(f"\n--- Summary (mean ± std across corruption seeds) ---")
        for method_key in sorted(summary):
            print(f"\n  {method_key}:")

            for k_str in sorted(summary[method_key], key=_sort_k):
                metrics = summary[method_key][k_str]
                parts = []
                for m, stats in sorted(metrics.items()):
                    parts.append(f"{m}={stats['mean']:.4f}±{stats['std']:.4f}")
                print(f"    k={k_str}: {', '.join(parts)}")


if __name__ == "__main__":
    main()
