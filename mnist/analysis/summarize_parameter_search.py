"""
Summarize outputs from parameter_search.py.

Reads the search directory produced by the staged LISTA search and writes:
    - summary.json
    - summary.txt
    - final_trials.csv

It also prints a concise terminal summary with:
    - stage survival counts
    - best final trial by probe accuracy
    - best final trial by reconstruction
    - best balanced final trial
"""

import argparse
import csv
import json
from pathlib import Path


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_stage_results(search_dir):
    """Load all available stage result files in order."""
    stage_paths = sorted(search_dir.glob("stage*_results.json"))
    stages = []
    for path in stage_paths:
        name = path.stem
        stage_num = int(name.split("_")[0].replace("stage", ""))
        results = load_json(path)
        stages.append((stage_num, results))
    return stages


def rank_final_trials(final_results):
    """Annotate final trials with recon/probe ranks and balanced scores."""
    if not final_results:
        return []

    by_recon = sorted(final_results, key=lambda r: (r["test_recon_loss"], -r["probe_test_accuracy"]))
    by_probe = sorted(final_results, key=lambda r: (-r["probe_test_accuracy"], r["test_recon_loss"]))

    recon_rank = {row["name"]: i + 1 for i, row in enumerate(by_recon)}
    probe_rank = {row["name"]: i + 1 for i, row in enumerate(by_probe)}

    ranked = []
    for row in final_results:
        item = dict(row)
        item["recon_rank"] = recon_rank[row["name"]]
        item["probe_rank"] = probe_rank[row["name"]]
        item["balanced_rank"] = max(item["recon_rank"], item["probe_rank"])
        item["rank_sum"] = item["recon_rank"] + item["probe_rank"]
        ranked.append(item)

    ranked.sort(
        key=lambda r: (
            r["balanced_rank"],
            r["rank_sum"],
            -r["probe_test_accuracy"],
            r["test_recon_loss"],
        )
    )
    return ranked


def summarize_stages(stages):
    """Build a compact per-stage survival summary."""
    summary = []
    for stage_num, results in stages:
        success = [r for r in results if r.get("status") == "ok"]
        failed = [r for r in results if r.get("status") != "ok"]
        epochs_completed = sorted({r.get("epochs_completed") for r in success if "epochs_completed" in r})
        summary.append(
            {
                "stage": stage_num,
                "num_trials": len(results),
                "num_success": len(success),
                "num_failed": len(failed),
                "epochs_completed": epochs_completed,
            }
        )
    return summary


def write_csv(rows, path):
    """Write final ranked trials as CSV."""
    fieldnames = [
        "name",
        "trial_index",
        "eps",
        "sparsity_coeff",
        "epochs_completed",
        "test_recon_loss",
        "probe_test_accuracy",
        "test_mean_l1",
        "test_mean_active",
        "recon_rank",
        "probe_rank",
        "balanced_rank",
        "rank_sum",
        "elapsed_seconds",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_text_summary(path, stage_summary, best_probe, best_recon, best_balanced, ranked_final, top_k):
    """Write a human-readable plain-text report."""
    lines = []
    lines.append("LISTA Search Summary")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Stage survival:")
    for item in stage_summary:
        epochs_str = ", ".join(str(x) for x in item["epochs_completed"]) or "-"
        lines.append(
            f"  Stage {item['stage']}: "
            f"{item['num_success']}/{item['num_trials']} succeeded, "
            f"{item['num_failed']} failed, "
            f"epochs_completed={epochs_str}"
        )

    lines.append("")
    lines.append("Best trials:")
    if best_probe is not None:
        lines.append(
            f"  Best probe accuracy: {best_probe['name']}  "
            f"acc={best_probe['probe_test_accuracy']:.4f}  "
            f"recon={best_probe['test_recon_loss']:.6f}  "
            f"eps={best_probe['eps']:.6g}  c={best_probe['sparsity_coeff']:.6g}"
        )
    if best_recon is not None:
        lines.append(
            f"  Best reconstruction: {best_recon['name']}  "
            f"recon={best_recon['test_recon_loss']:.6f}  "
            f"acc={best_recon['probe_test_accuracy']:.4f}  "
            f"eps={best_recon['eps']:.6g}  c={best_recon['sparsity_coeff']:.6g}"
        )
    if best_balanced is not None:
        lines.append(
            f"  Best balanced: {best_balanced['name']}  "
            f"balanced_rank={best_balanced['balanced_rank']}  "
            f"recon_rank={best_balanced['recon_rank']}  "
            f"probe_rank={best_balanced['probe_rank']}  "
            f"eps={best_balanced['eps']:.6g}  c={best_balanced['sparsity_coeff']:.6g}"
        )

    lines.append("")
    lines.append(f"Top {min(top_k, len(ranked_final))} final trials:")
    for row in ranked_final[:top_k]:
        lines.append(
            f"  {row['name']}: "
            f"acc={row['probe_test_accuracy']:.4f}, "
            f"recon={row['test_recon_loss']:.6f}, "
            f"eps={row['eps']:.6g}, "
            f"c={row['sparsity_coeff']:.6g}, "
            f"balanced_rank={row['balanced_rank']}"
        )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def summarize_search(search_dir, top_k=10):
    """Load a search output directory and write aggregate summaries."""
    search_dir = Path(search_dir)
    final_path = search_dir / "final_results.json"
    if not final_path.exists():
        raise FileNotFoundError(f"Missing final results file: {final_path}")

    final_results = load_json(final_path)
    ranked_final = rank_final_trials(final_results)
    stages = load_stage_results(search_dir)
    stage_summary = summarize_stages(stages)

    best_probe = max(ranked_final, key=lambda r: (r["probe_test_accuracy"], -r["test_recon_loss"]), default=None)
    best_recon = min(ranked_final, key=lambda r: (r["test_recon_loss"], -r["probe_test_accuracy"]), default=None)
    best_balanced = ranked_final[0] if ranked_final else None

    summary = {
        "search_dir": str(search_dir),
        "num_final_trials": len(ranked_final),
        "stage_summary": stage_summary,
        "best_probe_accuracy": best_probe,
        "best_reconstruction": best_recon,
        "best_balanced": best_balanced,
        "top_final_trials": ranked_final[:top_k],
    }

    with open(search_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    write_text_summary(
        search_dir / "summary.txt",
        stage_summary=stage_summary,
        best_probe=best_probe,
        best_recon=best_recon,
        best_balanced=best_balanced,
        ranked_final=ranked_final,
        top_k=top_k,
    )
    write_csv(ranked_final, search_dir / "final_trials.csv")

    print(f"Search dir: {search_dir}")
    for item in stage_summary:
        print(
            f"Stage {item['stage']}: "
            f"{item['num_success']}/{item['num_trials']} succeeded, "
            f"{item['num_failed']} failed"
        )

    if best_probe is not None:
        print(
            f"Best probe: {best_probe['name']}  "
            f"acc={best_probe['probe_test_accuracy']:.4f}  "
            f"recon={best_probe['test_recon_loss']:.6f}  "
            f"eps={best_probe['eps']:.6g}  c={best_probe['sparsity_coeff']:.6g}"
        )
    if best_recon is not None:
        print(
            f"Best recon: {best_recon['name']}  "
            f"recon={best_recon['test_recon_loss']:.6f}  "
            f"acc={best_recon['probe_test_accuracy']:.4f}  "
            f"eps={best_recon['eps']:.6g}  c={best_recon['sparsity_coeff']:.6g}"
        )
    if best_balanced is not None:
        print(
            f"Best balanced: {best_balanced['name']}  "
            f"balanced_rank={best_balanced['balanced_rank']}  "
            f"recon_rank={best_balanced['recon_rank']}  "
            f"probe_rank={best_balanced['probe_rank']}"
        )

    print(f"Wrote {(search_dir / 'summary.json')}")
    print(f"Wrote {(search_dir / 'summary.txt')}")
    print(f"Wrote {(search_dir / 'final_trials.csv')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize LISTA search outputs.")
    parser.add_argument(
        "--search_dir",
        type=str,
        required=True,
        help="Output directory from parameter_search.py",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="How many final trials to include in the summary outputs.",
    )
    args = parser.parse_args()

    summarize_search(args.search_dir, top_k=args.top_k)
