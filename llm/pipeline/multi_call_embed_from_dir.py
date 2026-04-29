#!/usr/bin/env python3
"""
Batch-embed documents across multiple models and subdirectories.

Calls embed_documents_from_dir.py for each (model, directory) pair.

Usage:
nohup python -u multi_call_embed_from_dir.py >& embed.log &
"""

import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EMBED_SCRIPT = SCRIPT_DIR / "embed_documents_from_dir.py"
REPO_ROOT = SCRIPT_DIR.parents[1]

ROOT_DIR = REPO_ROOT / "datasets" / "noised_luther" / "txt" / "noised_docs"
PARENT_DIR = REPO_ROOT / "datasets" / "noised_luther" / "activations" / "noised_activations"

MODELS = [
    # "EleutherAI/pythia-70m-deduped",
    # "EleutherAI/pythia-160m-deduped",
    "EleutherAI/pythia-410m-deduped",
]


def slugify_model(name: str) -> str:
    """Filesystem-friendly model name: 'EleutherAI/pythia-70m-deduped' -> 'EleutherAI__pythia-70m-deduped'"""
    return name.replace("/", "__")


def gather_input_dirs(root: Path) -> list[Path]:
    """
    If root itself contains .txt files, return [root].
    Otherwise return its immediate subdirectories (e.g. per-corruption dirs).
    """
    if any(root.glob("*.txt")):
        return [root]
    return sorted(d for d in root.iterdir() if d.is_dir())


def main():
    PARENT_DIR.mkdir(parents=True, exist_ok=True)

    input_dirs = gather_input_dirs(ROOT_DIR)
    if not input_dirs:
        print(f"No .txt files or subdirectories found under {ROOT_DIR}")
        return

    for model_name in MODELS:
        model_dir = PARENT_DIR / slugify_model(model_name)
        model_dir.mkdir(parents=True, exist_ok=True)

        for p in input_dirs:
            out_dir = model_dir / p.name
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n=== model={model_name}  {p} -> {out_dir} ===")

            cmd = [
                "python", str(EMBED_SCRIPT),
                "--data-dir", str(p),
                "--out-dir",  str(out_dir),
                "--model",    model_name,
            ]

            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
