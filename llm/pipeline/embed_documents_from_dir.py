#!/usr/bin/env python3
"""
Embed documents from a directory of .txt files using TransformerLens.

Extracts residual-stream (or other) activations from a HookedTransformer,
chunked to a fixed context length, and saves per-document .pt files.

Usage:
    python embed_documents_from_dir.py --data-dir datasets/noised_luther/txt/noised_docs \
                             --out-dir datasets/noised_luther/activations/noised_activations \
                             --model EleutherAI/pythia-410m-deduped

    python embed_documents_from_dir.py --data-dir datasets/noised_luther/txt/base \
                             --out-dir datasets/noised_luther/activations/base \
                             --model EleutherAI/pythia-410m-deduped

    python embed_documents_from_dir.py --data-dir data/texts --out-dir data/acts \
        --model EleutherAI/pythia-160m --layer 3 --location mlpout \
        --ctx-len 512 --min-tokens 100 --max-tokens 4000 --dtype float32
"""

import os
import argparse
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    model="EleutherAI/pythia-410m-deduped",
    layer=2,
    location="residual",
    ctx_len=256,
    min_tokens=300,
    max_tokens=2000,
    dtype="float16",
    device="cuda",
)

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

# ---------------------------------------------------------------------------
# Hook-name mapping
# ---------------------------------------------------------------------------
HOOK_TEMPLATES = {
    "residual": "blocks.{layer}.hook_resid_post",
    "mlpout":   "blocks.{layer}.hook_mlp_out",
    "mlp":      "blocks.{layer}.mlp.hook_post",
    "attn":     "blocks.{layer}.hook_attn_out",
}


def hook_name(layer: int, location: str) -> str:
    """Map a human-readable location tag to a TransformerLens hook name."""
    template = HOOK_TEMPLATES.get(location)
    if template is None:
        valid = ", ".join(sorted(HOOK_TEMPLATES))
        raise ValueError(
            f"Unknown location '{location}'. Choose from: {valid}"
        )
    return template.format(layer=layer)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def log_error(msg: str, log_path: str | None = None) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    line = f"[{ts}] {msg}"
    print(line)
    if log_path is not None:
        with open(log_path, "a") as f:
            f.write(line + "\n")


def iter_text_files(data_dir: str):
    """Yield (filepath, text) for every .txt file under *data_dir* (recursive, sorted)."""
    for fp in sorted(Path(data_dir).rglob("*.txt")):
        try:
            yield str(fp), fp.read_text(encoding="utf-8")
        except Exception as e:
            print(f"  [skip] Could not read {fp}: {e}")


def atomic_torch_save(obj, out_path: str) -> None:
    """Write a .pt file atomically so incomplete saves don't leave corrupt files."""
    tmp_path = out_path + ".tmp"
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def load_model(model_name: str, device: str):
    print(f"Loading model: {model_name}")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def embed_documents(
    data_dir: str,
    out_dir: str,
    model,
    tokenizer,
    layer: int,
    location: str,
    ctx_len: int,
    min_tokens: int,
    max_tokens: int,
    dtype: torch.dtype,
    device: str,
    max_docs: int = -1,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    error_log = os.path.join(out_dir, "embed_errors.log")

    hname = hook_name(layer, location)
    print(f"Hook: {hname}  |  ctx_len={ctx_len}  |  dtype={dtype}")

    docs = list(iter_text_files(data_dir))
    if max_docs > 0:
        docs = docs[:max_docs]
    if not docs:
        print(f"No .txt files found under {data_dir}")
        return

    skipped = saved = errored = 0

    for doc_idx, (fp, text) in enumerate(docs, 1):
        stem = Path(fp).stem
        out_path = os.path.join(out_dir, f"{stem}_L{layer}_{location}.pt")

        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            input_ids = tokenizer(
                text, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0]
            n_tokens = input_ids.shape[0]

            if not (min_tokens <= n_tokens <= max_tokens):
                print(f"  [{doc_idx}/{len(docs)}] Skip {stem} ({n_tokens} tokens, outside [{min_tokens}, {max_tokens}])")
                skipped += 1
                continue

            acts_chunks, tok_chunks = [], []
            for start in range(0, n_tokens, ctx_len):
                chunk = input_ids[start : start + ctx_len].unsqueeze(0).to(device)

                with torch.no_grad():
                    _, cache = model.run_with_cache(
                        chunk, names_filter=lambda n: n == hname
                    )

                acts_chunks.append(cache[hname][0].to(dtype=dtype).cpu())
                tok_chunks.append(input_ids[start : start + ctx_len].cpu())

            acts = torch.cat(acts_chunks, dim=0)
            toks = torch.cat(tok_chunks, dim=0)

            atomic_torch_save(
                {"doc_name": stem, "source_path": fp, "tokens": toks, "acts": acts},
                out_path,
            )
            saved += 1
            print(f"  [{doc_idx}/{len(docs)}] Saved {stem}  tokens={n_tokens}  acts={tuple(acts.shape)}")

        except Exception as e:
            errored += 1
            log_error(f"FAILED {fp}: {type(e).__name__}: {e}", log_path=error_log)

    print(f"Done — saved: {saved}, skipped/existed: {skipped}, errors: {errored}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-dir", required=True, help="Directory of .txt documents (or parent of subdirs if --multi-dir)")
    p.add_argument("--out-dir",  required=True, help="Directory for output .pt files (or parent if --multi-dir)")
    p.add_argument("--multi-dir", action="store_true",
                   help="Treat --data-dir as a parent of per-corruption subdirs; iterate each in-process so the model loads only once")
    p.add_argument("--model",    default=DEFAULTS["model"],      help="HuggingFace model name (default: %(default)s)")
    p.add_argument("--layer",    default=DEFAULTS["layer"],      type=int, help="Layer to extract from (default: %(default)s)")
    p.add_argument("--location", default=DEFAULTS["location"],   choices=sorted(HOOK_TEMPLATES), help="Activation site (default: %(default)s)")
    p.add_argument("--ctx-len",  default=DEFAULTS["ctx_len"],    type=int, help="Context-window chunk size (default: %(default)s)")
    p.add_argument("--min-tokens", default=DEFAULTS["min_tokens"], type=int, help="Skip docs shorter than this (default: %(default)s)")
    p.add_argument("--max-tokens", default=DEFAULTS["max_tokens"], type=int, help="Skip docs longer than this (default: %(default)s)")
    p.add_argument("--dtype",    default=DEFAULTS["dtype"],      choices=sorted(DTYPE_MAP), help="Storage dtype (default: %(default)s)")
    p.add_argument("--device",   default=DEFAULTS["device"],     help="Torch device (default: %(default)s)")
    p.add_argument("--max-docs", default=-1, type=int, help="If >0, embed at most this many .txt files per directory (default: all)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dtype = DTYPE_MAP[args.dtype]

    if args.multi_dir:
        parent_in = Path(args.data_dir)
        parent_out = Path(args.out_dir)
        subdirs = sorted(d for d in parent_in.iterdir() if d.is_dir())
        if not subdirs:
            print(f"No subdirectories found under {parent_in}")
            raise SystemExit(0)

        # Skip subdirs that are already fully embedded (avoid loading the model
        # if there's nothing to do).
        pending = []
        for sub in subdirs:
            n_in = sum(1 for _ in sub.glob("seed_*.txt"))
            if args.max_docs > 0:
                expected = min(n_in, args.max_docs)
            else:
                expected = n_in
            out_sub = parent_out / sub.name
            n_out = sum(1 for _ in out_sub.glob(f"seed_*_L{args.layer}_{args.location}.pt")) if out_sub.exists() else 0
            if expected > 0 and n_out >= expected:
                print(f"[skip] {sub.name}  (all {expected} already embedded)")
                continue
            pending.append(sub)

        if not pending:
            print("Nothing to embed.")
            raise SystemExit(0)

        model, tokenizer = load_model(args.model, args.device)
        for i, sub in enumerate(pending, 1):
            out_sub = parent_out / sub.name
            print(f"\n=== [{i}/{len(pending)}] {sub.name} -> {out_sub} ===")
            embed_documents(
                data_dir=str(sub),
                out_dir=str(out_sub),
                model=model,
                tokenizer=tokenizer,
                layer=args.layer,
                location=args.location,
                ctx_len=args.ctx_len,
                min_tokens=args.min_tokens,
                max_tokens=args.max_tokens,
                dtype=dtype,
                device=args.device,
                max_docs=args.max_docs,
            )
    else:
        model, tokenizer = load_model(args.model, args.device)
        embed_documents(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            model=model,
            tokenizer=tokenizer,
            layer=args.layer,
            location=args.location,
            ctx_len=args.ctx_len,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            dtype=dtype,
            device=args.device,
            max_docs=args.max_docs,
        )