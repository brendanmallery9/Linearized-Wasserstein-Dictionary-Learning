"""
Replay the same dataset iteration used by embed_document_from_hf.py
and save the raw text for every doc that was actually embedded.

Strategy:
  1. Look at existing .pt files to know which doc_indices were saved.
  2. Iterate the dataset with the SAME seed / streaming / shuffle / buffer
     settings so doc_index ↔ example mapping is identical.
  3. For each doc_index that has a .pt file, optionally verify token counts
     match, then write the raw text to disk.

Usage (mirror the flags you used for embedding):
  python pile100k_get_text.py \
      --pt_dir   datasets/pile-100k/activations/pile \
      --out_dir  datasets/pile-100k/raw_data/pile \
      --dataset  jannikbrinkmann/pile-100k \
      --seed     42 \
      --streaming \
      --shuffle \
      --shuffle_buffer 10000

  If you did NOT pass --streaming or --shuffle when embedding, omit them here too.
"""

import os
import re
import json
import argparse
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

# Same constants as the embedding script
MIN_DOC_TOKS = 200
MAX_DOC_TOKS = 2000


def discover_pt_files(pt_dir: Path):
    """
    Scan the .pt directory and return:
      - a set of doc_indices that were saved
      - the max doc_index seen (so we know when to stop iterating)
      - a dict mapping doc_index -> pt filename (for optional verification)
    """
    pattern = re.compile(r"doc_(\d+)_L\d+_\w+\.pt")
    index_to_file = {}
    for f in pt_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            idx = int(m.group(1))
            index_to_file[idx] = f
    if not index_to_file:
        raise FileNotFoundError(f"No doc_*_L*_*.pt files found in {pt_dir}")
    return index_to_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_dir", type=str, required=True,
                        help="Directory containing the .pt activation files")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to write raw text files into")
    parser.add_argument("--dataset", type=str, default="jannikbrinkmann/pile-100k")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-410m-deduped",
                        help="Tokenizer model (only used for optional token-count verification)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--shuffle_buffer", type=int, default=10000)
    parser.add_argument("--verify_tokens", action="store_true",
                        help="Load tokenizer and verify token counts match .pt files. "
                             "Slower but confirms ordering is correct.")
    parser.add_argument("--n_docs", type=int, default=-1,
                        help="Must match the value used during embedding (-1 = all)")
    args = parser.parse_args()

    pt_dir = Path(args.pt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: discover which doc_indices we need
    index_to_pt = discover_pt_files(pt_dir)
    needed = set(index_to_pt.keys())
    max_idx = max(needed)
    print(f"Found {len(needed)} .pt files, doc_indices range "
          f"[{min(needed)}, {max_idx}]")

    # Step 2: load dataset with identical settings
    print(f"Loading dataset: {args.dataset} (split={args.split}, "
          f"streaming={args.streaming}, shuffle={args.shuffle})")
    ds = load_dataset(args.dataset, split=args.split, streaming=args.streaming)

    if args.shuffle:
        if args.streaming:
            ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
            print(f"Streaming shuffle (buffer={args.shuffle_buffer}, seed={args.seed})")
        else:
            ds = ds.shuffle(seed=args.seed)
            print(f"Full shuffle (seed={args.seed})")

    # Optional tokenizer for verification
    tok = None
    if args.verify_tokens:
        tok = AutoTokenizer.from_pretrained(args.model_name)
        print("Token verification enabled")

    # Step 3: iterate and save
    saved = skipped = already = 0
    for doc_index, example in enumerate(ds):
        if args.n_docs > 0 and doc_index >= args.n_docs:
            break
        # Stop early if we've passed all needed indices (non-streaming only)
        if doc_index > max_idx and not args.streaming:
            break

        if doc_index not in needed:
            skipped += 1
            continue

        text = example["text"]
        out_path = out_dir / f"doc_{doc_index:05d}.txt"

        if out_path.exists():
            already += 1
            continue

        # Optional: verify token count matches the .pt file
        if tok is not None:
            input_ids = tok(text, return_tensors="pt",
                            add_special_tokens=False)["input_ids"][0]
            T = input_ids.shape[0]
            pt_data = torch.load(index_to_pt[doc_index], map_location="cpu",
                                 weights_only=True)
            pt_T = pt_data["tokens"].shape[0]
            if T != pt_T:
                print(f"WARNING doc {doc_index}: tokenized length {T} != "
                      f".pt token length {pt_T}  — ordering may be wrong!")

        out_path.write_text(text, encoding="utf-8")
        saved += 1
        if saved % 200 == 0 or saved <= 5:
            print(f"[{doc_index}] Saved {out_path.name} "
                  f"(saved={saved}, skipped={skipped}, already={already})")

    remaining = needed - {i for i in range(doc_index + 1) if i in needed}
    print(f"\nDone. saved={saved}, skipped_not_needed={skipped}, "
          f"already_existed={already}")
    if remaining:
        print(f"WARNING: {len(remaining)} doc_indices were NOT reached "
              f"during iteration. This likely means the dataset flags "
              f"don't match what was used during embedding.")
    else:
        print("All doc_indices matched — text files are complete.")


if __name__ == "__main__":
    main()