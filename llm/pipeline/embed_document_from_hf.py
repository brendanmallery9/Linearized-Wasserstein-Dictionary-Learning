import os
import json
import argparse
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

# ---- fixed config ----
DATASET_NAME = "jannikbrinkmann/pile-100k"
SPLIT = "train"
MIN_DOC_TOKS = 100
MAX_DOC_TOKS = 2000
CTX_LEN = 256
DTYPE = torch.float16
# ----------------------

def hook_name(layer: int, location: str) -> str:
    if location == "residual":
        return f"blocks.{layer}.hook_resid_post"
    elif location == "mlpout":
        return f"blocks.{layer}.hook_mlp_out"
    elif location == "mlp":
        return f"blocks.{layer}.mlp.hook_post"
    elif location == "attn":
        return f"blocks.{layer}.hook_attn_out"
    elif location == "attn_concat":
        return f"blocks.{layer}.hook_attn_out"
    else:
        raise ValueError(f"Unsupported location: {location}")

def atomic_torch_save(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

def main():
    parser = argparse.ArgumentParser(description="Embed HuggingFace text dataset docs with a HookedTransformer model")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Root output directory; files saved to out_dir/subdir/")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-410m-deduped")
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--location", type=str, default="residual")
    parser.add_argument("--subdir", type=str, default="pile",
                        help="Subdirectory under out_dir to write files into (for compatibility with multi_dir_brenier_embedding.py)")
    parser.add_argument("--dataset", type=str, default="jannikbrinkmann/pile-100k",
                        help="HuggingFace dataset name (default: jannikbrinkmann/pile-100k). "
                             "For very large datasets use e.g. 'monology/pile-uncopyrighted' with --streaming.")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--n_docs", type=int, default=-1,
                        help="Max documents to process (-1 = all)")
    parser.add_argument("--streaming", action="store_true",
                        help="Load dataset in streaming mode. Required for large datasets "
                             "that don't fit in memory (e.g. full Pile).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU index for this worker (0-based) — only used when --device=cuda")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Total number of workers splitting the dataset (DOC_INDEX % num_gpus == gpu_id)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Torch device: 'cuda', 'cpu', 'mps', or 'auto' (default). When 'cuda', actually uses cuda:<gpu_id>.")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            DEVICE = f"cuda:{args.gpu_id}"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            DEVICE = "mps"
        else:
            DEVICE = "cpu"
    elif args.device == "cuda":
        DEVICE = f"cuda:{args.gpu_id}"
    else:
        DEVICE = args.device

    out_dir = Path(args.out_dir) / args.subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Pre-flight: skip work entirely if this shard is already complete -----
    # If --n_docs is bounded, we can compute exactly which DOC_INDEX values this
    # worker is responsible for and check whether all of them already exist on
    # disk. If so, exit before paying the model-load cost.
    if args.n_docs > 0:
        suffix = f"_L{args.layer}_{args.location}.pt"
        existing = {p.name for p in out_dir.glob(f"doc_*{suffix}")}
        my_indices = [i for i in range(args.n_docs) if i % args.num_gpus == args.gpu_id]
        my_filenames = [f"doc_{i:05d}{suffix}" for i in my_indices]
        missing = [n for n in my_filenames if n not in existing]
        if not missing:
            print(f"[worker {args.gpu_id}/{args.num_gpus}] All {len(my_filenames)} target docs "
                  f"already embedded under {out_dir}. Nothing to do.")
            return
        else:
            print(f"[worker {args.gpu_id}/{args.num_gpus}] {len(missing)}/{len(my_filenames)} "
                  f"docs still need embedding (will load model)")

    print(f"[worker {args.gpu_id}/{args.num_gpus}] Loading dataset: {args.dataset} (split={args.split}, streaming={args.streaming})")
    ds = load_dataset(args.dataset, split=args.split, streaming=args.streaming)

    if not args.streaming:
        total = len(ds)
        print(f"Dataset size: {total} documents")

    print(f"[worker {args.gpu_id}/{args.num_gpus}] Loading model: {args.model_name} (device={DEVICE})")
    model = HookedTransformer.from_pretrained(args.model_name, device=DEVICE)
    tok = AutoTokenizer.from_pretrained(args.model_name)
    hname = hook_name(args.layer, args.location)

    saved = skipped_len = skipped_exists = skipped_gpu = 0

    for DOC_INDEX, example in enumerate(ds):
        if args.n_docs > 0 and DOC_INDEX >= args.n_docs:
            break

        # Multi-GPU splitting: each GPU only processes its assigned docs
        if DOC_INDEX % args.num_gpus != args.gpu_id:
            skipped_gpu += 1
            continue

        out_path = out_dir / f"doc_{DOC_INDEX:05d}_L{args.layer}_{args.location}.pt"
        if out_path.exists():
            skipped_exists += 1
            continue

        text = example["text"]
        # Extract meta data (dict column from HF dataset)
        meta = example.get("meta", {})

        input_ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        T = input_ids.shape[0]
        if T < MIN_DOC_TOKS or T > MAX_DOC_TOKS:
            skipped_len += 1
            if skipped_len <= 5 or skipped_len % 500 == 0:
                print(f"[worker {args.gpu_id}] Skipping doc {DOC_INDEX} ({T} tokens, outside [{MIN_DOC_TOKS}, {MAX_DOC_TOKS}])")
            continue

        acts, toks = [], []
        for start in range(0, T, CTX_LEN):
            end = min(start + CTX_LEN, T)
            chunk = input_ids[start:end].unsqueeze(0).to(model.cfg.device)
            with torch.no_grad():
                _, cache = model.run_with_cache(chunk, names_filter=lambda n: n == hname)
            a = cache[hname][0].to(dtype=DTYPE).cpu()
            acts.append(a)
            toks.append(input_ids[start:end].cpu())

        acts = torch.cat(acts, dim=0)
        toks = torch.cat(toks, dim=0)
        assert toks.shape[0] == acts.shape[0] == T

        atomic_torch_save({
            "doc_index": DOC_INDEX,
            "tokens": toks,
            "acts": acts,
            "text": text,
            "meta": meta,
        }, out_path)
        saved += 1
        print(f"[worker {args.gpu_id}][{DOC_INDEX}] Saved: {out_path.name}  T={T}  acts={acts.shape}  "
              f"(saved={saved}, skipped_len={skipped_len}, already_done={skipped_exists})")

if __name__ == "__main__":
    main()

# Examples:
# Single GPU (default):
#   python embed_document_from_hf.py --out_dir /data/.../activations
#
# Multi-GPU (4 GPUs in parallel):
#   for i in 0 1 2 3; do
#     python embed_document_from_hf.py \
#       --out_dir /data/.../activations \
#       --gpu_id $i --num_gpus 4 &
#   done
#   wait