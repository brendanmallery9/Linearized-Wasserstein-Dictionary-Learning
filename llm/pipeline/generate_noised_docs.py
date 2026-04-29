#!/usr/bin/env python3
"""
Generate corrupted ("noised") copies of a base text document.

Sweeps over corruption types and severity levels, writing one .txt file per
(type, severity, seed) to::

    <out_root>/<severity>_<type>/seed_<i>.txt

The resulting layout matches what ``embed_documents_from_dir.py`` /
``multi_call_embed_from_dir.py`` consume downstream.

Default corruption types: global_swap_words, delete_words, permute_letters,
duplicate_words.
Default severities: 1, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275.

Example:
    python generate_noised_docs.py \\
        --base-text datasets/noised_luther/txt/base/luther.txt \\
        --out-root  datasets/noised_luther/txt/noised_docs \\
        --n-seeds   400
"""

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Callable, List

# ---------------------------------------------------------------------------
# Tokenization (preserve punctuation/whitespace)
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def split_preserve(text: str) -> List[str]:
    """Split text into word / non-word tokens whose concatenation is the original text."""
    tokens: List[str] = []
    i = 0
    for m in _WORD_RE.finditer(text):
        if m.start() > i:
            tokens.append(text[i:m.start()])
        tokens.append(text[m.start():m.end()])
        i = m.end()
    if i < len(text):
        tokens.append(text[i:])
    return tokens


def word_token_indices(tokens: List[str]) -> List[int]:
    return [i for i, t in enumerate(tokens) if _WORD_RE.fullmatch(t) is not None]


# ---------------------------------------------------------------------------
# Noise operators
# ---------------------------------------------------------------------------
def global_swap_words(tokens: List[str], no_swaps: int, rng: random.Random) -> List[str]:
    out = tokens[:]
    widx = word_token_indices(out)
    if len(widx) < 2 or no_swaps <= 0:
        return out
    k = min(no_swaps, len(widx) // 2)
    chosen = rng.sample(widx, 2 * k)
    for a, b in zip(chosen[0::2], chosen[1::2]):
        out[a], out[b] = out[b], out[a]
    return out


def permute_letters_within_words(tokens: List[str], no_perms: int, rng: random.Random) -> List[str]:
    out = tokens[:]
    eligible = [i for i in word_token_indices(out) if len(out[i]) >= 3]
    if not eligible or no_perms <= 0:
        return out
    k = min(no_perms, len(eligible))
    for ti in rng.sample(eligible, k):
        letters = list(out[ti])
        rng.shuffle(letters)
        out[ti] = "".join(letters)
    return out


def delete_word_spans(tokens: List[str], no_spans: int, span_len: int, rng: random.Random) -> List[str]:
    out = tokens[:]
    widx = word_token_indices(out)
    n = len(widx)
    span_len = int(span_len)
    if n == 0 or no_spans <= 0 or span_len <= 0 or n < span_len:
        return out

    possible_starts = list(range(0, n - span_len + 1))
    rng.shuffle(possible_starts)

    chosen_starts: List[int] = []
    occupied = [False] * n
    for s in possible_starts:
        if len(chosen_starts) >= no_spans:
            break
        if any(occupied[s:s + span_len]):
            continue
        chosen_starts.append(s)
        for t in range(s, s + span_len):
            occupied[t] = True

    for s in chosen_starts:
        for wp in range(s, s + span_len):
            out[widx[wp]] = ""
    return out


def duplicate_words(tokens: List[str], no_dups: int, rng: random.Random) -> List[str]:
    out = tokens[:]
    widx = word_token_indices(out)
    if not widx or no_dups <= 0:
        return out

    k = min(no_dups, len(widx))
    chosen = sorted(rng.sample(widx, k))

    offset = 0
    for ti in chosen:
        ti2 = ti + offset
        word = out[ti2]
        needs_trailing_space = True
        if ti2 + 1 < len(out) and out[ti2 + 1].startswith((" ", "\n", "\t")):
            needs_trailing_space = False
        insert_tokens = [" ", word, (" " if needs_trailing_space else "")]
        out[ti2 + 1:ti2 + 1] = insert_tokens
        offset += len(insert_tokens)

    return [re.sub(r" {2,}", " ", t) for t in out]


# ---------------------------------------------------------------------------
# Corruption registry
# ---------------------------------------------------------------------------
# Each entry maps a folder-name tag -> a function (base_tokens, severity, rng) -> tokens.
# Folder names match what corrupted_text_analysis*.ipynb expects.
DEFAULT_DELETE_SPAN_LEN = 1


def make_corruptions(delete_span_len: int = DEFAULT_DELETE_SPAN_LEN) -> dict[str, Callable]:
    return {
        "global_swap_words": lambda toks, n, rng: global_swap_words(toks, n, rng),
        "delete_words":      lambda toks, n, rng: delete_word_spans(toks, n, delete_span_len, rng),
        "permute_letters":   lambda toks, n, rng: permute_letters_within_words(toks, n, rng),
        "duplicate_words":   lambda toks, n, rng: duplicate_words(toks, n, rng),
    }


DEFAULT_SEVERITIES = [1, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def write_variants(
    folder: Path,
    base_tokens: List[str],
    op: Callable,
    severity: int,
    n_seeds: int,
    overwrite: bool,
) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    written = 0
    for seed in range(n_seeds):
        out_path = folder / f"seed_{seed}.txt"
        if out_path.exists() and not overwrite:
            continue
        rng = random.Random(seed)
        toks = op(base_tokens, severity, rng)
        out_path.write_text("".join(toks), encoding="utf-8")
        written += 1
    return written


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--base-text", type=Path,
        default=Path("datasets/noised_luther/txt/base/luther.txt"),
        help="Path to the base .txt file to corrupt (default: %(default)s)",
    )
    p.add_argument(
        "--out-root", type=Path,
        default=Path("datasets/noised_luther/txt/noised_docs"),
        help="Output root for noised docs (default: %(default)s)",
    )
    p.add_argument(
        "--types", nargs="+",
        default=["global_swap_words", "delete_words", "permute_letters", "duplicate_words"],
        help="Corruption types to run (default: all four)",
    )
    p.add_argument(
        "--severities", type=int, nargs="+",
        default=DEFAULT_SEVERITIES,
        help=f"Severity levels (default: {DEFAULT_SEVERITIES})",
    )
    p.add_argument(
        "--n-seeds", type=int, default=400,
        help="Number of seeded variants per (type, severity) (default: %(default)s)",
    )
    p.add_argument(
        "--delete-span-len", type=int, default=DEFAULT_DELETE_SPAN_LEN,
        help="Span length for delete_words (default: %(default)s)",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing seed_*.txt files (default: skip existing)",
    )
    args = p.parse_args()

    if not args.base_text.is_file():
        sys.exit(f"Base text not found: {args.base_text}")

    base_text = args.base_text.read_text(encoding="utf-8")
    base_tokens = split_preserve(base_text)
    print(f"Loaded base text: {args.base_text} ({len(base_text)} chars, {len(base_tokens)} tokens)")

    corruptions = make_corruptions(delete_span_len=args.delete_span_len)
    unknown = [t for t in args.types if t not in corruptions]
    if unknown:
        sys.exit(f"Unknown corruption types: {unknown}. Choose from {sorted(corruptions)}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    total_written = 0
    for type_name in args.types:
        op = corruptions[type_name]
        for severity in args.severities:
            folder = args.out_root / f"{severity}_{type_name}"
            n_written = write_variants(
                folder, base_tokens, op, severity, args.n_seeds, args.overwrite,
            )
            total_written += n_written
            print(f"  {folder.name}: wrote {n_written}/{args.n_seeds} files")

    print(f"\nDone. Wrote {total_written} files under {args.out_root}")


if __name__ == "__main__":
    main()
