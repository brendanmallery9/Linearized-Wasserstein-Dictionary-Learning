"""
doc_analysis_functions.py

Targeted interpretability analysis for SAE-encoded documents.

Three-step workflow
-------------------
1. find_docs_by_words   – find corpus documents that contain a set of words
2. top_features_for_docs – given those documents, surface the most-activated
                           SAE features (top-k or top-fraction-of-activation)
3. feature_clusters_for_features – map those features back to their cluster
                                   assignments from the notebook's K-Means run
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np



# ─────────────────────────────────────────────────────────────────────────────
# 0.  Simple category definitions for targeted interpretability queries
# ─────────────────────────────────────────────────────────────────────────────
RUSSIAN_MARKERS = [
    "ж", "я", "ю", "щ", "ы", "э", "ъ",
    "Ж", "Я", "Ю", "Щ", "Ы", "Э", "Ъ",
]

LATEX_MATH_MARKERS = [
    r"\mathbb",
    r"\frac",
    r"\sum",
    r"\int",
    r"\partial",
    r"\nabla",
    r"\alpha",
    r"\beta",
    r"\gamma",
    r"\theta",
    r"\lambda",
    r"\infty",
    r"\left",
    r"\right",
    r"\begin{equation}",
    r"\begin{align}",
    r"\begin{theorem}",
    r"\begin{lemma}",
    r"\begin{proof}",
]

CATEGORY_WORDS = {
    "russian": RUSSIAN_MARKERS,
    "latex": LATEX_MATH_MARKERS,
}




# ─────────────────────────────────────────────────────────────────────────────
# 0.  Low-level text I/O  (mirrors the helper in the notebook)
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_text(doc_id: int, raw_text_dir: Path) -> str | None:
    """Return the raw text for *doc_id*, or None if the file is not found."""
    for candidate in [
        raw_text_dir / f'doc_{doc_id:05d}.txt',
        raw_text_dir / f'doc_{doc_id}.txt',
    ]:
        if candidate.exists():
            return candidate.read_text(encoding='utf-8', errors='replace')
    hits = (
        list(raw_text_dir.rglob(f'doc_{doc_id:05d}.txt'))
        + list(raw_text_dir.rglob(f'doc_{doc_id}.txt'))
    )
    return hits[0].read_text(encoding='utf-8', errors='replace') if hits else None


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Find documents by word list
# ─────────────────────────────────────────────────────────────────────────────

def find_docs_by_words(
    words: list[str],
    raw_text_dir: Path,
    index: list[dict],
    match: Literal['all', 'any'] = 'all',
    case_sensitive: bool = False,
    verbose: bool = True,
) -> list[dict]:
    """
    Search the corpus for documents that contain a set of words (or phrases).

    Parameters
    ----------
    words : list[str]
        Words or short phrases to search for.
    raw_text_dir : Path
        Directory containing the raw .txt files.
    index : list[dict]
        Row-level corpus index built in the notebook.  Each entry must have
        keys 'row_idx', 'doc_id', 'subdir', 'filename'.
    match : {'all', 'any'}
        'all' – document must contain *every* word (AND logic, default).
        'any' – document must contain *at least one* word (OR logic).
    case_sensitive : bool
        If False (default), matching is case-insensitive.
    verbose : bool
        Print a one-line progress summary when done.

    Returns
    -------
    list[dict]
        Subset of *index* entries for matching documents, in corpus order.
        Each dict is extended with a 'matched_words' key listing which of
        *words* were found in that document.
    """
    if not words:
        raise ValueError("'words' must be a non-empty list.")

    words_norm = words if case_sensitive else [w.lower() for w in words]

    results: list[dict] = []
    for entry in index:
        doc_id = entry.get('doc_id')
        if doc_id is None:
            continue
        text = load_raw_text(doc_id, raw_text_dir)
        if text is None:
            continue
        text_cmp = text if case_sensitive else text.lower()

        matched = [w for w in words_norm if w in text_cmp]

        hit = (match == 'all' and len(matched) == len(words_norm)) or \
              (match == 'any' and len(matched) > 0)
        if hit:
            results.append({**entry, 'matched_words': matched})

    if verbose:
        print(
            f"find_docs_by_words: {len(results)} / {len(index)} documents matched "
            f"(mode='{match}', case_sensitive={case_sensitive})"
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Top features for a set of documents
# ─────────────────────────────────────────────────────────────────────────────

def top_features_for_docs(
    docs: list[int | dict],
    codes_np: np.ndarray,
    k: int = 10,
    top_fraction: float | None = None,
    agg: Literal['mean', 'sum'] = 'mean',
) -> list[tuple[int, float]]:
    """
    Return the most-activated SAE features for a set of documents.

    Parameters
    ----------
    docs : list[int | dict]
        Either a list of integer row indices (into *codes_np*), or the list
        of metadata dicts returned by ``find_docs_by_words`` (the function
        will extract 'row_idx' automatically).
    codes_np : np.ndarray
        Shape (N, HIDDEN_DIM).  Full SAE activation matrix for the corpus.
    k : int
        Number of top features to return.  Ignored when *top_fraction* is set.
    top_fraction : float or None
        If given (e.g. 0.9), return the *minimal* set of features (ranked by
        score, highest first) whose cumulative activation accounts for at least
        this fraction of the total activation across the selected documents.
        Overrides *k*.
    agg : {'mean', 'sum'}
        Aggregation over the selected documents before ranking.
        'mean' (default) is scale-invariant to the number of documents.

    Returns
    -------
    list[tuple[int, float]]
        ``[(feature_idx, score), ...]``, sorted by descending score.
    """
    if len(docs) == 0:
        return []

    # Accept either raw row indices or metadata dicts
    row_indices = np.array(
        [d['row_idx'] if isinstance(d, dict) else int(d) for d in docs],
        dtype=int,
    )

    sub = codes_np[row_indices]               # (n_docs, HIDDEN_DIM)
    scores: np.ndarray = sub.mean(axis=0) if agg == 'mean' else sub.sum(axis=0)

    ranked = np.argsort(scores)[::-1]         # best feature first

    if top_fraction is not None:
        if not 0.0 < top_fraction <= 1.0:
            raise ValueError("top_fraction must be in (0, 1].")
        total = float(scores.sum())
        if total == 0.0:
            return []
        cumsum   = np.cumsum(scores[ranked])
        # first index i where cumsum[i] >= top_fraction * total
        i        = int(np.searchsorted(cumsum, top_fraction * total, side='left'))
        n_needed = min(i + 1, len(ranked))
        ranked   = ranked[:n_needed]
    else:
        ranked = ranked[:k]

    return [(int(fi), float(scores[fi])) for fi in ranked]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Map features → feature clusters
# ─────────────────────────────────────────────────────────────────────────────

def feature_clusters_for_features(
    feature_indices: list[int] | list[tuple[int, float]],
    feat_cluster_labels: np.ndarray,
    live_indices: np.ndarray,
    feature_l1_np: np.ndarray | None = None,
) -> dict:
    """
    Given a set of feature indices, return their cluster assignments.

    Parameters
    ----------
    feature_indices : list[int] or list[tuple[int, float]]
        Original (0-based) feature indices.  Accepts either plain ints *or*
        the ``(feature_idx, score)`` tuples returned by
        ``top_features_for_docs`` directly.
    feat_cluster_labels : np.ndarray
        Shape (n_live,).  Cluster label for each live feature, in the same
        positional order as *live_indices*.  Produced by KMeans in the
        notebook (``feat_cluster_labels``).
    live_indices : np.ndarray
        Shape (n_live,).  Maps each position in *feat_cluster_labels* back to
        an original feature index (``live_indices`` from the notebook).
    feature_l1_np : np.ndarray or None
        Shape (HIDDEN_DIM,).  Optional total-activation array used to report
        each feature's corpus-wide activity alongside cluster info.

    Returns
    -------
    dict with keys:
        'feature_to_cluster' : dict[int, int | None]
            Maps each requested feature index to its cluster id.
            Dead features (not in *live_indices*) map to None.
        'cluster_to_features' : dict[int, list[int]]
            Maps each cluster id to the requested features it contains,
            sorted by descending corpus-wide total activation (if
            *feature_l1_np* is supplied) or by feature index otherwise.
        'dead_features' : list[int]
            Features from *feature_indices* that are dead (no cluster).
        'summary' : list[dict]
            One dict per unique cluster, sorted by cluster id, each with:
              'cluster_id', 'n_features', 'feature_indices',
              and (if *feature_l1_np* given) 'total_activations'.
    """
    # Normalise input – accept (feature_idx, score) tuples or plain ints
    raw_indices: list[int] = []
    for item in feature_indices:
        if isinstance(item, (tuple, list)):
            raw_indices.append(int(item[0]))
        else:
            raw_indices.append(int(item))

    # Reverse lookup: original feature index → position in live_indices
    live_pos: dict[int, int] = {int(fi): pos for pos, fi in enumerate(live_indices)}

    feature_to_cluster: dict[int, int | None]      = {}
    cluster_to_features: dict[int, list[int]]       = defaultdict(list)
    dead_features: list[int]                        = []

    for fi in raw_indices:
        pos = live_pos.get(fi)
        if pos is None:
            feature_to_cluster[fi] = None
            dead_features.append(fi)
        else:
            cluster_id = int(feat_cluster_labels[pos])
            feature_to_cluster[fi] = cluster_id
            cluster_to_features[cluster_id].append(fi)

    # Sort features within each cluster by total activation (desc) if available
    for cid, feats in cluster_to_features.items():
        if feature_l1_np is not None:
            cluster_to_features[cid] = sorted(
                feats, key=lambda f: float(feature_l1_np[f]), reverse=True
            )
        else:
            cluster_to_features[cid] = sorted(feats)

    # Build human-readable summary
    summary = []
    for cid in sorted(cluster_to_features.keys()):
        feats = cluster_to_features[cid]
        entry: dict = {
            'cluster_id'     : cid,
            'n_features'     : len(feats),
            'feature_indices': feats,
        }
        if feature_l1_np is not None:
            entry['total_activations'] = [float(feature_l1_np[f]) for f in feats]
        summary.append(entry)

    return {
        'feature_to_cluster' : dict(feature_to_cluster),
        'cluster_to_features': dict(cluster_to_features),
        'dead_features'      : dead_features,
        'summary'            : summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: print a readable report of the full pipeline output
# ─────────────────────────────────────────────────────────────────────────────

def print_cluster_report(
    cluster_result: dict,
    feature_indices_with_scores: list[tuple[int, float]] | None = None,
) -> None:
    """
    Pretty-print the output of ``feature_clusters_for_features``.

    Parameters
    ----------
    cluster_result : dict
        Return value of ``feature_clusters_for_features``.
    feature_indices_with_scores : list[tuple[int, float]] or None
        The ``(feature_idx, score)`` list from ``top_features_for_docs``,
        used to annotate each feature with its doc-set activation score.
    """
    score_map: dict[int, float] = {}
    if feature_indices_with_scores:
        for fi, sc in feature_indices_with_scores:
            score_map[fi] = sc

    summary = cluster_result['summary']
    print(f"{'━'*60}")
    print(f"  {len(summary)} cluster(s) represented  |  "
          f"{len(cluster_result['dead_features'])} dead feature(s)")
    print(f"{'━'*60}")

    for entry in summary:
        cid   = entry['cluster_id']
        feats = entry['feature_indices']
        print(f"\n  Cluster {cid}  ({entry['n_features']} feature(s))")
        for i, fi in enumerate(feats):
            parts = [f"    feat {fi:5d}"]
            if 'total_activations' in entry:
                parts.append(f"corpus_act={entry['total_activations'][i]:.2f}")
        if fi in score_map:
            parts.append(f"enrichment={score_map[fi]:.4f}")
            print("  " + "  ".join(parts))

    if cluster_result['dead_features']:
        print(f"\n  Dead (no cluster): {cluster_result['dead_features']}")
    print()


def analyze_category(
    category: str,
    raw_text_dir: Path,
    index: list[dict],
    codes_np: np.ndarray,
    feat_cluster_labels: np.ndarray,
    live_indices: np.ndarray,
    feature_l1_np: np.ndarray | None = None,
    *,
    match: Literal['all', 'any'] = 'any',
    case_sensitive: bool = True,
    k: int = 20,
    top_fraction: float | None = None,
    verbose: bool = True,
    activation_threshold: float = 0.0,
    eps: float = 1e-12,
    ) -> dict:
    """
    Analyze a document category by feature enrichment:
        score_j = P(feature j active | subset) / P(feature j active)

    A feature is counted as active when codes_np[:, j] > activation_threshold.
    """
    if category not in CATEGORY_WORDS:
        raise ValueError(f"Unknown category '{category}'. Available: {sorted(CATEGORY_WORDS)}")

    words = CATEGORY_WORDS[category]

    docs = find_docs_by_words(
        words=words,
        raw_text_dir=raw_text_dir,
        index=index,
        match=match,
        case_sensitive=case_sensitive,
        verbose=verbose,
    )

    if len(docs) == 0:
        top_features = []
        cluster_result = {
            'feature_to_cluster': {},
            'cluster_to_features': {},
            'dead_features': [],
            'summary': [],
        }
        return {
            "category": category,
            "words": words,
            "docs": docs,
            "top_features": top_features,
            "cluster_result": cluster_result,
            "subset_frac_active": None,
            "global_frac_active": None,
            "enrichment_scores": None,
        }

    row_indices = np.array(
        [d['row_idx'] if isinstance(d, dict) else int(d) for d in docs],
        dtype=int,
    )

    active_all = (codes_np > activation_threshold)              # (N, H)
    active_sub = active_all[row_indices]                        # (n_subset, H)

    global_frac_active = active_all.mean(axis=0)               # P(feature active)
    subset_frac_active = active_sub.mean(axis=0)               # P(feature active | subset)

    enrichment_scores = subset_frac_active / (global_frac_active + eps)

    ranked = np.argsort(enrichment_scores)[::-1]

    if top_fraction is not None:
        if not 0.0 < top_fraction <= 1.0:
            raise ValueError("top_fraction must be in (0, 1].")
        total = float(enrichment_scores.sum())
        if total <= 0:
            top_features = []
        else:
            cumsum = np.cumsum(enrichment_scores[ranked])
            i = int(np.searchsorted(cumsum, top_fraction * total, side='left'))
            n_needed = min(i + 1, len(ranked))
            ranked = ranked[:n_needed]
            top_features = [(int(fi), float(enrichment_scores[fi])) for fi in ranked]
    else:
        ranked = ranked[:k]
        top_features = [(int(fi), float(enrichment_scores[fi])) for fi in ranked]

    cluster_result = feature_clusters_for_features(
        feature_indices=top_features,
        feat_cluster_labels=feat_cluster_labels,
        live_indices=live_indices,
        feature_l1_np=feature_l1_np,
    )

    return {
        "category": category,
        "words": words,
        "docs": docs,
        "top_features": top_features,
        "cluster_result": cluster_result,
        "subset_frac_active": subset_frac_active,
        "global_frac_active": global_frac_active,
        "enrichment_scores": enrichment_scores,
    }

def print_analysis_summary(
    result: dict,
    n_doc_examples: int = 5,) -> None:
        print(f"\n{'='*72}")
        print(f"Category: {result['category']}")
        print(f"Markers: {result['words']}")
        print(f"Matched documents: {len(result['docs'])}")

        if result["docs"]:
            print("\nExample matched docs:")
            for d in result["docs"][:n_doc_examples]:
                print(
                    f"  row_idx={d['row_idx']}  doc_id={d['doc_id']}  "
                    f"matched={d.get('matched_words', [])}"
                )

        subset_frac_active = result.get("subset_frac_active")
        global_frac_active = result.get("global_frac_active")

        print("\nTop features:")
        for fi, sc in result["top_features"]:
            parts = [f"  feat {fi:5d}  enrichment={sc:.6f}"]
            if subset_frac_active is not None and global_frac_active is not None:
                parts.append(f"P(active|subset)={subset_frac_active[fi]:.6f}")
                parts.append(f"P(active)={global_frac_active[fi]:.6f}")
            print("  ".join(parts))

        print_cluster_report(
            cluster_result=result["cluster_result"],
            feature_indices_with_scores=result["top_features"],
        )