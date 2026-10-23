"""Ranking metrics. Each takes the true relevance of every candidate and the model's scores.

Ties between equal scores are broken at random (seeded) rather than by list position, because
some models (the rule-based one especially) give many candidates the same score and a
position-based tie-break would silently favour whoever happens to be listed first.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def rank_order(scores: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Indices that sort candidates from best to worst score, random tie-break."""
    scores = np.asarray(scores, dtype=float)
    tiebreak = rng.random(len(scores)) if rng is not None else np.arange(len(scores))
    return np.lexsort((tiebreak, -scores))


def precision_at_k(rel, scores, k, rng=None) -> float:
    """Share of the top-k candidates that are relevant."""
    rel = np.asarray(rel) > 0
    k = min(k, len(rel))
    return float(rel[rank_order(scores, rng)[:k]].sum() / k)


def recall_at_k(rel, scores, k, rng=None) -> float:
    """Share of all relevant candidates that made it into the top-k."""
    rel = np.asarray(rel) > 0
    total = rel.sum()
    if total == 0:
        return float("nan")
    return float(rel[rank_order(scores, rng)[:k]].sum() / total)


def r_precision(rel, scores, rng=None) -> float:
    """Precision at k = number of relevant candidates (1.0 means every relevant one ranks first)."""
    rel = np.asarray(rel) > 0
    r = int(rel.sum())
    if r == 0:
        return float("nan")
    return float(rel[rank_order(scores, rng)[:r]].sum() / r)


def average_precision(rel, scores, rng=None) -> float:
    """Mean of precision@rank over the ranks where a relevant candidate appears (MAP when averaged)."""
    rel = np.asarray(rel) > 0
    if rel.sum() == 0:
        return float("nan")
    hits = rel[rank_order(scores, rng)]
    precisions = np.cumsum(hits) / (np.arange(len(hits)) + 1)
    return float(precisions[hits].mean())


def reciprocal_rank(rel, scores, rng=None) -> float:
    """1 / rank of the first relevant candidate (MRR when averaged)."""
    rel = np.asarray(rel) > 0
    hits = np.flatnonzero(rel[rank_order(scores, rng)])
    return float(1.0 / (hits[0] + 1)) if len(hits) else float("nan")


def ndcg_at_k(rel, scores, k, rng=None) -> float:
    """Normalised discounted cumulative gain: rewards relevant candidates ranked near the top."""
    rel = np.asarray(rel, dtype=float)
    if rel.sum() == 0:
        return float("nan")
    k = min(k, len(rel))
    discounts = 1.0 / np.log2(np.arange(k) + 2)
    dcg = float((rel[rank_order(scores, rng)[:k]] * discounts).sum())
    ideal = float((np.sort(rel)[::-1][:k] * discounts).sum())
    return dcg / ideal


def rank_correlation(scores, labels, method: str = "spearman") -> float:
    """Correlation between model scores and reference labels (recruiter ratings or the proxy)."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=float)
    if np.ptp(scores) == 0 or np.ptp(labels) == 0:
        return float("nan")
    fn = stats.spearmanr if method == "spearman" else stats.kendalltau
    return float(fn(scores, labels)[0])


def evaluate_ranking(rel, scores, ks=(5, 10, 20), rng=None) -> dict[str, float]:
    """All metrics for one job's ranking of one candidate pool."""
    out = {f"p@{k}": precision_at_k(rel, scores, k, rng) for k in ks}
    out["r_precision"] = r_precision(rel, scores, rng)
    out["ndcg@10"] = ndcg_at_k(rel, scores, 10, rng)
    out["map"] = average_precision(rel, scores, rng)
    out["mrr"] = reciprocal_rank(rel, scores, rng)
    out["spearman"] = rank_correlation(scores, rel, "spearman")
    out["kendall"] = rank_correlation(scores, rel, "kendall")
    return out
