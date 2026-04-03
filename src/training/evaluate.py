"""
training/evaluate.py
---------------------
Offline evaluation of all recommendation models.

Metrics
-------
- Hit Rate @ K  : fraction of test sessions where target item is in top-K.
- NDCG @ K      : normalised discounted cumulative gain.
- MRR @ K       : mean reciprocal rank.
- Precision @ K
- Recall @ K
- Coverage      : fraction of catalogue items recommended at least once.
- Novelty       : mean popularity-adjusted surprisal.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def hit_rate_at_k(recommendations: dict[str, list[int]], ground_truth: dict[str, int], k: int = 10) -> float:
    """Hit Rate @ K averaged over all sessions."""
    hits = sum(
        1 for sid, recs in recommendations.items()
        if ground_truth.get(sid) in recs[:k]
    )
    return hits / max(len(recommendations), 1)


def ndcg_at_k(recommendations: dict[str, list[int]], ground_truth: dict[str, int], k: int = 10) -> float:
    """NDCG @ K averaged over all sessions (single relevant item per session)."""
    total = 0.0
    for sid, recs in recommendations.items():
        target = ground_truth.get(sid)
        if target in recs[:k]:
            rank = recs[:k].index(target) + 1
            total += 1.0 / math.log2(rank + 1)
    return total / max(len(recommendations), 1)


def mrr_at_k(recommendations: dict[str, list[int]], ground_truth: dict[str, int], k: int = 10) -> float:
    """Mean Reciprocal Rank @ K."""
    total = 0.0
    for sid, recs in recommendations.items():
        target = ground_truth.get(sid)
        if target in recs[:k]:
            rank = recs[:k].index(target) + 1
            total += 1.0 / rank
    return total / max(len(recommendations), 1)


def precision_at_k(recommendations: dict[str, list[int]], ground_truth: dict[str, int], k: int = 5) -> float:
    """Precision @ K (1 relevant item per session)."""
    hits = sum(
        1 for sid, recs in recommendations.items()
        if ground_truth.get(sid) in recs[:k]
    )
    return hits / max(len(recommendations) * k, 1)


def recall_at_k(recommendations: dict[str, list[int]], ground_truth: dict[str, int], k: int = 10) -> float:
    """Recall @ K (equivalent to Hit Rate when there is one relevant item)."""
    return hit_rate_at_k(recommendations, ground_truth, k)


def catalogue_coverage(recommendations: dict[str, list[int]], catalogue_size: int, k: int = 10) -> float:
    """Fraction of catalogue items that appear in at least one recommendation."""
    recommended_items: set[int] = set()
    for recs in recommendations.values():
        recommended_items.update(recs[:k])
    return len(recommended_items) / max(catalogue_size, 1)


def novelty(
    recommendations: dict[str, list[int]],
    item_popularity: dict[int, int],
    k: int = 10,
) -> float:
    """Mean self-information (surprisal) of recommended items."""
    total_pop = sum(item_popularity.values())
    scores = []
    for recs in recommendations.values():
        for item in recs[:k]:
            p = item_popularity.get(item, 1) / total_pop
            scores.append(-math.log2(p))
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Pipeline evaluation function
# ---------------------------------------------------------------------------

def evaluate_all_models(
    model,
    test_sequences: pd.DataFrame,
    interactions: pd.DataFrame,
    catalogue_size: int,
    item_popularity: dict[int, int],
    k_values: list[int] | None = None,
) -> pd.DataFrame:
    """Run offline evaluation on a trained recommender.

    Parameters
    ----------
    model:
        Any model with a ``recommend(session_items, top_k)`` interface
        (or a HybridRecommender).
    test_sequences:
        DataFrame with columns [session_id, visitorid, item_sequence, target_item].
    interactions:
        Full user-item interaction DataFrame (for CF sub-model).
    catalogue_size:
        Total number of distinct items in the catalogue.
    item_popularity:
        Dict mapping itemid to interaction count.
    k_values:
        List of K values to evaluate. Defaults to [5, 10].

    Returns
    -------
    pd.DataFrame
        Metrics table with one row per K value.
    """
    if k_values is None:
        k_values = [5, 10]

    max_k = max(k_values)
    logger.info("Evaluating model on %d test sessions …", len(test_sequences))

    recommendations: dict[str, list[int]] = {}
    ground_truth: dict[str, int] = {}

    for _, row in test_sequences.iterrows():
        sid = row["session_id"]
        session_items = [i for i in row["item_sequence"] if i != 0]
        target = row["target_item"]
        ground_truth[sid] = target

        try:
            has_visitor = hasattr(model, "recommend") and "visitor_id" in str(
                model.recommend.__code__.co_varnames
            )
            if has_visitor:
                recs = model.recommend(
                    visitor_id=row["visitorid"],
                    session_items=session_items,
                    interactions=interactions,
                    top_k=max_k,
                )
            else:
                recs = model.recommend(session_items, top_k=max_k)
            recommendations[sid] = [r[0] for r in recs]
        except Exception as exc:
            logger.debug("Recommendation failed for session %s: %s", sid, exc)
            recommendations[sid] = []

    results = []
    for k in k_values:
        results.append(
            {
                "k": k,
                "hit_rate": round(hit_rate_at_k(recommendations, ground_truth, k), 4),
                "ndcg": round(ndcg_at_k(recommendations, ground_truth, k), 4),
                "mrr": round(mrr_at_k(recommendations, ground_truth, k), 4),
                "precision": round(precision_at_k(recommendations, ground_truth, k), 4),
                "recall": round(recall_at_k(recommendations, ground_truth, k), 4),
                "coverage": round(catalogue_coverage(recommendations, catalogue_size, k), 4),
                "novelty": round(novelty(recommendations, item_popularity, k), 4),
            }
        )

    results_df = pd.DataFrame(results).set_index("k")
    logger.info("Evaluation complete:\n%s", results_df.to_string())
    return results_df
