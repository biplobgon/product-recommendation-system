"""
models/hybrid.py
-----------------
Hybrid Recommender that blends Collaborative Filtering, Content-Based,
and Session-Based scores into a final ranked list.

Architecture
------------
  1. Retrieve candidates from each sub-model.
  2. Normalise scores to [0, 1] per model.
  3. Compute weighted sum:
       final_score = w_cf * score_cf + w_cb * score_cb + w_sb * score_sb
  4. (Optional) Re-rank with ConversionRanker for business-metric optimisation.
  5. Return top-k.

Fallback policy (EDA-informed)
-------------------------------
- Session-based is primary: always available for any active session.
- CF fallback: requires ≥2 historical interactions.
- CB fallback: requires item metadata (covers ~230k cold items).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


class HybridRecommender:
    """Weighted-blend hybrid recommender.

    Parameters
    ----------
    cf_model:
        Fitted ALSRecommender instance (or None to skip).
    cb_model:
        Fitted ContentBasedRecommender instance (or None to skip).
    sb_model:
        Fitted SessionBasedRecommender instance (or None to skip).
    ranker:
        Fitted ConversionRanker for final re-ranking (or None to skip).
    weights:
        Dict with keys 'collaborative_filtering', 'content_based',
        'session_based'. Values should sum to 1.0.
    """

    DEFAULT_WEIGHTS = {
        "collaborative_filtering": 0.35,
        "content_based": 0.25,
        "session_based": 0.40,
    }

    def __init__(
        self,
        cf_model=None,
        cb_model=None,
        sb_model=None,
        ranker=None,
        weights: dict | None = None,
    ) -> None:
        self.cf_model = cf_model
        self.cb_model = cb_model
        self.sb_model = sb_model
        self.ranker = ranker
        self.weights = weights or self.DEFAULT_WEIGHTS

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def recommend(
        self,
        visitor_id: int,
        session_items: list[int],
        interactions: pd.DataFrame | None = None,
        candidate_features: pd.DataFrame | None = None,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Generate hybrid recommendations.

        Parameters
        ----------
        visitor_id:
            ID of the visitor requesting recommendations.
        session_items:
            Ordered list of item IDs in the current session.
        interactions:
            Full user-item interaction DataFrame (needed for CF).
        candidate_features:
            Item feature DataFrame indexed by itemid (needed for re-ranking).
        top_k:
            Number of final recommendations.

        Returns
        -------
        List of (itemid, score) tuples sorted by descending score.
        """
        score_map: dict[int, float] = {}

        # --- Session-Based ---------------------------------------------------
        if self.sb_model is not None and session_items:
            sb_recs = self.sb_model.recommend(session_items, top_k=100)
            w = self.weights.get("session_based", 0.0)
            _merge(score_map, _normalise(sb_recs), w)
            logger.debug("SB contributed %d candidates.", len(sb_recs))

        # --- Content-Based ---------------------------------------------------
        if self.cb_model is not None and session_items:
            cb_recs = self.cb_model.recommend_for_session(session_items, top_k=100)
            w = self.weights.get("content_based", 0.0)
            _merge(score_map, _normalise(cb_recs), w)
            logger.debug("CB contributed %d candidates.", len(cb_recs))

        # --- Collaborative Filtering -----------------------------------------
        if self.cf_model is not None and interactions is not None:
            cf_recs = self.cf_model.recommend(
                visitor_id, interactions, top_k=100, filter_already_seen=True
            )
            w = self.weights.get("collaborative_filtering", 0.0)
            _merge(score_map, _normalise(cf_recs), w)
            logger.debug("CF contributed %d candidates.", len(cf_recs))

        if not score_map:
            logger.warning("No candidates generated for visitor %s.", visitor_id)
            return []

        # Sort by blended score
        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)

        # --- Conversion Re-Ranking -------------------------------------------
        if self.ranker is not None and candidate_features is not None:
            ranked = self.ranker.rerank(ranked, candidate_features)

        return ranked[:top_k]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(recs: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Min-max normalise scores to [0, 1]."""
    if not recs:
        return []
    scores = [s for _, s in recs]
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [(iid, 1.0) for iid, _ in recs]
    return [(iid, (s - mn) / (mx - mn)) for iid, s in recs]


def _merge(
    score_map: dict[int, float],
    recs: list[tuple[int, float]],
    weight: float,
) -> None:
    """Add weighted scores to the aggregate score map in-place."""
    for item_id, score in recs:
        score_map[item_id] = score_map.get(item_id, 0.0) + weight * score
