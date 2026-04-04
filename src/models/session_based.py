"""
models/session_based.py
------------------------
Session-Based Recommender using Item-KNN with session co-occurrence.

Pure numpy/scipy implementation â€” no PyTorch required.

Algorithm
---------
For a current session S = [i1, i2, ..., in]:
  1. Look up all items that co-occurred with any item in S across training sessions.
  2. Score each candidate by sum of co-occurrence counts weighted by recency
     (more recent items in the session get higher weight: 1, 2, â€¦, n).
  3. Exclude items already seen in the current session.
  4. Return top-k by score.

EDA rationale
-------------
- >70% of visitors have only 1â€“3 events â€” sessions are the most reliable signal.
- Average session length ~2-4 items â†’ co-occurrence is well-defined.
- Item-KNN is competitive with GRU4Rec on short sessions (see Ludewig & Jannach 2018).
"""
from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


class SessionBasedRecommender:
    """Item-KNN session-based recommender using co-occurrence counts.

    Parameters
    ----------
    max_session_length:
        Maximum number of recent items in a session to consider.
    top_k_similar:
        Number of co-occurrence neighbours to store per item.
    """

    def __init__(
        self,
        emb_dim: int = 64,           # kept for API compatibility (unused)
        hidden_size: int = 128,      # kept for API compatibility (unused)
        num_layers: int = 1,         # kept for API compatibility (unused)
        dropout: float = 0.2,        # kept for API compatibility (unused)
        max_session_length: int = 20,
    ) -> None:
        self.max_session_length = max_session_length
        # co-occurrence index: item_id â†’ {neighbour_id: count}
        self._cooc: dict[int, dict[int, float]] = {}
        # global item popularity scores (fallback)
        self._popularity: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        sequences: pd.DataFrame,
        epochs: int = 1,        # unused, kept for API compatibility
        batch_size: int = 256,  # unused, kept for API compatibility
        lr: float = 0.001,      # unused, kept for API compatibility
    ) -> "SessionBasedRecommender":
        """Build co-occurrence index from session sequences.

        Parameters
        ----------
        sequences:
            DataFrame with columns [session_id, item_sequence, target_item].
        """
        logger.info("Building session co-occurrence index from %d sequences â€¦", len(sequences))

        cooc: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        popularity: dict[int, float] = defaultdict(float)

        for row in sequences.itertuples(index=False):
            seq: list[int] = list(row.item_sequence)
            target: int = int(row.target_item)
            n = len(seq)
            # Weight items by position (last item = highest weight)
            for pos, item in enumerate(seq):
                weight = float(pos + 1) / n
                popularity[item] += weight
                # co-occurrence: every item in seq co-occurs with target
                cooc[item][target] += weight
                cooc[target][item] += weight

        self._cooc = {k: dict(v) for k, v in cooc.items()}
        self._popularity = dict(popularity)
        logger.info(
            "Co-occurrence index built: %d items, %d total entries.",
            len(self._cooc),
            sum(len(v) for v in self._cooc.values()),
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def recommend(
        self, session_items: list[int], top_k: int = 10
    ) -> list[tuple[int, float]]:
        """Score candidates given the current session.

        Parameters
        ----------
        session_items:
            Ordered list of item IDs viewed in the current session.
        top_k:
            Number of top items to return.
        """
        session_items = session_items[-self.max_session_length:]
        seen = set(session_items)
        n = len(session_items)

        scores: dict[int, float] = defaultdict(float)
        for pos, item in enumerate(session_items):
            weight = float(pos + 1) / n   # recency weighting
            for neighbour, co_score in self._cooc.get(item, {}).items():
                if neighbour not in seen:
                    scores[neighbour] += weight * co_score

        if not scores:
            # Fallback: global popularity
            candidates = [
                (iid, sc) for iid, sc in self._popularity.items() if iid not in seen
            ]
        else:
            candidates = list(scores.items())

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("SessionBasedRecommender saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "SessionBasedRecommender":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        logger.info("SessionBasedRecommender loaded from %s", path)
        return obj
