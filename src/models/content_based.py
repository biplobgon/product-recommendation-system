"""
models/content_based.py
------------------------
Content-Based Filtering using TF-IDF on item property values.

EDA rationale
-------------
- ~230k items have metadata but no behavioral events (pure cold-start).
- ~50k items have events but no metadata — no CB signal for those.
- TF-IDF on concatenated property values captures item similarity without
  requiring any user interaction history.
- Cosine similarity on the TF-IDF matrix is the core similarity function.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

from utils.logger import get_logger

logger = get_logger(__name__)


class ContentBasedRecommender:
    """Item-to-item content similarity recommender.

    Parameters
    ----------
    top_k_similar:
        Number of most-similar items to pre-compute per item.
    """

    def __init__(self, top_k_similar: int = 50) -> None:
        self.top_k_similar = top_k_similar
        self._similarity_index: dict[int, list[tuple[int, float]]] = {}
        self._item_ids: list[int] = []
        self._vectorizer = None
        self._tfidf_matrix = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        item_ids: pd.Series,
        tfidf_matrix: sp.spmatrix,
        vectorizer=None,
    ) -> "ContentBasedRecommender":
        """Build the item similarity index.

        Parameters
        ----------
        item_ids:
            Series of item IDs aligned with rows of tfidf_matrix.
        tfidf_matrix:
            Sparse TF-IDF matrix (n_items × n_features).
        vectorizer:
            Fitted TfidfVectorizer (stored for later inference on new items).

        Returns
        -------
        self
        """
        logger.info(
            "Building content similarity index for %d items …", len(item_ids)
        )
        self._item_ids = item_ids.tolist()
        self._tfidf_matrix = tfidf_matrix
        self._vectorizer = vectorizer

        id_to_idx = {iid: i for i, iid in enumerate(self._item_ids)}
        batch_size = 1000
        n = len(self._item_ids)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = tfidf_matrix[start:end]
            sims = cosine_similarity(batch, tfidf_matrix)   # (batch, n)

            for local_i, global_i in enumerate(range(start, end)):
                row = sims[local_i]
                row[global_i] = -1.0          # exclude self
                top_indices = np.argpartition(row, -self.top_k_similar)[-self.top_k_similar:]
                top_indices = top_indices[np.argsort(row[top_indices])[::-1]]
                self._similarity_index[self._item_ids[global_i]] = [
                    (self._item_ids[j], float(row[j])) for j in top_indices
                ]

            if start % 10000 == 0:
                logger.info("  Similarity index: %d / %d items processed.", end, n)

        logger.info("Content similarity index built.")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def recommend_similar(
        self, item_id: int, top_k: int = 10
    ) -> list[tuple[int, float]]:
        """Return items most similar to a given item.

        Parameters
        ----------
        item_id:
            Seed item.
        top_k:
            Number of similar items to return.

        Returns
        -------
        List of (itemid, similarity_score) tuples.
        """
        if item_id not in self._similarity_index:
            logger.warning("Item %s not in similarity index.", item_id)
            return []
        return self._similarity_index[item_id][:top_k]

    def recommend_for_session(
        self, session_items: list[int], top_k: int = 10
    ) -> list[tuple[int, float]]:
        """Recommend items based on the items viewed in the current session.

        Aggregates similarity scores across all session items and returns
        the top-k candidates not already in the session.

        Parameters
        ----------
        session_items:
            Ordered list of item IDs in the current session.
        top_k:
            Number of recommendations.

        Returns
        -------
        List of (itemid, aggregated_score) tuples.
        """
        scores: dict[int, float] = {}
        seen = set(session_items)

        for seed_item in session_items:
            for candidate_id, sim in self.recommend_similar(seed_item, top_k=50):
                if candidate_id not in seen:
                    scores[candidate_id] = scores.get(candidate_id, 0.0) + sim

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("ContentBasedRecommender saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "ContentBasedRecommender":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        logger.info("ContentBasedRecommender loaded from %s", path)
        return obj
