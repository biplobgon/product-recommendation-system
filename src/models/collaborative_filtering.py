"""
models/collaborative_filtering.py
----------------------------------
Implicit-feedback Collaborative Filtering using Alternating Least Squares (ALS).

EDA rationale
-------------
- Events are purely implicit (no explicit ratings).
- ~2.75M interactions across ~235k items and ~1.4M visitors.
- Weighted confidence: c_ui = 1 + alpha * r_ui  (view=1, addtocart=5, txn=10).
- Suitable for users with ≥2 interactions (~30% of visitors).
- Cold-start users fall back to content-based or session-based models.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from utils.logger import get_logger

logger = get_logger(__name__)


class ALSRecommender:
    """Wrapper around ``implicit`` ALS for implicit-feedback CF.

    Parameters
    ----------
    factors:
        Number of latent factors.
    regularization:
        L2 regularisation weight.
    iterations:
        Number of ALS iterations.
    alpha:
        Confidence scaling factor.
    """

    def __init__(
        self,
        factors: int = 128,
        regularization: float = 0.01,
        iterations: int = 20,
        alpha: float = 40.0,
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self._model = None
        self._user_index: dict[int, int] = {}   # visitorid → row index
        self._item_index: dict[int, int] = {}   # itemid    → col index
        self._item_ids: list[int] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, interactions: pd.DataFrame) -> "ALSRecommender":
        """Train the ALS model.

        Parameters
        ----------
        interactions:
            DataFrame with columns [visitorid, itemid, score]
            (output of ``build_user_item_matrix``).

        Returns
        -------
        self
        """
        try:
            import implicit
        except ImportError as exc:
            raise ImportError(
                "Install 'implicit' to use ALSRecommender: pip install implicit"
            ) from exc

        logger.info("Fitting ALS model (factors=%d, iterations=%d) …", self.factors, self.iterations)

        users = interactions["visitorid"].unique()
        items = interactions["itemid"].unique()
        self._user_index = {u: i for i, u in enumerate(users)}
        self._item_index = {it: i for i, it in enumerate(items)}
        self._item_ids = list(items)

        rows = interactions["visitorid"].map(self._user_index)
        cols = interactions["itemid"].map(self._item_index)
        data = interactions["score"].values

        # item × user matrix (implicit convention)
        item_user = sp.csr_matrix(
            (data, (cols, rows)),
            shape=(len(items), len(users)),
            dtype=np.float32,
        )
        item_user_conf = (item_user * self.alpha).astype(np.float32)

        self._model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=42,
        )
        self._model.fit(item_user_conf)
        logger.info("ALS training complete.")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def recommend(
        self,
        visitor_id: int,
        interactions: pd.DataFrame,
        top_k: int = 10,
        filter_already_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Return top-k item recommendations for a visitor.

        Parameters
        ----------
        visitor_id:
            Target visitor.
        interactions:
            Full user-item interaction DataFrame for filtering seen items.
        top_k:
            Number of recommendations to return.
        filter_already_seen:
            Exclude items the visitor has already interacted with.

        Returns
        -------
        List of (itemid, score) tuples sorted by descending score.
        """
        if self._model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")

        if visitor_id not in self._user_index:
            logger.warning("Visitor %s not in training set — no CF recommendations.", visitor_id)
            return []

        user_idx = self._user_index[visitor_id]
        users = interactions["visitorid"].unique()
        items_all = interactions["itemid"].unique()

        rows = interactions["visitorid"].map(self._user_index).dropna().astype(int)
        cols = interactions["itemid"].map(self._item_index).dropna().astype(int)
        data = interactions["score"].values[rows.index]

        user_items = sp.csr_matrix(
            (data, (rows.values, cols.values)),
            shape=(len(users), len(items_all)),
        )

        recs = self._model.recommend(
            user_idx,
            user_items[user_idx],
            N=top_k,
            filter_already_liked=filter_already_seen,
        )
        return [(self._item_ids[i], float(s)) for i, s in zip(recs[0], recs[1])]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("ALS model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "ALSRecommender":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        logger.info("ALS model loaded from %s", path)
        return obj
