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
    """Pure numpy/scipy ALS for implicit-feedback CF.

    Implements the Hu, Koren & Volinsky (2008) weighted ALS:
        confidence C_ui = 1 + alpha * r_ui
        objective: minimise sum_{u,i} c_ui*(p_ui - x_u^T*y_i)^2 + lambda*(||x||^2+||y||^2)

    No external C++ dependencies — works on any Python version.

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
        factors: int = 32,
        regularization: float = 0.01,
        iterations: int = 10,
        alpha: float = 40.0,
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self._X: np.ndarray | None = None   # user factors (n_users × factors)
        self._Y: np.ndarray | None = None   # item factors (n_items × factors)
        self._user_index: dict[int, int] = {}
        self._item_index: dict[int, int] = {}
        self._item_ids: list[int] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, interactions: pd.DataFrame) -> "ALSRecommender":
        """Train ALS on an interactions DataFrame with columns [visitorid, itemid, score]."""
        logger.info(
            "Fitting ALS model (factors=%d, iterations=%d, %d interactions) …",
            self.factors, self.iterations, len(interactions),
        )

        users = interactions["visitorid"].unique()
        items = interactions["itemid"].unique()
        n_users, n_items = len(users), len(items)
        self._user_index = {u: i for i, u in enumerate(users)}
        self._item_index = {it: i for i, it in enumerate(items)}
        self._item_ids = list(items)

        u_idx = interactions["visitorid"].map(self._user_index).values
        i_idx = interactions["itemid"].map(self._item_index).values
        data = interactions["score"].values.astype(np.float32)

        # Sparse user×item confidence matrix: C_ui = 1 + alpha * r_ui
        # We store only the non-zero (alpha * r_ui) part; background = 1 is handled analytically
        R = sp.csr_matrix((data, (u_idx, i_idx)), shape=(n_users, n_items), dtype=np.float32)
        # C_data[k] = 1 + alpha * R.data[k]  for each stored entry
        C_data = (1.0 + self.alpha * R.data).astype(np.float32)

        # CSC version for item-step
        R_csc = R.tocsc()
        C_csc = sp.csc_matrix(
            (C_data[R.tocsr().tocoo().row.argsort()],   # re-sort nnz entries for csc
             (R_csc.indices, np.repeat(np.arange(n_items), np.diff(R_csc.indptr)))),
            shape=(n_users, n_items), dtype=np.float32,
        )
        # Simpler: rebuild C_csc from scratch
        C_csc = sp.csc_matrix(
            (1.0 + self.alpha * R_csc.data, R_csc.indices, R_csc.indptr),
            shape=(n_users, n_items), dtype=np.float32,
        )

        rng = np.random.default_rng(42)
        self._X = (rng.standard_normal((n_users, self.factors)) * 0.01).astype(np.float32)
        self._Y = (rng.standard_normal((n_items, self.factors)) * 0.01).astype(np.float32)

        lam_I = self.regularization * np.eye(self.factors, dtype=np.float32)

        for iteration in range(self.iterations):
            # --- update user factors ---
            YtY = self._Y.T @ self._Y
            for u in range(n_users):
                s, e = R.indptr[u], R.indptr[u + 1]
                if s == e:
                    continue
                ii = R.indices[s:e]
                c_u = C_data[s:e]          # confidence weights for user u
                Y_u = self._Y[ii]          # (nnz × f)
                A = YtY + (Y_u * (c_u - 1.0)[:, None]).T @ Y_u + lam_I
                b = (Y_u * c_u[:, None]).sum(axis=0)
                self._X[u] = np.linalg.solve(A, b)

            # --- update item factors ---
            XtX = self._X.T @ self._X
            for i in range(n_items):
                s, e = R_csc.indptr[i], R_csc.indptr[i + 1]
                if s == e:
                    continue
                uu = R_csc.indices[s:e]
                c_i = C_csc.data[s:e]
                X_i = self._X[uu]
                A = XtX + (X_i * (c_i - 1.0)[:, None]).T @ X_i + lam_I
                b = (X_i * c_i[:, None]).sum(axis=0)
                self._Y[i] = np.linalg.solve(A, b)

            logger.info("  ALS iteration %d / %d done.", iteration + 1, self.iterations)

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
        """Return top-k item recommendations for a visitor."""
        if self._X is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")

        if visitor_id not in self._user_index:
            logger.warning("Visitor %s not in training set — no CF recommendations.", visitor_id)
            return []

        u = self._user_index[visitor_id]
        scores = self._X[u] @ self._Y.T   # (n_items,)

        if filter_already_seen:
            seen_items = interactions.loc[
                interactions["visitorid"] == visitor_id, "itemid"
            ]
            for iid in seen_items:
                if iid in self._item_index:
                    scores[self._item_index[iid]] = -np.inf

        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(self._item_ids[i], float(scores[i])) for i in top_idx]

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
