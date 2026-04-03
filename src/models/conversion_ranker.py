"""
models/conversion_ranker.py
----------------------------
LightGBM binary classifier that re-ranks candidate items by predicted
probability of conversion (addtocart or transaction).

EDA rationale
-------------
- Strong funnel imbalance: views >> addtocart >> transactions.
- Re-ranking candidate recommendations by conversion probability lifts
  business metrics (revenue) even when recall metrics are similar.
- Features: session-level (n_views, session_duration) + item-level
  (n_property_updates, category_depth, price) + interaction signals.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

FEATURE_COLS = [
    # User/session features
    "n_views",
    "n_addtocart",
    "session_duration_minutes",
    "started_hour",
    "n_unique_items_session",
    # Item features
    "item_n_properties",
    "item_n_property_updates",
    "item_category_depth",
    "item_price",
    # Cross features
    "user_weighted_score",
    "item_global_view_count",
    "item_global_addtocart_count",
]


class ConversionRanker:
    """LightGBM-based conversion probability ranker.

    Parameters
    ----------
    n_estimators:
        Number of boosting rounds.
    num_leaves:
        Maximum number of leaves per tree.
    learning_rate:
        Boosting learning rate.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        num_leaves: int = 63,
        learning_rate: float = 0.05,
    ) -> None:
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self._model = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "ConversionRanker":
        """Train the conversion ranker.

        Parameters
        ----------
        X_train:
            Feature DataFrame with columns in FEATURE_COLS.
        y_train:
            Binary target: 1 if event was addtocart or transaction, else 0.
        X_val / y_val:
            Optional validation set for early stopping.

        Returns
        -------
        self
        """
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("Install LightGBM: pip install lightgbm") from exc

        logger.info(
            "Training ConversionRanker on %d samples (pos rate: %.2f%%) …",
            len(X_train),
            y_train.mean() * 100,
        )
        available_cols = [c for c in FEATURE_COLS if c in X_train.columns]

        callbacks = [lgb.log_evaluation(period=50)]
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_val[available_cols], y_val)]
            callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=False))

        self._model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            objective="binary",
            metric="auc",
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            min_child_samples=20,
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(
            X_train[available_cols],
            y_train,
            eval_set=eval_set or None,
            callbacks=callbacks,
        )
        logger.info("ConversionRanker training complete.")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Return conversion probability for each row.

        Parameters
        ----------
        X:
            Feature DataFrame.

        Returns
        -------
        np.ndarray of shape (n_samples,) with probabilities in [0, 1].
        """
        if self._model is None:
            raise RuntimeError("Call fit() before score().")
        available_cols = [c for c in FEATURE_COLS if c in X.columns]
        return self._model.predict_proba(X[available_cols])[:, 1]

    def rerank(
        self,
        candidates: list[tuple[int, float]],
        candidate_features: pd.DataFrame,
    ) -> list[tuple[int, float]]:
        """Re-rank a candidate list by predicted conversion probability.

        Parameters
        ----------
        candidates:
            List of (itemid, base_score) from upstream models.
        candidate_features:
            Feature DataFrame indexed by itemid.

        Returns
        -------
        Re-ranked list of (itemid, conversion_probability) tuples.
        """
        if not candidates:
            return []
        item_ids = [c[0] for c in candidates]
        feats = candidate_features.loc[candidate_features.index.isin(item_ids)]
        if feats.empty:
            return candidates
        probs = self.score(feats)
        ranked = sorted(
            zip(feats.index.tolist(), probs.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importances as a DataFrame."""
        if self._model is None:
            raise RuntimeError("Call fit() before feature_importance().")
        return pd.DataFrame(
            {
                "feature": self._model.feature_name_,
                "importance": self._model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("ConversionRanker saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "ConversionRanker":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        logger.info("ConversionRanker loaded from %s", path)
        return obj
