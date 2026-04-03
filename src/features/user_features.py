"""
features/user_features.py
--------------------------
Build user-level features from the events DataFrame.

EDA context
-----------
- Most visitors have only 1–3 events (highly sparse user history).
- Three event types: view (>95%), addtocart (<3%), transaction (<0.5%).
- Peak activity hours: 17:00–21:00.
- Session boundary: ~1 hour of inactivity.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

# Implicit feedback weights derived from EDA funnel analysis
EVENT_WEIGHTS = {"view": 1, "addtocart": 5, "transaction": 10}


def build_user_features(events: pd.DataFrame) -> pd.DataFrame:
    """Compute per-visitor aggregate features.

    Parameters
    ----------
    events:
        Raw events DataFrame with columns:
        [timestamp, visitorid, event, itemid, transactionid].

    Returns
    -------
    pd.DataFrame
        One row per visitor with columns:
        - visitorid
        - n_views, n_addtocart, n_transactions
        - n_unique_items
        - weighted_interaction_score
        - active_days
        - preferred_hour  (mode hour of activity)
        - conversion_rate  (transactions / views)
        - is_cold_start    (True if only 1 interaction)
    """
    logger.info("Building user features from %d events …", len(events))

    df = events.copy()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["hour"] = df["datetime"].dt.hour
    df["date"] = df["datetime"].dt.date
    df["weight"] = df["event"].map(EVENT_WEIGHTS).fillna(1)

    agg = df.groupby("visitorid").agg(
        n_views=("event", lambda x: (x == "view").sum()),
        n_addtocart=("event", lambda x: (x == "addtocart").sum()),
        n_transactions=("event", lambda x: (x == "transaction").sum()),
        n_unique_items=("itemid", "nunique"),
        weighted_interaction_score=("weight", "sum"),
        active_days=("date", "nunique"),
        preferred_hour=("hour", lambda x: x.mode()[0] if not x.empty else -1),
        first_seen=("datetime", "min"),
        last_seen=("datetime", "max"),
    ).reset_index()

    agg["conversion_rate"] = (
        agg["n_transactions"] / agg["n_views"].replace(0, np.nan)
    ).fillna(0.0)

    agg["is_cold_start"] = (
        agg["n_views"] + agg["n_addtocart"] + agg["n_transactions"]
    ) <= 1

    logger.info(
        "User features built: %d users, %.1f%% cold-start.",
        len(agg),
        agg["is_cold_start"].mean() * 100,
    )
    return agg


def build_user_item_matrix(
    events: pd.DataFrame,
    event_weights: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Build a sparse-friendly user × item interaction matrix.

    Returns a DataFrame in COO-style format (visitorid, itemid, score)
    suitable for passing to implicit ALS or SVD models.

    Parameters
    ----------
    events:
        Raw events DataFrame.
    event_weights:
        Override default EVENT_WEIGHTS mapping.

    Returns
    -------
    pd.DataFrame with columns [visitorid, itemid, score].
    """
    weights = event_weights or EVENT_WEIGHTS
    df = events.copy()
    df["score"] = df["event"].map(weights).fillna(1)

    matrix = (
        df.groupby(["visitorid", "itemid"])["score"]
        .sum()
        .reset_index()
    )
    logger.info(
        "User-item matrix: %d interactions across %d users × %d items.",
        len(matrix),
        matrix["visitorid"].nunique(),
        matrix["itemid"].nunique(),
    )
    return matrix
