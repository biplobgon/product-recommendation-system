"""
features/session_features.py
-----------------------------
Build session-level features and sequences for the session-based recommender.

EDA context
-----------
- Most users have 1–3 events — sessions are naturally short.
- Session boundary defined as 1-hour inactivity gap.
- Event sequence within a session is the primary signal for GRU4Rec.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)


def build_sessions(
    events: pd.DataFrame,
    session_gap_hours: float = 1.0,
) -> pd.DataFrame:
    """Segment user events into sessions based on inactivity gap.

    Parameters
    ----------
    events:
        Raw events DataFrame [timestamp, visitorid, event, itemid].
    session_gap_hours:
        Maximum inactivity gap (in hours) within a single session.

    Returns
    -------
    pd.DataFrame with original columns plus:
        - session_id   : globally unique session identifier
        - session_pos  : 0-based position of event within session
    """
    logger.info("Segmenting events into sessions (gap=%.1fh) …", session_gap_hours)

    df = events.copy().sort_values(["visitorid", "timestamp"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

    gap_threshold = pd.Timedelta(hours=session_gap_hours)
    df["time_diff"] = df.groupby("visitorid")["datetime"].diff()
    df["new_session"] = (df["time_diff"] > gap_threshold) | df["time_diff"].isna()
    df["session_id"] = df.groupby("visitorid")["new_session"].cumsum()
    df["session_id"] = df["visitorid"].astype(str) + "_" + df["session_id"].astype(str)
    df["session_pos"] = df.groupby("session_id").cumcount()

    df = df.drop(columns=["time_diff", "new_session"])

    n_sessions = df["session_id"].nunique()
    mean_len = df.groupby("session_id").size().mean()
    logger.info(
        "Sessions built: %d total, mean length %.2f events.", n_sessions, mean_len
    )
    return df


def build_session_sequences(
    sessions_df: pd.DataFrame,
    max_len: int = 20,
    min_len: int = 2,
) -> pd.DataFrame:
    """Convert sessions into padded item-id sequences for model input.

    Parameters
    ----------
    sessions_df:
        Output of :func:`build_sessions`.
    max_len:
        Truncate/pad sequences to this length (from the right).
    min_len:
        Discard sessions shorter than this.

    Returns
    -------
    pd.DataFrame with columns:
        - session_id
        - visitorid
        - item_sequence   : list of itemids (padded with 0s on the left)
        - target_item     : last item in the original untruncated sequence
        - session_length  : original session length (before padding)
        - has_purchase     : True if any event in session was a transaction
    """
    logger.info("Building session sequences (max_len=%d, min_len=%d) …", max_len, min_len)

    grp = sessions_df.groupby("session_id")

    records = []
    for sid, group in grp:
        group = group.sort_values("session_pos")
        items = group["itemid"].tolist()
        if len(items) < min_len:
            continue

        target = items[-1]
        input_items = items[:-1][-max_len:]          # all but last, capped
        padded = [0] * (max_len - len(input_items)) + input_items

        records.append(
            {
                "session_id": sid,
                "visitorid": group["visitorid"].iloc[0],
                "item_sequence": padded,
                "target_item": target,
                "session_length": len(items),
                "has_purchase": (group["event"] == "transaction").any(),
            }
        )

    result = pd.DataFrame(records)
    logger.info("Session sequences built: %d training samples.", len(result))
    return result


def build_session_features(
    sessions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Extract aggregate session-level features for the conversion ranker.

    Parameters
    ----------
    sessions_df:
        Output of :func:`build_sessions`.

    Returns
    -------
    pd.DataFrame with columns:
        - session_id
        - visitorid
        - n_views, n_addtocart, n_transactions
        - n_unique_items
        - session_duration_minutes
        - started_hour
        - has_purchase
    """
    df = sessions_df.copy()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

    agg = df.groupby(["session_id", "visitorid"]).agg(
        n_views=("event", lambda x: (x == "view").sum()),
        n_addtocart=("event", lambda x: (x == "addtocart").sum()),
        n_transactions=("event", lambda x: (x == "transaction").sum()),
        n_unique_items=("itemid", "nunique"),
        session_start=("datetime", "min"),
        session_end=("datetime", "max"),
    ).reset_index()

    agg["session_duration_minutes"] = (
        (agg["session_end"] - agg["session_start"]).dt.total_seconds() / 60
    ).round(2)
    agg["started_hour"] = agg["session_start"].dt.hour
    agg["has_purchase"] = agg["n_transactions"] > 0

    return agg.drop(columns=["session_start", "session_end"])
