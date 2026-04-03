"""
features/item_features.py
--------------------------
Build item-level features from item_properties and category_tree DataFrames.

EDA context
-----------
- ~20M property records combined across two time-split files.
- ~230k unique items in the catalog.
- ~185k items have both events and metadata (well-covered).
- ~230k items have metadata but no events (cold items).
- ~50k items have events but no metadata.
- Category tree: 1,600+ nodes, max depth 5, dominant depth 2–3.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from utils.logger import get_logger

logger = get_logger(__name__)


def build_item_features(
    item_props: pd.DataFrame,
    category_tree: pd.DataFrame,
) -> pd.DataFrame:
    """Produce a per-item feature DataFrame.

    Parameters
    ----------
    item_props:
        Combined item_properties DataFrame (parts 1 & 2) with columns
        [timestamp, itemid, property, value].
    category_tree:
        Category tree DataFrame with columns [categoryid, parentid].

    Returns
    -------
    pd.DataFrame
        One row per item with columns:
        - itemid
        - categoryid           (most recent category assignment)
        - category_depth       (depth of category in tree)
        - n_properties         (number of distinct property keys)
        - n_property_updates   (total property records)
        - price                (latest numeric price, if available)
        - available            (latest availability flag, if available)
        - first_seen, last_seen
    """
    logger.info("Building item features from %d property records …", len(item_props))

    df = item_props.copy()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

    # --- Latest value per (item, property) -----------------------------------
    latest = (
        df.sort_values("datetime")
        .groupby(["itemid", "property"])
        .last()
        .reset_index()[["itemid", "property", "value"]]
    )
    pivoted = latest.pivot(index="itemid", columns="property", values="value")

    # --- Core features -------------------------------------------------------
    agg = df.groupby("itemid").agg(
        n_property_updates=("property", "count"),
        n_properties=("property", "nunique"),
        first_seen=("datetime", "min"),
        last_seen=("datetime", "max"),
    ).reset_index()

    # Attach categoryid if the property exists
    if "categoryid" in pivoted.columns:
        cat_map = pivoted["categoryid"].reset_index()
        cat_map.columns = ["itemid", "categoryid"]
        cat_map["categoryid"] = pd.to_numeric(cat_map["categoryid"], errors="coerce")
        agg = agg.merge(cat_map, on="itemid", how="left")
    else:
        agg["categoryid"] = np.nan

    # Attach price
    if "790" in pivoted.columns:  # property 790 is commonly price in RetailRocket
        price_map = pivoted["790"].reset_index()
        price_map.columns = ["itemid", "price_raw"]
        price_map["price"] = pd.to_numeric(
            price_map["price_raw"].str.extract(r"([\d.]+)", expand=False),
            errors="coerce",
        )
        agg = agg.merge(price_map[["itemid", "price"]], on="itemid", how="left")
    else:
        agg["price"] = np.nan

    # Attach availability
    if "available" in pivoted.columns:
        avail_map = pivoted["available"].reset_index()
        avail_map.columns = ["itemid", "available_raw"]
        avail_map["available"] = avail_map["available_raw"].map(
            {"1": True, "0": False, 1: True, 0: False}
        )
        agg = agg.merge(avail_map[["itemid", "available"]], on="itemid", how="left")
    else:
        agg["available"] = np.nan

    # --- Category depth ------------------------------------------------------
    depth_map = _compute_category_depths(category_tree)
    agg["category_depth"] = agg["categoryid"].map(depth_map)

    logger.info("Item features built for %d unique items.", len(agg))
    return agg


def build_item_tfidf_matrix(
    item_props: pd.DataFrame,
    max_features: int = 5000,
) -> tuple[pd.DataFrame, object]:
    """Create a TF-IDF representation of item property values for content-based
    similarity computation.

    Parameters
    ----------
    item_props:
        Combined item properties DataFrame.
    max_features:
        Vocabulary size cap for TfidfVectorizer.

    Returns
    -------
    (item_ids_series, tfidf_matrix)
        item_ids_series : pd.Series of itemid values aligned with matrix rows.
        tfidf_matrix    : scipy sparse matrix of shape (n_items, max_features).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    logger.info("Building TF-IDF item matrix …")
    text_per_item = (
        item_props.groupby("itemid")["value"]
        .apply(lambda v: " ".join(v.dropna().astype(str)))
        .reset_index()
    )
    vectorizer = TfidfVectorizer(max_features=max_features, sublinear_tf=True)
    matrix = vectorizer.fit_transform(text_per_item["value"])
    logger.info("TF-IDF matrix shape: %s", matrix.shape)
    return text_per_item["itemid"], matrix, vectorizer


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_category_depths(category_tree: pd.DataFrame) -> dict:
    parent_map = dict(
        zip(category_tree["categoryid"], category_tree["parentid"])
    )
    depths: dict[int, int] = {}
    for cat in category_tree["categoryid"]:
        depth, node = 0, cat
        visited: set = set()
        while node in parent_map and not pd.isna(parent_map.get(node)) and node not in visited:
            visited.add(node)
            node = parent_map[node]
            depth += 1
        depths[cat] = depth
    return depths
