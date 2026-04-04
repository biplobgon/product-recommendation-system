"""
training/train.py
-----------------
End-to-end training pipeline.

Steps
-----
1. Load raw data.
2. Feature engineering (user, item, session).
3. Temporal train/val/test split.
4. Train: ALS, ContentBased, GRU4Rec, ConversionRanker.
5. Assemble Hybrid model.
6. Persist all artifacts to outputs/models/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

# Resolve project root so imports work regardless of cwd
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent
sys.path.insert(0, str(_SRC))

from utils.config import load_config
from utils.logger import get_logger
from features.user_features import build_user_item_matrix
from features.item_features import build_item_features, build_item_tfidf_matrix
from features.session_features import build_sessions, build_session_sequences
from models.collaborative_filtering import ALSRecommender
from models.content_based import ContentBasedRecommender
from models.session_based import SessionBasedRecommender
from models.hybrid import HybridRecommender

logger = get_logger(__name__)


def run_training_pipeline(config_path: str | Path | None = None) -> HybridRecommender:
    """Execute the full training pipeline.

    Parameters
    ----------
    config_path:
        Path to a YAML config file. Defaults to configs/model_config.yaml +
        configs/pipeline_config.yaml in the project root.

    Returns
    -------
    HybridRecommender
        Fully assembled and trained hybrid model.
    """
    cfg = load_config(config_path)
    project_root = _find_project_root()
    models_dir = project_root / cfg.outputs.models_dir

    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------
    logger.info("Loading raw data …")
    events = pd.read_csv(project_root / cfg.data.events)
    category_tree = pd.read_csv(project_root / cfg.data.category_tree)
    _TARGET_PROPS = {"categoryid", "790", "available"}
    _prop_chunks = []
    for _path in [cfg.data.item_props_part1, cfg.data.item_props_part2]:
        for _chunk in pd.read_csv(
            project_root / _path,
            chunksize=500_000,
            dtype=str,
        ):
            _chunk = _chunk[_chunk["property"].isin(_TARGET_PROPS)]
            _prop_chunks.append(_chunk)
    item_props = pd.concat(_prop_chunks, ignore_index=True)
    logger.info(
        "Loaded: %d events, %d item-property records.", len(events), len(item_props)
    )

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    logger.info("Engineering features …")
    interactions = build_user_item_matrix(events)
    item_features = build_item_features(item_props, category_tree)
    item_ids, tfidf_matrix, vectorizer = build_item_tfidf_matrix(
        item_props, max_features=min(cfg.features.item_embedding_dim * 10, 500)
    )

    sessions_df = build_sessions(
        events, session_gap_hours=cfg.features.session_boundary_hours
    )
    sequences = build_session_sequences(
        sessions_df, max_len=cfg.session_based.max_session_length
    )

    # ------------------------------------------------------------------
    # 3. Temporal split (use last 20% of time as test)
    # ------------------------------------------------------------------
    logger.info("Splitting data temporally …")
    events["datetime"] = pd.to_datetime(events["timestamp"], unit="ms")
    cutoff = events["datetime"].quantile(1 - cfg.evaluation.test_split_ratio)
    train_events = events[events["datetime"] <= cutoff]
    train_interactions = build_user_item_matrix(train_events)

    # Build a lookup: session_id → min datetime — O(n) not O(n²)
    session_min_dt = (
        sessions_df.groupby("session_id")["datetime"].min()
    )
    train_sequences = sequences[
        sequences["session_id"].map(session_min_dt).fillna(cutoff) <= cutoff
    ]

    # ------------------------------------------------------------------
    # 4a. Collaborative Filtering
    # ------------------------------------------------------------------
    logger.info("Training ALS Collaborative Filtering model …")
    _min_inter = getattr(cfg.collaborative_filtering, "min_interactions", 5)
    _user_counts = train_interactions.groupby("visitorid")["score"].count()
    _active = _user_counts[_user_counts >= _min_inter].index
    als_interactions = train_interactions[train_interactions["visitorid"].isin(_active)]
    logger.info(
        "ALS training on %d users with ≥%d interactions (%d total interactions).",
        len(_active), _min_inter, len(als_interactions),
    )
    als = ALSRecommender(
        factors=cfg.collaborative_filtering.factors,
        regularization=cfg.collaborative_filtering.regularization,
        iterations=cfg.collaborative_filtering.iterations,
        alpha=cfg.collaborative_filtering.alpha,
    )
    als.fit(als_interactions)
    als.save(models_dir / "als_model.pkl")

    # ------------------------------------------------------------------
    # 4b. Content-Based
    # ------------------------------------------------------------------
    logger.info("Training Content-Based model …")
    cb = ContentBasedRecommender(top_k_similar=cfg.content_based.top_k_similar_items)
    cb.fit(item_ids, tfidf_matrix, vectorizer)
    cb.save(models_dir / "content_based_model.pkl")

    # ------------------------------------------------------------------
    # 4c. Session-Based (GRU4Rec)
    # ------------------------------------------------------------------
    logger.info("Training Session-Based (Item-KNN co-occurrence) model …")
    sb = SessionBasedRecommender(
        max_session_length=cfg.session_based.max_session_length,
    )
    sb.fit(
        train_sequences,
        epochs=cfg.session_based.epochs,
        batch_size=cfg.session_based.batch_size,
        lr=cfg.session_based.learning_rate,
    )
    sb.save(models_dir / "session_based_model.pkl")

    # ------------------------------------------------------------------
    # 5. Assemble Hybrid
    # ------------------------------------------------------------------
    logger.info("Assembling Hybrid Recommender …")
    hybrid = HybridRecommender(
        cf_model=als,
        cb_model=cb,
        sb_model=sb,
        weights=cfg.hybrid.weights.to_dict() if hasattr(cfg.hybrid.weights, "to_dict") else cfg.hybrid.weights,
    )

    logger.info("Training pipeline complete. Models saved to %s", models_dir)
    return hybrid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    candidate = Path(__file__).resolve().parent
    for _ in range(6):
        if (candidate / "configs").is_dir():
            return candidate
        candidate = candidate.parent
    raise FileNotFoundError("Could not locate project root.")


if __name__ == "__main__":
    run_training_pipeline()
