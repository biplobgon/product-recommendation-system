"""
resume_training.py
------------------
Loads already-saved ALS + CB models, trains Session-Based model,
assembles and saves the Hybrid model.  Skips the ~40-min CB step.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent
sys.path.insert(0, str(_SRC))

from utils.config import load_config
from utils.logger import get_logger
from features.user_features import build_user_item_matrix
from features.session_features import build_sessions, build_session_sequences
from models.collaborative_filtering import ALSRecommender
from models.content_based import ContentBasedRecommender
from models.session_based import SessionBasedRecommender
from models.hybrid import HybridRecommender

logger = get_logger(__name__)


def _find_project_root() -> Path:
    candidate = Path(__file__).resolve().parent
    for _ in range(6):
        if (candidate / "configs").is_dir():
            return candidate
        candidate = candidate.parent
    raise FileNotFoundError("Could not locate project root.")


def main() -> None:
    cfg = load_config()
    root = _find_project_root()
    models_dir = root / cfg.outputs.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load saved models
    # ------------------------------------------------------------------
    logger.info("Loading saved ALS model …")
    als = ALSRecommender.load(models_dir / "als_model.pkl")

    logger.info("Loading saved Content-Based model …")
    cb = ContentBasedRecommender.load(models_dir / "content_based_model.pkl")

    # ------------------------------------------------------------------
    # Build session sequences (need raw events)
    # ------------------------------------------------------------------
    logger.info("Loading events for session sequences …")
    events = pd.read_csv(root / cfg.data.events)

    logger.info("Engineering session features …")
    sessions_df = build_sessions(
        events, session_gap_hours=cfg.features.session_boundary_hours
    )
    sequences = build_session_sequences(
        sessions_df, max_len=cfg.session_based.max_session_length
    )

    # Temporal split
    events["datetime"] = pd.to_datetime(
        pd.to_numeric(events["timestamp"], errors="coerce"), unit="ms"
    )
    cutoff = events["datetime"].quantile(1 - cfg.evaluation.test_split_ratio)
    session_min_dt = sessions_df.groupby("session_id")["datetime"].min()
    train_sequences = sequences[
        sequences["session_id"].map(session_min_dt).fillna(cutoff) <= cutoff
    ]
    logger.info("Train sequences: %d", len(train_sequences))

    # ------------------------------------------------------------------
    # Train Session-Based model
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
    # Assemble Hybrid
    # ------------------------------------------------------------------
    logger.info("Assembling Hybrid Recommender …")
    weights = cfg.hybrid.weights
    weights_dict = weights.to_dict() if hasattr(weights, "to_dict") else weights
    hybrid = HybridRecommender(
        cf_model=als,
        cb_model=cb,
        sb_model=sb,
        weights=weights_dict,
    )
    import pickle
    with open(models_dir / "hybrid_model.pkl", "wb") as fh:
        pickle.dump(hybrid, fh)
    logger.info("Hybrid model saved.")
    logger.info("Done. All models saved to %s", models_dir)


if __name__ == "__main__":
    main()
