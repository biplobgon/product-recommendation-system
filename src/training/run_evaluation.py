"""
run_evaluation.py
-----------------
Loads saved models, evaluates them on the held-out test set, and writes
outputs/reports/evaluation_report.csv for the Streamlit dashboard.
"""
from __future__ import annotations

import sys
import pickle
from pathlib import Path

import numpy as np
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
from training.evaluate import evaluate_all_models

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
    reports_dir = root / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load raw data & build test sequences
    # ------------------------------------------------------------------
    logger.info("Loading events …")
    events = pd.read_csv(root / cfg.data.events)
    events["datetime"] = pd.to_datetime(
        pd.to_numeric(events["timestamp"], errors="coerce"), unit="ms"
    )
    cutoff = events["datetime"].quantile(1 - cfg.evaluation.test_split_ratio)

    logger.info("Building sessions …")
    sessions_df = build_sessions(events, session_gap_hours=cfg.features.session_boundary_hours)
    sequences = build_session_sequences(sessions_df, max_len=cfg.session_based.max_session_length)

    session_min_dt = sessions_df.groupby("session_id")["datetime"].min()
    test_sequences = sequences[
        sequences["session_id"].map(session_min_dt).fillna(cutoff) > cutoff
    ].reset_index(drop=True)
    logger.info("Test sequences: %d", len(test_sequences))

    # Cap at 5000 for speed
    if len(test_sequences) > 5000:
        test_sequences = test_sequences.sample(5000, random_state=42).reset_index(drop=True)
        logger.info("Sampled 5000 test sequences for evaluation speed.")

    interactions = build_user_item_matrix(events[events["datetime"] <= cutoff])
    catalogue_size = events["itemid"].nunique()
    item_popularity = events["itemid"].value_counts().to_dict()

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    logger.info("Loading models …")
    als = ALSRecommender.load(models_dir / "als_model.pkl")
    cb = ContentBasedRecommender.load(models_dir / "content_based_model.pkl")
    sb = SessionBasedRecommender.load(models_dir / "session_based_model.pkl")
    with open(models_dir / "hybrid_model.pkl", "rb") as fh:
        hybrid = pickle.load(fh)

    models = {
        "ALS (Collaborative Filtering)": als,
        "Content-Based (TF-IDF)": cb,
        "Session-Based (Item-KNN)": sb,
        "Hybrid (All Models)": hybrid,
    }

    # ------------------------------------------------------------------
    # Evaluate each model
    # ------------------------------------------------------------------
    all_results = []
    for name, model in models.items():
        logger.info("Evaluating: %s …", name)
        try:
            df = evaluate_all_models(
                model=model,
                test_sequences=test_sequences,
                interactions=interactions,
                catalogue_size=catalogue_size,
                item_popularity=item_popularity,
                k_values=[5, 10],
            )
            df = df.reset_index()
            df.insert(0, "model", name)
            all_results.append(df)
        except Exception as exc:
            logger.warning("Evaluation failed for %s: %s", name, exc)

    if not all_results:
        logger.error("No models evaluated successfully.")
        return

    report = pd.concat(all_results, ignore_index=True)
    out_path = reports_dir / "evaluation_report.csv"
    report.to_csv(out_path, index=False)
    logger.info("Saved evaluation report to %s", out_path)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
