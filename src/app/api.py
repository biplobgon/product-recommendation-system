"""
app/api.py
----------
FastAPI service for the product recommendation system.

Endpoints
---------
GET  /health                         → liveness check
GET  /recommend/{visitor_id}         → hybrid recommendations for a known user
POST /recommend/session              → recommendations based on current session items
GET  /similar/{item_id}              → content-based similar items
GET  /popular                        → globally popular items (fallback)

Usage
-----
    uvicorn src.app.api:app --reload --host 0.0.0.0 --port 8000
    # or from project root:
    python -m uvicorn src.app.api:app --reload
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Ensure src/ is on the path
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utils.config import load_config
from utils.logger import get_logger
from models.collaborative_filtering import ALSRecommender
from models.content_based import ContentBasedRecommender
from models.session_based import SessionBasedRecommender
from models.hybrid import HybridRecommender

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Product Recommendation API",
    description="Hybrid recommendation system: ALS + Content-Based + Session-Based (Item-KNN)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model loading (done once at startup)
# ---------------------------------------------------------------------------

cfg = load_config()
_models_dir = _ROOT / cfg.outputs.models_dir
_data_dir = _ROOT / "data" / "processed"

_hybrid: HybridRecommender | None = None
_interactions: pd.DataFrame | None = None
_item_features: pd.DataFrame | None = None
_popularity: list[tuple[int, float]] = []


@app.on_event("startup")
def _load_models() -> None:
    global _hybrid, _interactions, _item_features, _popularity

    als, cb, sb = None, None, None

    als_path = _models_dir / "als_model.pkl"
    if als_path.exists():
        try:
            als = ALSRecommender.load(als_path)
            logger.info("Loaded ALS model from %s", als_path)
        except Exception as exc:
            logger.warning("Could not load ALS model: %s", exc)

    cb_path = _models_dir / "content_based_model.pkl"
    if cb_path.exists():
        try:
            cb = ContentBasedRecommender.load(cb_path)
            logger.info("Loaded CB model from %s", cb_path)
        except Exception as exc:
            logger.warning("Could not load CB model: %s", exc)

    sb_path = _models_dir / "session_based_model.pkl"
    if sb_path.exists():
        try:
            sb = SessionBasedRecommender.load(sb_path)
            logger.info("Loaded Session model from %s", sb_path)
        except Exception as exc:
            logger.warning("Could not load Session model: %s", exc)

    if als is None and cb is None and sb is None:
        logger.warning("No trained models found in %s — run train.py first.", _models_dir)

    weights = cfg.hybrid.weights if hasattr(cfg, "hybrid") else None
    weights_dict = weights.to_dict() if hasattr(weights, "to_dict") else (weights or {})
    _hybrid = HybridRecommender(cf_model=als, cb_model=cb, sb_model=sb, weights=weights_dict)

    # Load interaction history for CF filtering
    events_path = _ROOT / cfg.data.events
    if events_path.exists():
        events = pd.read_csv(events_path, usecols=["visitorid", "itemid", "event"])
        w_map = {"view": 1, "addtocart": 5, "transaction": 10}
        events["score"] = events["event"].map(w_map).fillna(1).astype(float)
        _interactions = events[["visitorid", "itemid", "score"]]
        # Compute global popularity
        pop = events.groupby("itemid")["score"].sum().sort_values(ascending=False)
        _popularity = list(zip(pop.index.tolist(), pop.values.tolist()))
        logger.info("Loaded %d interactions.", len(_interactions))

    # Load item features parquet if available
    feat_path = _data_dir / "item_features.parquet"
    if feat_path.exists():
        _item_features = pd.read_parquet(feat_path)
        logger.info("Loaded item features: %d items.", len(_item_features))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class RecommendationItem(BaseModel):
    item_id: int
    score: float
    category_id: Optional[str] = None
    available: Optional[str] = None

class SessionRequest(BaseModel):
    session_items: list[int] = Field(..., description="Ordered list of item IDs in the current session")
    visitor_id: Optional[int] = Field(None, description="Visitor ID (enables CF blending if known)")
    top_k: int = Field(10, ge=1, le=100)

class RecommendResponse(BaseModel):
    visitor_id: Optional[int]
    recommendations: list[RecommendationItem]
    model_used: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _enrich(recs: list[tuple[int, float]]) -> list[RecommendationItem]:
    """Attach item metadata if available."""
    out = []
    for item_id, score in recs:
        meta = {}
        if _item_features is not None and item_id in _item_features.index:
            row = _item_features.loc[item_id]
            meta = {
                "category_id": str(row.get("categoryid", "")) or None,
                "available": str(row.get("available", "")) or None,
            }
        out.append(RecommendationItem(item_id=item_id, score=round(score, 6), **meta))
    return out


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness probe."""
    models_loaded = {
        "als": _hybrid.cf_model is not None if _hybrid else False,
        "content_based": _hybrid.cb_model is not None if _hybrid else False,
        "session_based": _hybrid.sb_model is not None if _hybrid else False,
    }
    return {"status": "ok", "models": models_loaded}


@app.get(
    "/recommend/{visitor_id}",
    response_model=RecommendResponse,
    summary="Hybrid recommendations for a known visitor",
)
def recommend_for_visitor(
    visitor_id: int,
    top_k: int = Query(10, ge=1, le=100),
    session_items: str = Query("", description="Comma-separated recent item IDs"),
):
    """Return hybrid recommendations for a visitor ID.

    - Uses CF if visitor is in the training set.
    - Falls back to session-based and content-based.
    """
    if _hybrid is None:
        raise HTTPException(503, "Models not loaded — run train.py first.")

    items = [int(x) for x in session_items.split(",") if x.strip().isdigit()]
    recs = _hybrid.recommend(
        visitor_id=visitor_id,
        session_items=items,
        interactions=_interactions,
        top_k=top_k,
    )

    if not recs:
        # Fallback to popularity
        recs = _popularity[:top_k]
        model_used = "popularity_fallback"
    else:
        model_used = "hybrid"

    return RecommendResponse(
        visitor_id=visitor_id,
        recommendations=_enrich(recs),
        model_used=model_used,
    )


@app.post(
    "/recommend/session",
    response_model=RecommendResponse,
    summary="Session-based recommendations",
)
def recommend_for_session(body: SessionRequest):
    """Return recommendations based on the current session item sequence."""
    if _hybrid is None:
        raise HTTPException(503, "Models not loaded — run train.py first.")

    if not body.session_items:
        raise HTTPException(400, "session_items must not be empty.")

    recs = _hybrid.recommend(
        visitor_id=body.visitor_id or -1,
        session_items=body.session_items,
        interactions=_interactions if body.visitor_id else None,
        top_k=body.top_k,
    )

    if not recs:
        recs = _popularity[:body.top_k]
        model_used = "popularity_fallback"
    else:
        model_used = "hybrid"

    return RecommendResponse(
        visitor_id=body.visitor_id,
        recommendations=_enrich(recs),
        model_used=model_used,
    )


@app.get(
    "/similar/{item_id}",
    response_model=RecommendResponse,
    summary="Content-based similar items",
)
def similar_items(
    item_id: int,
    top_k: int = Query(10, ge=1, le=100),
):
    """Return items most similar to the given item (content-based)."""
    if _hybrid is None or _hybrid.cb_model is None:
        raise HTTPException(503, "Content-based model not loaded.")

    recs = _hybrid.cb_model.recommend_similar(item_id, top_k=top_k)
    if not recs:
        raise HTTPException(404, f"Item {item_id} not found in content model.")

    return RecommendResponse(
        visitor_id=None,
        recommendations=_enrich(recs),
        model_used="content_based",
    )


@app.get(
    "/popular",
    response_model=RecommendResponse,
    summary="Globally popular items",
)
def popular_items(top_k: int = Query(10, ge=1, le=100)):
    """Return the most popular items by interaction score."""
    recs = _popularity[:top_k]
    if not recs:
        raise HTTPException(503, "Popularity data not available.")

    return RecommendResponse(
        visitor_id=None,
        recommendations=_enrich(recs),
        model_used="popularity",
    )
