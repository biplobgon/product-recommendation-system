"""
app/api.py
----------
FastAPI service for the product recommendation system.

Endpoints
---------
GET  /                               → branded HTML landing page
GET  /health                         → liveness check + model availability flags
GET  /models                         → model registry + offline evaluation metrics
GET  /recommend/{visitor_id}         → recommendations for a known visitor (model selectable)
POST /recommend/session              → recommendations based on current session items (model selectable)
GET  /similar/{item_id}              → content-based similar items
GET  /popular                        → globally popular items (fallback)
GET  /metrics                        → offline evaluation report (JSON)

Model choices (model= query param)
-----------------------------------
  hybrid          → Blends ALS + TF-IDF + Item-KNN scores (default)
  als             → ALS collaborative filtering only
  content_based   → TF-IDF content-based only
  session_based   → Item-KNN session-based only

Usage
-----
    uvicorn src.app.api:app --reload --host 0.0.0.0 --port 8000
    # or from project root:
    python -m uvicorn src.app.api:app --reload
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Literal, Optional

# Ensure src/ is on the path
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from utils.config import load_config
from utils.logger import get_logger
from models.collaborative_filtering import ALSRecommender
from models.content_based import ContentBasedRecommender
from models.session_based import SessionBasedRecommender
from models.hybrid import HybridRecommender

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# OpenAPI description (rendered in Swagger UI)
# ---------------------------------------------------------------------------

_DESCRIPTION = """
## 🛍️ Product Recommendation System — REST API

End-to-end ML recommendation service trained on **2.75 M real e-commerce events**
from the [RetailRocket dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset).

### 🏗️ Architecture

A **4-model hybrid stack** that gracefully handles cold-start and warm-start users:

| Model | Key | Best for |
|---|---|---|
| 🔀 **Hybrid** *(default)* | `hybrid` | All users — blends all three signals |
| 🤝 **ALS Collaborative Filtering** | `als` | Returning visitors (≥ 2 events) |
| 🏷️ **Content-Based (TF-IDF)** | `content_based` | Cold items, no purchase history |
| ⚡ **Session-Based (Item-KNN)** | `session_based` | Anonymous visitors, session only |

### 📊 Offline Evaluation Highlights (@ K=10)

| Model | Hit Rate | NDCG | Coverage |
|---|---|---|---|
| Session-Based (Item-KNN) | **11.9 %** | 5.83 % | 6.71 % |
| Hybrid | 11.5 % | 5.64 % | 6.52 % |

> ALS and Content-Based show 0.0 on the temporal test split due to cold-start —
> this is realistic because ~71 % of visitors are new in any given window.

### 🔗 Links

- 🌐 **Live Streamlit App:** [recomsys.streamlit.app](https://recomsys.streamlit.app/)
- 🤗 **Model Artefacts:** [HuggingFace Hub](https://huggingface.co/datasets/biplobgon/product-recommendation-data)
- 🐳 **Docker Space:** [HF Spaces](https://huggingface.co/spaces/biplobgon/product-recommendation-system)
- 💻 **GitHub:** [biplobgon/product-recommendation-system](https://github.com/biplobgon/product-recommendation-system)

---
Built by **Biplob Gon** · [GitHub](https://github.com/biplobgon) · [LinkedIn](https://linkedin.com/in/biplobgon)
"""

# ---------------------------------------------------------------------------
# App setup  (docs disabled so we serve custom Swagger below)
# ---------------------------------------------------------------------------

app = FastAPI(
    title="🛍️ Product Recommendation API",
    description=_DESCRIPTION,
    version="2.0.0",
    docs_url=None,   # served manually at /docs with custom theme
    redoc_url=None,  # served manually at /redoc
    contact={
        "name": "Biplob Gon",
        "url":  "https://github.com/biplobgon",
    },
    license_info={
        "name": "MIT",
        "url":  "https://opensource.org/licenses/MIT",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Custom OpenAPI schema with tags ordering
# ---------------------------------------------------------------------------

def _custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        contact=app.contact,
        license_info=app.license_info,
        routes=app.routes,
        tags=[
            {"name": "System",       "description": "Health checks and service info."},
            {"name": "Recommend",    "description": "Generate personalised product recommendations."},
            {"name": "Discover",     "description": "Explore similar and trending items."},
            {"name": "Model Info",   "description": "Model registry and offline evaluation metrics."},
        ],
    )
    app.openapi_schema = schema
    return schema

app.openapi = _custom_openapi  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Custom docs & landing routes
# ---------------------------------------------------------------------------

_LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Product Recommendation API</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: linear-gradient(135deg, #0f1117 0%, #1a1d2e 60%, #0d1b2a 100%);
      min-height: 100vh; color: #e2e8f0;
    }
    header {
      background: rgba(255,255,255,0.04);
      border-bottom: 1px solid rgba(255,255,255,0.08);
      padding: 1.2rem 2.5rem;
      display: flex; align-items: center; gap: 1rem;
    }
    header .logo { font-size: 2rem; }
    header h1 { font-size: 1.3rem; font-weight: 700; color: #fff; letter-spacing: -0.02em; }
    header span { font-size: 0.75rem; background: #009688; color: #fff;
                   padding: 2px 8px; border-radius: 20px; margin-left: 0.5rem; }
    .hero {
      text-align: center; padding: 5rem 2rem 3rem;
    }
    .hero h2 {
      font-size: 3rem; font-weight: 800; letter-spacing: -0.04em;
      background: linear-gradient(90deg, #4ade80, #38bdf8, #a78bfa);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      margin-bottom: 1rem;
    }
    .hero p {
      font-size: 1.15rem; color: #94a3b8; max-width: 640px; margin: 0 auto 2.5rem;
      line-height: 1.7;
    }
    .cta-row { display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; }
    .btn {
      display: inline-flex; align-items: center; gap: 0.5rem;
      padding: 0.75rem 1.8rem; border-radius: 8px; font-size: 0.95rem;
      font-weight: 600; text-decoration: none; transition: all .2s;
    }
    .btn-primary { background: #009688; color: #fff; }
    .btn-primary:hover { background: #00bfa5; transform: translateY(-1px); }
    .btn-outline { border: 1px solid rgba(255,255,255,0.2); color: #e2e8f0; }
    .btn-outline:hover { background: rgba(255,255,255,0.06); transform: translateY(-1px); }
    .stats {
      display: flex; gap: 2rem; justify-content: center; flex-wrap: wrap;
      padding: 2rem 2rem 4rem;
    }
    .stat {
      background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
      border-radius: 12px; padding: 1.5rem 2.2rem; text-align: center;
    }
    .stat .num { font-size: 2rem; font-weight: 800; color: #38bdf8; }
    .stat .lbl { font-size: 0.8rem; color: #64748b; margin-top: 0.2rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .section { max-width: 1100px; margin: 0 auto; padding: 2rem; }
    .section h3 {
      font-size: 1.3rem; font-weight: 700; color: #fff;
      margin-bottom: 1.2rem; padding-bottom: 0.5rem;
      border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 1rem; margin-bottom: 3rem; }
    .card {
      background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
      border-radius: 12px; padding: 1.5rem; transition: border-color .2s;
    }
    .card:hover { border-color: rgba(56,189,248,0.4); }
    .card .icon { font-size: 1.8rem; margin-bottom: 0.7rem; }
    .card h4 { font-size: 0.95rem; font-weight: 700; color: #f1f5f9; margin-bottom: 0.4rem; }
    .card p { font-size: 0.82rem; color: #64748b; line-height: 1.5; }
    .card .badge {
      display: inline-block; margin-top: 0.6rem;
      font-size: 0.7rem; font-weight: 600; padding: 2px 8px;
      border-radius: 20px; background: rgba(56,189,248,0.15); color: #38bdf8;
    }
    .endpoint-list { margin-bottom: 3rem; }
    .ep {
      display: flex; align-items: center; gap: 1rem;
      padding: 0.85rem 1.2rem; border-radius: 8px;
      border: 1px solid rgba(255,255,255,0.06);
      background: rgba(255,255,255,0.02); margin-bottom: 0.5rem;
      text-decoration: none; color: inherit; transition: background .15s;
    }
    .ep:hover { background: rgba(255,255,255,0.06); }
    .method {
      font-size: 0.72rem; font-weight: 700; padding: 3px 8px;
      border-radius: 4px; width: 42px; text-align: center; flex-shrink: 0;
    }
    .get  { background: rgba(74,222,128,0.15); color: #4ade80; }
    .post { background: rgba(251,191,36,0.15); color: #fbbf24; }
    .path { font-family: "SF Mono", "Fira Code", monospace; font-size: 0.88rem; color: #e2e8f0; flex: 1; }
    .ep-desc { font-size: 0.8rem; color: #64748b; }
    .links { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 3rem; }
    .link-pill {
      display: inline-flex; align-items: center; gap: 0.4rem;
      padding: 0.5rem 1rem; border-radius: 8px; font-size: 0.85rem;
      border: 1px solid rgba(255,255,255,0.1); color: #94a3b8;
      text-decoration: none; transition: all .15s;
    }
    .link-pill:hover { border-color: #38bdf8; color: #38bdf8; }
    footer {
      text-align: center; padding: 2rem; font-size: 0.8rem; color: #334155;
      border-top: 1px solid rgba(255,255,255,0.05);
    }
    footer a { color: #475569; text-decoration: none; }
    footer a:hover { color: #94a3b8; }
  </style>
</head>
<body>
  <header>
    <span class="logo">&#x1F6CD;&#xFE0F;</span>
    <div>
      <h1>Product Recommendation API <span>v2.0.0</span></h1>
    </div>
  </header>

  <div class="hero">
    <h2>Predict What Shoppers Want Next</h2>
    <p>
      Production-grade recommendation service trained on <strong style="color:#f1f5f9">2.75 million</strong>
      real e-commerce events. Four-model hybrid stack &mdash; ALS &middot; TF-IDF &middot; Item-KNN &middot; Fusion.
    </p>
    <div class="cta-row">
      <a class="btn btn-primary" href="/docs">&#x1F4D6; Explore API Docs (Swagger)</a>
      <a class="btn btn-outline" href="/redoc">&#x1F4C4; ReDoc Reference</a>
      <a class="btn btn-outline" href="/health">&#x26A1; Health Check</a>
    </div>
  </div>

  <div class="stats">
    <div class="stat"><div class="num">2.75 M</div><div class="lbl">Training Events</div></div>
    <div class="stat"><div class="num">1.4 M</div><div class="lbl">Unique Visitors</div></div>
    <div class="stat"><div class="num">417 k</div><div class="lbl">Unique Items</div></div>
    <div class="stat"><div class="num">11.9 %</div><div class="lbl">Hit Rate @ K=10</div></div>
    <div class="stat"><div class="num">4</div><div class="lbl">Recommendation Models</div></div>
  </div>

  <div class="section">
    <h3>&#x1F3D7;&#xFE0F; Models</h3>
    <div class="cards">
      <div class="card">
        <div class="icon">&#x1F500;</div>
        <h4>Hybrid (All Models)</h4>
        <p>Blends ALS + TF-IDF + Item-KNN scores. Best overall &mdash; works for both new and returning visitors.</p>
        <span class="badge">model=hybrid &middot; Default</span>
      </div>
      <div class="card">
        <div class="icon">&#x1F91D;</div>
        <h4>ALS Collaborative Filtering</h4>
        <p>Learns user taste from past interactions. Works best for returning visitors with &ge; 2 events.</p>
        <span class="badge">model=als &middot; Warm users</span>
      </div>
      <div class="card">
        <div class="icon">&#x1F3F7;&#xFE0F;</div>
        <h4>Content-Based (TF-IDF)</h4>
        <p>Matches items by category / price / availability metadata. Great for cold items with no purchase history.</p>
        <span class="badge">model=content_based &middot; Cold items</span>
      </div>
      <div class="card">
        <div class="icon">&#x26A1;</div>
        <h4>Session-Based (Item-KNN)</h4>
        <p>Uses only the current browsing session. Works instantly for anonymous visitors &mdash; no history needed.</p>
        <span class="badge">model=session_based &middot; Any visitor</span>
      </div>
    </div>

    <h3>&#x1F50C; Endpoints</h3>
    <div class="endpoint-list">
      <a class="ep" href="/docs#/System/health_health_get">
        <span class="method get">GET</span>
        <span class="path">/health</span>
        <span class="ep-desc">Liveness probe + model availability status</span>
      </a>
      <a class="ep" href="/docs#/Model%20Info/model_info_models_get">
        <span class="method get">GET</span>
        <span class="path">/models</span>
        <span class="ep-desc">Model registry, descriptions, and offline evaluation metrics</span>
      </a>
      <a class="ep" href="/docs#/Model%20Info/evaluation_metrics_metrics_get">
        <span class="method get">GET</span>
        <span class="path">/metrics</span>
        <span class="ep-desc">Full offline evaluation report (Hit Rate, NDCG, MRR, Coverage, Novelty)</span>
      </a>
      <a class="ep" href="/docs#/Recommend/recommend_for_visitor_recommend__visitor_id__get">
        <span class="method get">GET</span>
        <span class="path">/recommend/{visitor_id}</span>
        <span class="ep-desc">Personalised recommendations for a known visitor</span>
      </a>
      <a class="ep" href="/docs#/Recommend/recommend_for_session_recommend_session_post">
        <span class="method post">POST</span>
        <span class="path">/recommend/session</span>
        <span class="ep-desc">Session-based recommendations &mdash; works for anonymous visitors too</span>
      </a>
      <a class="ep" href="/docs#/Discover/similar_items_similar__item_id__get">
        <span class="method get">GET</span>
        <span class="path">/similar/{item_id}</span>
        <span class="ep-desc">Content-based similar items</span>
      </a>
      <a class="ep" href="/docs#/Discover/popular_items_popular_get">
        <span class="method get">GET</span>
        <span class="path">/popular</span>
        <span class="ep-desc">Globally trending items ranked by interaction score</span>
      </a>
    </div>

    <h3>&#x1F517; Project Links</h3>
    <div class="links">
      <a class="link-pill" href="https://recomsys.streamlit.app/" target="_blank">&#x1F310; Live Streamlit App</a>
      <a class="link-pill" href="https://huggingface.co/datasets/biplobgon/product-recommendation-data" target="_blank">&#x1F917; Model Artefacts (HF Hub)</a>
      <a class="link-pill" href="https://huggingface.co/spaces/biplobgon/product-recommendation-system" target="_blank">&#x1F433; Docker Space (HF Spaces)</a>
      <a class="link-pill" href="https://github.com/biplobgon/product-recommendation-system" target="_blank">&#x1F4BB; GitHub Repository</a>
      <a class="link-pill" href="https://linkedin.com/in/biplobgon" target="_blank">&#x1F4BC; LinkedIn</a>
    </div>
  </div>

  <footer>
    Built by <a href="https://github.com/biplobgon"><strong>Biplob Gon</strong></a> &middot;
    RetailRocket Dataset (Jun&ndash;Sep 2015) &middot; FastAPI &middot; Python 3.14 &middot;
    <a href="/docs">Swagger UI</a> &middot; <a href="/redoc">ReDoc</a>
  </footer>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def landing_page():
    """Branded HTML landing page."""
    return HTMLResponse(_LANDING_HTML)


@app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
def custom_swagger():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Product Recommendation API — Swagger UI",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
        swagger_ui_parameters={
            "defaultModelsExpandDepth": -1,  # collapse schemas by default
            "docExpansion": "list",           # show endpoints collapsed
            "tryItOutEnabled": True,          # pre-enable Try It Out
            "deepLinking": True,
            "displayRequestDuration": True,
        },
    )


@app.get("/redoc", response_class=HTMLResponse, include_in_schema=False)
def custom_redoc():
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="Product Recommendation API — ReDoc",
        redoc_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
    )


# ---------------------------------------------------------------------------
# Model registry — mirrors dashboard.py MODEL_FILES
# ---------------------------------------------------------------------------

MODEL_KEY = Literal["hybrid", "als", "content_based", "session_based"]

_MODEL_META: dict[str, dict] = {
    "hybrid": {
        "path":  "hybrid_model.pkl",
        "label": "Hybrid (All Models)",
        "badge": "Recommended",
        "desc":  "Blends ALS + TF-IDF + Item-KNN scores. Best overall — works for both new and returning visitors.",
        "best_for": "All users",
        "metrics": {"hit_rate@10": 0.115, "ndcg@10": 0.0564, "mrr@10": 0.0381,
                    "coverage@10": 0.0652, "novelty@10": 15.3574},
    },
    "als": {
        "path":  "als_model.pkl",
        "label": "ALS (Collaborative Filtering)",
        "badge": "Warm users",
        "desc":  "Learns user taste from past interactions. Works best for returning visitors with >= 2 events.",
        "best_for": "Returning visitors",
        "metrics": {"hit_rate@10": 0.0, "ndcg@10": 0.0, "mrr@10": 0.0,
                    "coverage@10": 0.0, "novelty@10": 0.0},
    },
    "content_based": {
        "path":  "content_based_model.pkl",
        "label": "Content-Based (TF-IDF)",
        "badge": "Cold items",
        "desc":  "Matches items by category / price / availability metadata. Great for cold items with no purchase history.",
        "best_for": "Cold-start items",
        "metrics": {"hit_rate@10": 0.0, "ndcg@10": 0.0, "mrr@10": 0.0,
                    "coverage@10": 0.0, "novelty@10": 0.0},
    },
    "session_based": {
        "path":  "session_based_model.pkl",
        "label": "Session-Based (Item-KNN)",
        "badge": "Any visitor",
        "desc":  "Uses only the current browsing session. Works instantly for anonymous visitors — no history needed.",
        "best_for": "Anonymous visitors",
        "metrics": {"hit_rate@10": 0.1188, "ndcg@10": 0.0583, "mrr@10": 0.0394,
                    "coverage@10": 0.0671, "novelty@10": 15.4431},
    },
}

# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------

cfg = load_config()
_models_dir = _ROOT / cfg.outputs.models_dir
_data_dir = _ROOT / "data" / "processed"

_hybrid: HybridRecommender | None = None
_als: ALSRecommender | None = None
_cb: ContentBasedRecommender | None = None
_sb: SessionBasedRecommender | None = None
_interactions: pd.DataFrame | None = None
_item_features: pd.DataFrame | None = None
_category_depths: dict = {}
_popularity: list[tuple[int, float]] = []


# ---------------------------------------------------------------------------
# Model loading (done once at startup)
# ---------------------------------------------------------------------------

def _load_category_depths() -> dict:
    tree_path = _ROOT / "data" / "raw" / "category_tree.csv"
    if not tree_path.exists():
        return {}
    ct = pd.read_csv(tree_path)
    ct["categoryid"] = pd.to_numeric(ct["categoryid"], errors="coerce")
    ct["parentid"]   = pd.to_numeric(ct["parentid"],   errors="coerce")
    parent = dict(zip(ct["categoryid"], ct["parentid"]))
    depth: dict = {}

    def _d(cid, seen=None):
        if cid in depth:
            return depth[cid]
        if seen is None:
            seen = set()
        if cid in seen:
            return 0
        seen.add(cid)
        pid = parent.get(cid)
        depth[cid] = 0 if (pd.isna(pid) or pid not in parent) else _d(int(pid), seen) + 1
        return depth[cid]

    for cid in parent:
        _d(int(cid))
    return depth


@app.on_event("startup")
def _load_models() -> None:
    global _hybrid, _als, _cb, _sb, _interactions, _item_features, _category_depths, _popularity

    als_path = _models_dir / "als_model.pkl"
    if als_path.exists():
        try:
            _als = ALSRecommender.load(als_path)
            logger.info("Loaded ALS model from %s", als_path)
        except Exception as exc:
            logger.warning("Could not load ALS model: %s", exc)

    cb_path = _models_dir / "content_based_model.pkl"
    if cb_path.exists():
        try:
            _cb = ContentBasedRecommender.load(cb_path)
            logger.info("Loaded Content-Based model from %s", cb_path)
        except Exception as exc:
            logger.warning("Could not load Content-Based model: %s", exc)

    sb_path = _models_dir / "session_based_model.pkl"
    if sb_path.exists():
        try:
            _sb = SessionBasedRecommender.load(sb_path)
            logger.info("Loaded Session-Based model from %s", sb_path)
        except Exception as exc:
            logger.warning("Could not load Session-Based model: %s", exc)

    hybrid_path = _models_dir / "hybrid_model.pkl"
    if hybrid_path.exists():
        try:
            with open(hybrid_path, "rb") as f:
                _hybrid = pickle.load(f)
            logger.info("Loaded Hybrid model from %s", hybrid_path)
        except Exception as exc:
            logger.warning("Could not load Hybrid model from pickle (%s), building from parts.", exc)
            _hybrid = None

    # Build hybrid from parts if the pkl wasn't available
    if _hybrid is None:
        weights = cfg.hybrid.weights if hasattr(cfg, "hybrid") else None
        weights_dict = weights.to_dict() if hasattr(weights, "to_dict") else (weights or {})
        _hybrid = HybridRecommender(cf_model=_als, cb_model=_cb, sb_model=_sb, weights=weights_dict)
        logger.info("Built Hybrid model from component models.")

    if _als is None and _cb is None and _sb is None:
        logger.warning("No trained models found in %s — run train.py first.", _models_dir)

    # Interaction history for CF filtering + global popularity
    events_path = _ROOT / cfg.data.events
    if events_path.exists():
        events = pd.read_csv(events_path, usecols=["visitorid", "itemid", "event"])
        w_map = {"view": 1, "addtocart": 5, "transaction": 10}
        events["score"] = events["event"].map(w_map).fillna(1).astype(float)
        _interactions = events[["visitorid", "itemid", "score"]]
        pop = events.groupby("itemid")["score"].sum().sort_values(ascending=False)
        _popularity = list(zip(pop.index.tolist(), pop.values.tolist()))
        logger.info("Loaded %d interactions.", len(_interactions))

    # Item features (parquet preferred, CSV fallback)
    for name, reader in [("item_features.parquet", pd.read_parquet), ("item_features.csv", pd.read_csv)]:
        feat_path = _data_dir / name
        if feat_path.exists():
            df = reader(feat_path)
            if "itemid" in df.columns:
                df = df.set_index("itemid")
            _item_features = df
            logger.info("Loaded item features (%s): %d items.", name, len(df))
            break

    # Category depth lookup
    _category_depths = _load_category_depths()
    if _category_depths:
        logger.info("Loaded category depths for %d categories.", len(_category_depths))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class RecommendationItem(BaseModel):
    rank: int
    item_id: int
    score: float
    category_id: Optional[str] = None
    category_depth: Optional[int] = None
    price: Optional[str] = None
    available: Optional[str] = None


class SessionRequest(BaseModel):
    session_items: list[int] = Field(..., description="Ordered list of item IDs in the current session")
    visitor_id: Optional[int] = Field(None, description="Visitor ID (enables CF blending if known)")
    top_k: int = Field(10, ge=1, le=100)
    model: MODEL_KEY = Field("hybrid", description="Which model to use for recommendations")


class RecommendResponse(BaseModel):
    visitor_id: Optional[int]
    recommendations: list[RecommendationItem]
    model_used: str
    total_returned: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _enrich(recs: list[tuple[int, float]]) -> list[RecommendationItem]:
    """Attach full item metadata — mirrors _enrich_items() in dashboard.py."""
    out = []
    for rank, (item_id, score) in enumerate(recs, start=1):
        item_id = int(item_id)
        meta: dict = {}
        if _item_features is not None and item_id in _item_features.index:
            row = _item_features.loc[item_id]
            cat   = row.get("categoryid", None)
            price = row.get("price", None)
            avail = row.get("available", None)

            if pd.notna(cat):
                cid = int(cat)
                meta["category_id"]    = str(cid)
                meta["category_depth"] = _category_depths.get(cid)
            if pd.notna(price):
                meta["price"] = f"{float(price):.2f}"
            if avail is True:
                meta["available"] = "Yes"
            elif avail is False:
                meta["available"] = "No"
        out.append(RecommendationItem(rank=rank, item_id=item_id, score=round(float(score), 6), **meta))
    return out


def _get_recs(
    model_key: str,
    visitor_id: Optional[int],
    session_items: list[int],
    top_k: int,
) -> tuple[list[tuple[int, float]], str]:
    """Dispatch recommendations to the requested model."""
    if model_key == "als":
        if _als is None:
            raise HTTPException(503, "ALS model not loaded — run train.py first.")
        interactions = (
            pd.DataFrame({"visitorid": [visitor_id], "itemid": session_items[:1], "weight": [1]})
            if session_items else pd.DataFrame(columns=["visitorid", "itemid", "weight"])
        )
        return _als.recommend(visitor_id or -1, interactions, top_k=top_k), "als"

    elif model_key == "content_based":
        if _cb is None:
            raise HTTPException(503, "Content-Based model not loaded — run train.py first.")
        return _cb.recommend_for_session(session_items, top_k=top_k), "content_based"

    elif model_key == "session_based":
        if _sb is None:
            raise HTTPException(503, "Session-Based model not loaded — run train.py first.")
        return _sb.recommend(session_items, top_k=top_k), "session_based"

    else:  # hybrid (default)
        if _hybrid is None:
            raise HTTPException(503, "Hybrid model not loaded — run train.py first.")
        recs = _hybrid.recommend(
            visitor_id=visitor_id or -1,
            session_items=session_items,
            interactions=_interactions if visitor_id else None,
            top_k=top_k,
        )
        return recs, "hybrid"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    tags=["System"],
    summary="Liveness probe + model availability",
    response_description="Service status, loaded models map, and dataset statistics.",
)
def health():
    """Returns service status and which recommendation models are currently loaded.

    All four models should be `true` if `train.py` has been run successfully.
    """
    return {
        "status": "ok",
        "version": app.version,
        "models": {
            "hybrid":        _hybrid is not None,
            "als":           _als is not None,
            "content_based": _cb is not None,
            "session_based": _sb is not None,
        },
        "dataset": {
            "total_events":    "2.75M",
            "unique_visitors": "1.4M",
            "unique_items":    "417k",
            "source":          "RetailRocket (Jun–Sep 2015)",
        },
        "links": {
            "docs":      "/docs",
            "redoc":     "/redoc",
            "streamlit": "https://recomsys.streamlit.app/",
            "github":    "https://github.com/biplobgon/product-recommendation-system",
        },
    }


@app.get(
    "/models",
    tags=["Model Info"],
    summary="Model registry + offline evaluation metrics",
    response_description="All four models with descriptions, status, and offline performance metrics.",
)
def model_info():
    """Returns the full model registry including:

    - Human-readable descriptions for each model
    - Whether each model is currently loaded
    - Offline evaluation metrics @ K=10 (Hit Rate, NDCG, MRR, Coverage, Novelty)

    > **Note:** ALS and Content-Based show 0.0 on the temporal test split due to cold-start.
    > ~71 % of visitors are new in any given time window, which is realistic for production.
    """
    return {
        key: {
            "label":    meta["label"],
            "badge":    meta["badge"],
            "best_for": meta["best_for"],
            "description": meta["desc"],
            "loaded":   {
                "hybrid":        _hybrid is not None,
                "als":           _als is not None,
                "content_based": _cb is not None,
                "session_based": _sb is not None,
            }.get(key, False),
            "offline_metrics_at_k10": meta["metrics"],
        }
        for key, meta in _MODEL_META.items()
    }


@app.get(
    "/metrics",
    tags=["Model Info"],
    summary="Full offline evaluation report",
    response_description="Evaluation rows from evaluation_report.csv across all models and K values.",
)
def evaluation_metrics():
    """Returns the complete offline evaluation report across all models and K values.

    Metrics included:
    - **Hit Rate** — fraction of users whose next item appeared in top-K
    - **NDCG** — Normalised Discounted Cumulative Gain (rewards higher-ranked hits)
    - **MRR** — Mean Reciprocal Rank of the first correct hit
    - **Precision / Recall** — standard IR metrics
    - **Coverage** — share of the catalogue ever recommended (diversity)
    - **Novelty** — avg. log-popularity of recommended items (higher = more niche)
    """
    eval_path = _ROOT / "outputs" / "reports" / "evaluation_report.csv"
    if not eval_path.exists():
        raise HTTPException(
            404,
            "evaluation_report.csv not found — run `python src/training/run_evaluation.py` to generate it.",
        )
    df = pd.read_csv(eval_path)
    return JSONResponse(content={"evaluation_report": df.to_dict(orient="records")})


@app.get(
    "/recommend/{visitor_id}",
    response_model=RecommendResponse,
    tags=["Recommend"],
    summary="Personalised recommendations for a known visitor",
    response_description="Ranked list of recommended items with metadata.",
    responses={
        200: {
            "description": "Successful recommendation response",
            "content": {
                "application/json": {
                    "example": {
                        "visitor_id": 1287309,
                        "model_used": "hybrid",
                        "total_returned": 3,
                        "recommendations": [
                            {"rank": 1, "item_id": 461686, "score": 0.923, "category_id": "1016",
                             "category_depth": 2, "price": "3.50", "available": "Yes"},
                            {"rank": 2, "item_id": 119736, "score": 0.871, "category_id": "213",
                             "category_depth": 1, "price": "12.00", "available": "Yes"},
                            {"rank": 3, "item_id": 312728, "score": 0.754, "category_id": "1016",
                             "category_depth": 2, "price": None,   "available": "No"},
                        ],
                    }
                }
            },
        },
        503: {"description": "Model not loaded — run `train.py` first."},
    },
)
def recommend_for_visitor(
    visitor_id: int,
    top_k: int = Query(10, ge=1, le=100, description="Number of recommendations to return (1–100)."),
    session_items: str = Query(
        "",
        description="Comma-separated item IDs the visitor browsed this session, most-recent last. "
                    "Example: `461686,119736,312728`",
    ),
    model: MODEL_KEY = Query(
        "hybrid",
        description="Which model to use.\n\n"
                    "- `hybrid` *(default)* — best overall\n"
                    "- `als` — collaborative filtering\n"
                    "- `content_based` — TF-IDF metadata\n"
                    "- `session_based` — Item-KNN co-occurrence",
    ),
):
    """Return personalised recommendations for a known visitor.

    - **hybrid** *(default)*: blends CF + content + session signals.
    - **als**: collaborative filtering — requires visitor history in training data.
    - **content_based**: TF-IDF metadata similarity — uses session items.
    - **session_based**: Item-KNN — uses session items only, no history required.

    Automatically falls back to **global popularity** when the model returns no results.
    """
    items = [int(x) for x in session_items.split(",") if x.strip().isdigit()]
    recs, model_used = _get_recs(model, visitor_id, items, top_k)

    if not recs:
        recs = _popularity[:top_k]
        model_used = "popularity_fallback"

    enriched = _enrich(recs)
    return RecommendResponse(
        visitor_id=visitor_id,
        recommendations=enriched,
        model_used=model_used,
        total_returned=len(enriched),
    )


@app.post(
    "/recommend/session",
    response_model=RecommendResponse,
    tags=["Recommend"],
    summary="Session-based recommendations (anonymous or known visitor)",
    response_description="Ranked recommendations driven by the current session item list.",
    responses={
        400: {"description": "`session_items` must not be empty."},
        503: {"description": "Requested model not loaded."},
    },
)
def recommend_for_session(body: SessionRequest):
    """Return recommendations based on the current session item sequence.

    Works for **anonymous visitors** — no user history required.

    Pass `visitor_id` to enable collaborative-filtering blending in `hybrid` mode.
    Use `model` to choose which recommendation model to run.

    Automatically falls back to **global popularity** when no recommendations are returned.
    """
    if not body.session_items:
        raise HTTPException(400, "session_items must not be empty.")

    recs, model_used = _get_recs(body.model, body.visitor_id, body.session_items, body.top_k)

    if not recs:
        recs = _popularity[:body.top_k]
        model_used = "popularity_fallback"

    enriched = _enrich(recs)
    return RecommendResponse(
        visitor_id=body.visitor_id,
        recommendations=enriched,
        model_used=model_used,
        total_returned=len(enriched),
    )


@app.get(
    "/similar/{item_id}",
    response_model=RecommendResponse,
    tags=["Discover"],
    summary="Content-based similar items",
    response_description="Items most similar to the given item by TF-IDF metadata.",
    responses={
        404: {"description": "Item not found in content model."},
        503: {"description": "Content-Based model not loaded."},
    },
)
def similar_items(
    item_id: int,
    top_k: int = Query(10, ge=1, le=100, description="Number of similar items to return."),
):
    """Return items most similar to the given item using **TF-IDF content-based similarity**.

    Similarity is computed from item metadata: category, price band, and availability.
    Works even for items with no purchase history.
    """
    if _cb is None:
        raise HTTPException(503, "Content-Based model not loaded — run train.py first.")

    recs = _cb.recommend_similar(item_id, top_k=top_k)
    if not recs:
        raise HTTPException(404, f"Item {item_id} not found in the content model.")

    enriched = _enrich(recs)
    return RecommendResponse(
        visitor_id=None,
        recommendations=enriched,
        model_used="content_based",
        total_returned=len(enriched),
    )


@app.get(
    "/popular",
    response_model=RecommendResponse,
    tags=["Discover"],
    summary="Globally trending items",
    response_description="Top items ranked by weighted interaction score across all users.",
    responses={
        503: {"description": "Popularity data not available — events.csv was not loaded."},
    },
)
def popular_items(
    top_k: int = Query(10, ge=1, le=100, description="Number of popular items to return."),
):
    """Return the top-K most popular items ranked by **weighted interaction score**.

    Scoring: `view × 1 + add-to-cart × 5 + transaction × 10`

    Useful as a **cold-start fallback** when no visitor or session context is available.
    """
    recs = _popularity[:top_k]
    if not recs:
        raise HTTPException(503, "Popularity data not available — check that events.csv was loaded.")

    enriched = _enrich(recs)
    return RecommendResponse(
        visitor_id=None,
        recommendations=enriched,
        model_used="popularity",
        total_returned=len(enriched),
    )
