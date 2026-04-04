"""
app/dashboard.py
-----------------
Streamlit dashboard for the Product Recommendation System.

Features (per README)
---------------------
- Enter visitor ID or paste a browsing session to get live recommendations
- Toggle model  (ALS / Content-Based / Session / Hybrid)
- View per-model metric comparison bar charts from evaluation_report.csv
- Browse top recommended items with category labels

Run
---
    streamlit run src/app/dashboard.py
"""

from __future__ import annotations

import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# PATH SETUP — allow imports from src/
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]   # project root
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
MODELS_DIR  = ROOT / "outputs" / "models"
REPORTS_DIR = ROOT / "outputs" / "reports"
DATA_DIR    = ROOT / "data" / "processed"
EVAL_CSV    = REPORTS_DIR / "evaluation_report.csv"

MODEL_FILES = {
    "ALS (Collaborative Filtering)": {
        "model":  MODELS_DIR / "als_model.pkl",
        "meta":   MODELS_DIR / "als_model.pkl",
        "type":   "als",
    },
    "Content-Based (TF-IDF)": {
        "model":  MODELS_DIR / "content_based_model.pkl",
        "meta":   MODELS_DIR / "content_based_model.pkl",
        "type":   "cb",
    },
    "Session-Based (Item-KNN)": {
        "model":  MODELS_DIR / "session_based_model.pkl",
        "meta":   MODELS_DIR / "session_based_model.pkl",
        "type":   "sb",
    },
    "Hybrid (All Models)": {
        "meta":   MODELS_DIR / "hybrid_model.pkl",
        "type":   "hybrid",
    },
}

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Product Recommendation System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_item_features() -> pd.DataFrame | None:
    for name, reader in [("item_features.parquet", pd.read_parquet), ("item_features.csv", pd.read_csv)]:
        path = DATA_DIR / name
        if path.exists():
            df = reader(path)
            if "itemid" in df.columns:
                df = df.set_index("itemid")
            return df
    return None


def load_eval_report() -> pd.DataFrame | None:
    if EVAL_CSV.exists():
        return pd.read_csv(EVAL_CSV)
    return None


@st.cache_data(show_spinner=False)
def load_events_sample(n: int = 5000) -> pd.DataFrame | None:
    for name, reader in [("user_features.parquet", pd.read_parquet), ("user_features.csv", pd.read_csv)]:
        path = DATA_DIR / name
        if path.exists():
            return reader(path).head(n)
    return None


def _model_available(key: str) -> bool:
    """Return True if the model file for the given key exists on disk."""
    info = MODEL_FILES[key]
    path = info.get("model") or info.get("meta")
    return path is not None and Path(path).exists()


def _load_model(key: str):
    """Load a model/meta pair.  Returns (model_obj, meta_obj) or (None, None)."""
    info = MODEL_FILES[key]
    mtype = info["type"]

    try:
        if mtype == "als":
            from models.collaborative_filtering import ALSRecommender
            model = ALSRecommender.load(str(info["model"]))
            return model, model

        elif mtype == "cb":
            from models.content_based import ContentBasedRecommender
            model = ContentBasedRecommender.load(str(info["model"]))
            return model, model

        elif mtype == "sb":
            from models.session_based import SessionBasedRecommender
            model = SessionBasedRecommender.load(str(info["model"]))
            return model, model

        elif mtype == "hybrid":
            with open(info["meta"], "rb") as f:
                model = pickle.load(f)
            return model, model

    except Exception as exc:
        st.error(f"Could not load model **{key}**: {exc}")
        return None, None

    return None, None


def _get_recommendations(
    model_key: str,
    visitor_id: int | None,
    session_items: list[int],
    top_k: int,
) -> list[int]:
    """Return a list of recommended item IDs (may be empty on error)."""
    model, meta = _load_model(model_key)
    mtype = MODEL_FILES[model_key]["type"]

    try:
        if mtype == "als" and model is not None:
            # Need interaction matrix — use empty if visitor unknown
            interactions = pd.DataFrame(
                {"visitorid": [visitor_id], "itemid": session_items[:1], "weight": [1]}
            ) if session_items else pd.DataFrame(columns=["visitorid", "itemid", "weight"])
            recs = model.recommend(visitor_id, interactions, top_k=top_k)
            return recs

        elif mtype == "cb" and model is not None:
            return model.recommend_for_session(session_items, top_k=top_k)

        elif mtype == "sb" and model is not None:
            return model.recommend(session_items, top_k=top_k)

        elif mtype == "hybrid" and model is not None:
            return model.recommend(
                visitor_id=visitor_id,
                session_items=session_items,
                top_k=top_k,
            )

    except Exception as exc:
        st.warning(f"Recommendation error: {exc}")
        return []

    return []


def _enrich_items(
    item_ids: list[int] | list[tuple[int, float]], item_df: pd.DataFrame | None
) -> pd.DataFrame:
    """Build a display DataFrame from a list of item IDs or (item_id, score) tuples."""
    rows = []
    for rank, entry in enumerate(item_ids, start=1):
        if isinstance(entry, (tuple, list)):
            iid, score = int(entry[0]), float(entry[1])
        else:
            iid, score = int(entry), None
        row: dict = {"Rank": rank, "Item ID": iid}
        if score is not None:
            row["Score"] = round(score, 4)
        if item_df is not None and iid in item_df.index:
            r = item_df.loc[iid]
            row["Category ID"]    = r.get("categoryid", "—")
            row["Category Depth"] = r.get("category_depth", "—")
            row["Price"]          = r.get("price", "—")
            row["Available"]      = "✅" if r.get("available", 1) else "❌"
        else:
            row["Category ID"]    = "—"
            row["Category Depth"] = "—"
            row["Price"]          = "—"
            row["Available"]      = "—"
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

st.sidebar.image(
    "https://img.shields.io/badge/Product-Recommendation_System-green?style=for-the-badge",
    width=280,
)
st.sidebar.title("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Model",
    options=list(MODEL_FILES.keys()),
    index=3,   # default: Hybrid
    help="Choose which recommendation model to use",
)

top_k = st.sidebar.slider(
    "Top-K recommendations",
    min_value=5,
    max_value=50,
    value=10,
    step=5,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Model blend weights (Hybrid)**  
    - ALS CF → 0.35  
    - TF-IDF CB → 0.25  
    - GRU4Rec SB → 0.40  
    
    **Event weights**  
    view = 1 · addtocart = 5 · transaction = 10  
    
    **Session boundary**: 1-hour inactivity gap  
    """
)

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

st.title("🛍️ Product Recommendation System")
st.caption(
    "RetailRocket dataset · ~2.75M events · 4-model hybrid architecture"
)

tab_recommend, tab_metrics, tab_data = st.tabs(
    ["🔮 Get Recommendations", "📊 Model Metrics", "🗂️ Data Explorer"]
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — RECOMMENDATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_recommend:
    col_input, col_output = st.columns([1, 2], gap="large")

    with col_input:
        st.subheader("Input")

        visitor_id_str = st.text_input(
            "Visitor ID",
            placeholder="e.g. 1287765",
            help="Leave blank if the user is anonymous / cold-start",
        )

        session_items_str = st.text_area(
            "Browsing Session (comma-separated item IDs)",
            placeholder="e.g. 461686, 277273, 122576",
            height=100,
            help="Items the visitor browsed in the current session (most recent last)",
        )

        run_btn = st.button("Get Recommendations ▶", type="primary", use_container_width=True)

        st.markdown("---")
        st.markdown(
            f"**Selected model:** `{model_choice}`  \n"
            f"**Top-K:** `{top_k}`"
        )

        # Model availability indicator
        available = _model_available(model_choice)
        if available:
            st.success("Model artefacts found ✅")
        else:
            st.warning(
                "Model artefacts not found in `outputs/models/`.  "
                "Run `python src/training/train.py` or notebook 03 first."
            )

    with col_output:
        st.subheader("Recommendations")

        if run_btn:
            # Parse inputs
            visitor_id: int | None = None
            if visitor_id_str.strip():
                try:
                    visitor_id = int(visitor_id_str.strip())
                except ValueError:
                    st.error("Visitor ID must be an integer.")
                    st.stop()

            session_items: list[int] = []
            if session_items_str.strip():
                try:
                    session_items = [
                        int(x.strip())
                        for x in session_items_str.split(",")
                        if x.strip()
                    ]
                except ValueError:
                    st.error("Session items must be comma-separated integers.")
                    st.stop()

            if not visitor_id and not session_items:
                st.warning("Provide at least a Visitor ID or one session item.")
            elif not available:
                # Demo mode — show placeholder output
                st.info(
                    "**Demo mode** — model not loaded. Showing random item IDs as placeholder."
                )
                rng = np.random.default_rng(seed=42)
                recs = rng.integers(100_000, 500_000, size=top_k).tolist()
                item_df = load_item_features()
                display_df = _enrich_items(recs, item_df)
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                with st.spinner("Generating recommendations…"):
                    recs = _get_recommendations(
                        model_choice, visitor_id, session_items, top_k
                    )

                if recs:
                    item_df = load_item_features()
                    display_df = _enrich_items(recs, item_df)
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    st.caption(f"{len(recs)} items recommended by **{model_choice}**")
                else:
                    st.info("No recommendations returned. Check inputs or model artefacts.")

        else:
            st.info("Fill in the inputs on the left and click **Get Recommendations ▶**")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — MODEL METRICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_metrics:
    st.subheader("Offline Evaluation Metrics")

    eval_df = load_eval_report()

    if eval_df is None:
        st.warning(
            "`outputs/reports/evaluation_report.csv` not found.  "
            "Run notebook `04_model_evaluation.ipynb` to generate it."
        )
    else:
        # Filter controls
        k_values = sorted(eval_df["k"].unique()) if "k" in eval_df.columns else []
        selected_k = st.selectbox(
            "K value", options=k_values, index=len(k_values) - 1 if k_values else 0
        )

        filtered = eval_df[eval_df["k"] == selected_k] if "k" in eval_df.columns else eval_df

        st.dataframe(filtered, use_container_width=True, hide_index=True)

        # Bar charts for each metric
        metric_cols = [
            c for c in filtered.columns
            if c not in ("model", "k") and filtered[c].dtype in (float, np.float64, int, np.int64)
        ]

        if metric_cols and "model" in filtered.columns:
            st.markdown("#### Visual Comparison")
            chart_cols = st.columns(min(3, len(metric_cols)))
            for idx, metric in enumerate(metric_cols):
                with chart_cols[idx % 3]:
                    chart_data = filtered.set_index("model")[[metric]]
                    st.markdown(f"**{metric}**")
                    st.bar_chart(chart_data, use_container_width=True)

        # Metric definitions
        with st.expander("ℹ️ Metric definitions"):
            st.markdown(
                """
| Metric | Description |
|---|---|
| **hit_rate** | Fraction of test users whose ground-truth item appears in top-K |
| **ndcg** | Normalised DCG — rewards ranking the true item higher |
| **mrr** | Mean Reciprocal Rank of the first relevant item |
| **precision** | Fraction of K recommendations that are relevant |
| **recall** | Fraction of all relevant items captured in top-K |
| **coverage** | Fraction of catalogue ever recommended (diversity proxy) |
| **novelty** | Avg. log-popularity of recommended items — higher = more niche |
                """
            )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — DATA EXPLORER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_data:
    st.subheader("Data Explorer")

    col_user, col_item = st.columns(2)

    with col_user:
        st.markdown("#### User Features (sample)")
        user_df = load_events_sample()
        if user_df is not None:
            st.dataframe(user_df, use_container_width=True, hide_index=False)

            with st.expander("Distribution: weighted_interaction_score"):
                if "weighted_interaction_score" in user_df.columns:
                    st.bar_chart(
                        user_df["weighted_interaction_score"]
                        .clip(upper=user_df["weighted_interaction_score"].quantile(0.95))
                        .value_counts()
                        .sort_index()
                    )
        else:
            st.info(
                "`data/processed/user_features.parquet` not found.  "
                "Run notebook `02_feature_engineering.ipynb` first."
            )

    with col_item:
        st.markdown("#### Item Features (sample)")
        item_df = load_item_features()
        if item_df is not None:
            st.dataframe(item_df.head(500), use_container_width=True, hide_index=False)

            with st.expander("Category depth distribution"):
                if "category_depth" in item_df.columns:
                    st.bar_chart(item_df["category_depth"].value_counts().sort_index())
        else:
            st.info(
                "`data/processed/item_features.parquet` not found.  "
                "Run notebook `02_feature_engineering.ipynb` first."
            )

    # EDA quick stats
    st.markdown("---")
    st.markdown("#### Dataset Summary (EDA findings)")
    summary_cols = st.columns(4)
    stats = [
        ("Total Events",   "~2.75M"),
        ("Unique Visitors","~1.4M"),
        ("Unique Items",   "~235k"),
        ("Categories",     "~1,600"),
    ]
    for col, (label, value) in zip(summary_cols, stats):
        col.metric(label, value)

    funnel_cols = st.columns(3)
    funnel = [
        ("Views",       "95.8 %"),
        ("Add-to-Cart", "2.7 %"),
        ("Purchases",   "0.5 %"),
    ]
    for col, (label, value) in zip(funnel_cols, funnel):
        col.metric(label, value)

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Built by **Biplob Gon** · "
    "[GitHub](https://github.com/biplobgon) · "
    "[LinkedIn](https://linkedin.com/in/biplobgon) · "
    "RetailRocket dataset (Jun–Sep 2015)"
)
