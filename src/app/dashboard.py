"""
app/dashboard.py
-----------------
Streamlit dashboard for the Product Recommendation System.

Tabs
----
1. Get Recommendations  - select a visitor + session items -> ranked output table
2. Model Metrics        - offline evaluation comparison across all 4 models
3. Data Explorer        - user/item feature distributions + EDA stats

Run
---
    streamlit run src/app/dashboard.py
"""

from __future__ import annotations

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
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
    "Hybrid (All Models)": {
        "meta":  MODELS_DIR / "hybrid_model.pkl",
        "type":  "hybrid",
        "desc":  "Blends ALS + TF-IDF + Item-KNN scores. Best overall - works for both new and returning visitors.",
        "badge": "Recommended",
    },
    "ALS (Collaborative Filtering)": {
        "model": MODELS_DIR / "als_model.pkl",
        "meta":  MODELS_DIR / "als_model.pkl",
        "type":  "als",
        "desc":  "Learns user taste from past interactions. Works best for returning visitors with >= 2 events.",
        "badge": "Warm users",
    },
    "Content-Based (TF-IDF)": {
        "model": MODELS_DIR / "content_based_model.pkl",
        "meta":  MODELS_DIR / "content_based_model.pkl",
        "type":  "cb",
        "desc":  "Matches items by category / price / availability metadata. Great for cold items with no purchase history.",
        "badge": "Cold items",
    },
    "Session-Based (Item-KNN)": {
        "model": MODELS_DIR / "session_based_model.pkl",
        "meta":  MODELS_DIR / "session_based_model.pkl",
        "type":  "sb",
        "desc":  "Uses only the current browsing session. Works instantly for anonymous visitors - no history needed.",
        "badge": "Any visitor",
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

st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; padding: 0.5rem 1.2rem; }
    [data-testid="metric-container"] { background: #f8f9fa; border-radius: 8px; padding: 0.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_category_depths() -> dict:
    tree_path = ROOT / "data" / "raw" / "category_tree.csv"
    if not tree_path.exists():
        return {}
    ct = pd.read_csv(tree_path)
    ct["categoryid"] = pd.to_numeric(ct["categoryid"], errors="coerce")
    ct["parentid"]   = pd.to_numeric(ct["parentid"],   errors="coerce")
    parent = dict(zip(ct["categoryid"], ct["parentid"]))
    depth = {}
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


@st.cache_data(show_spinner=False)
def load_item_features():
    for name, reader in [("item_features.parquet", pd.read_parquet), ("item_features.csv", pd.read_csv)]:
        path = DATA_DIR / name
        if path.exists():
            df = reader(path)
            if "itemid" in df.columns:
                df = df.set_index("itemid")
            if "category_depth" not in df.columns and "categoryid" in df.columns:
                depths = _load_category_depths()
                df["category_depth"] = df["categoryid"].map(depths)
            return df
    return None


def load_eval_report():
    return pd.read_csv(EVAL_CSV) if EVAL_CSV.exists() else None


@st.cache_data(show_spinner=False)
def load_user_features(n: int = 5000):
    for name, reader in [("user_features.parquet", pd.read_parquet), ("user_features.csv", pd.read_csv)]:
        path = DATA_DIR / name
        if path.exists():
            return reader(path).head(n)
    return None


@st.cache_data(show_spinner=False)
def load_top_visitor_ids(n: int = 300):
    for name, reader in [("user_features.parquet", pd.read_parquet), ("user_features.csv", pd.read_csv)]:
        path = DATA_DIR / name
        if path.exists():
            df = reader(path)
            cols = [c for c in ("n_views", "n_addtocart", "n_transactions") if c in df.columns]
            if cols:
                df["_total"] = df[cols].sum(axis=1)
                return df.nlargest(n, "_total")["visitorid"].astype(int).tolist()
            return df["visitorid"].astype(int).head(n).tolist()
    return []


@st.cache_data(show_spinner=False)
def load_top_item_ids(n: int = 500):
    inter_path = DATA_DIR / "interactions.csv"
    if inter_path.exists():
        df = pd.read_csv(inter_path, usecols=["itemid", "score"])
        return df.groupby("itemid")["score"].sum().nlargest(n).index.astype(int).tolist()
    item_df = load_item_features()
    return item_df.index.astype(int).tolist()[:n] if item_df is not None else []


# ---------------------------------------------------------------------------
# MODEL HELPERS
# ---------------------------------------------------------------------------

def _model_available(key: str) -> bool:
    info = MODEL_FILES[key]
    path = info.get("model") or info.get("meta")
    return path is not None and Path(path).exists()


def _load_model(key: str):
    info = MODEL_FILES[key]
    mtype = info["type"]
    try:
        if mtype == "als":
            from models.collaborative_filtering import ALSRecommender
            m = ALSRecommender.load(str(info["model"]))
            return m, m
        elif mtype == "cb":
            from models.content_based import ContentBasedRecommender
            m = ContentBasedRecommender.load(str(info["model"]))
            return m, m
        elif mtype == "sb":
            from models.session_based import SessionBasedRecommender
            m = SessionBasedRecommender.load(str(info["model"]))
            return m, m
        elif mtype == "hybrid":
            with open(info["meta"], "rb") as f:
                m = pickle.load(f)
            return m, m
    except Exception as exc:
        st.error(f"Could not load **{key}**: {exc}")
    return None, None


def _get_recommendations(key: str, visitor_id, session_items, top_k: int):
    model, _ = _load_model(key)
    mtype = MODEL_FILES[key]["type"]
    try:
        if mtype == "als" and model:
            interactions = (
                pd.DataFrame({"visitorid": [visitor_id], "itemid": session_items[:1], "weight": [1]})
                if session_items else pd.DataFrame(columns=["visitorid", "itemid", "weight"])
            )
            return model.recommend(visitor_id, interactions, top_k=top_k)
        elif mtype == "cb" and model:
            return model.recommend_for_session(session_items, top_k=top_k)
        elif mtype == "sb" and model:
            return model.recommend(session_items, top_k=top_k)
        elif mtype == "hybrid" and model:
            return model.recommend(visitor_id=visitor_id, session_items=session_items, top_k=top_k)
    except Exception as exc:
        st.warning(f"Recommendation error: {exc}")
    return []


def _enrich_items(item_ids, item_df) -> pd.DataFrame:
    rows = []
    for rank, entry in enumerate(item_ids, start=1):
        if isinstance(entry, (tuple, list)):
            iid, score = int(entry[0]), round(float(entry[1]), 4)
        else:
            iid, score = int(entry), None
        row = {"Rank": rank, "Item ID": iid}
        if score is not None:
            row["Relevance Score"] = score
        if item_df is not None and iid in item_df.index:
            r = item_df.loc[iid]
            cat   = r.get("categoryid", None)
            dep   = r.get("category_depth", None)
            price = r.get("price", None)
            avail = r.get("available", None)
            row["Category ID"]    = int(cat)           if pd.notna(cat)   else "—"
            row["Category Depth"] = int(dep)           if pd.notna(dep)   else "—"
            row["Price"]          = f"{price:.2f}"     if pd.notna(price) else "—"
            row["Available"]      = "Yes" if avail is True else ("No" if avail is False else "—")
        else:
            row.update({"Category ID": "—", "Category Depth": "—", "Price": "—", "Available": "—"})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛍️ Recommendation System")
    st.caption("RetailRocket · 2.75M events · 4-model hybrid stack")
    st.divider()

    st.markdown("### ⚙️ Controls")
    model_choice = st.selectbox(
        "Recommendation Model",
        options=list(MODEL_FILES.keys()),
        index=0,
        help="Select which model drives the recommendations in the Get Recommendations tab.",
    )
    top_k = st.slider("How many recommendations?", min_value=5, max_value=50, value=10, step=5)

    st.divider()
    info = MODEL_FILES[model_choice]
    st.markdown("**About this model**")
    st.info(f"**{info['badge']}**\n\n{info['desc']}")

    st.divider()
    st.markdown("### 📖 How to Use")
    st.markdown(
        "1. **Get Recommendations** tab\n"
        "   → Pick a visitor + items → click Run\n\n"
        "2. **Model Metrics** tab\n"
        "   → Compare all models side-by-side\n\n"
        "3. **Data Explorer** tab\n"
        "   → Browse user & item distributions"
    )
    st.divider()
    st.markdown(
        "<small>Built by **Biplob Gon** · "
        "[GitHub](https://github.com/biplobgon) · "
        "[LinkedIn](https://linkedin.com/in/biplobgon)</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# HERO HEADER
# ---------------------------------------------------------------------------
st.title("🛍️ Product Recommendation System")
st.markdown(
    "> **What this does:** Given a visitor's browsing session and/or past history, "
    "this system predicts the products they are most likely to purchase next — "
    "using a 4-model hybrid architecture trained on 2.75M real e-commerce events."
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Events",       "2.75 M")
k2.metric("Unique Visitors",    "1.4 M")
k3.metric("Unique Items",       "417 k")
k4.metric("Session-KNN Hit@10", "11.9 %", help="Hit Rate at K=10 from offline evaluation")
k5.metric("Hybrid Hit@10",      "11.5 %", help="Hit Rate at K=10 from offline evaluation")

st.divider()

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab_recommend, tab_metrics, tab_data = st.tabs([
    "🔮 Get Recommendations",
    "📊 Model Metrics",
    "🗂️ Data Explorer",
])

# ===========================================================================
# TAB 1 — GET RECOMMENDATIONS
# ===========================================================================
with tab_recommend:

    st.markdown(
        "### How to get recommendations\n"
        "Select a **visitor** from the dropdown (or leave as Anonymous for cold-start) and "
        "optionally pick items from their current browsing session. "
        "The model will return the top-K products most likely to be purchased next."
    )
    st.caption(
        "**Tip:** The Hybrid model works best for most inputs. "
        "Use Session-Based if you have no visitor history. "
        "Use Content-Based to explore similar items."
    )

    col_input, col_output = st.columns([1, 2], gap="large")

    with col_input:
        st.markdown("#### 🎛️ Inputs")

        top_visitors = load_top_visitor_ids()
        visitor_options = ["— Anonymous (cold-start) —"] + [str(v) for v in top_visitors]
        visitor_sel = st.selectbox(
            "👤 Visitor ID",
            options=visitor_options,
            index=0,
            help=(
                "Top 300 most-active visitors are listed by total interaction count. "
                "Choose Anonymous to get session-only recommendations with no history required."
            ),
        )
        visitor_id_str = "" if visitor_sel.startswith("—") else visitor_sel

        if visitor_id_str:
            st.caption("✅ Returning visitor — ALS + Session + Content signals will all be used.")
        else:
            st.caption("👁️ Anonymous visitor — session signal or Content-Based will drive recommendations.")

        st.markdown("")

        top_items = load_top_item_ids()
        session_sel = st.multiselect(
            "🛒 Browsing Session (items viewed this visit)",
            options=top_items,
            default=[],
            help=(
                "Select items the visitor browsed in this session, most recent last. "
                "Session-Based and Hybrid models use this directly."
            ),
        )
        if session_sel:
            st.caption(f"📦 {len(session_sel)} item(s) selected in session.")
        else:
            st.caption("No session items — Visitor ID alone will drive recommendations.")

        st.divider()

        available = _model_available(model_choice)
        run_btn = st.button("▶ Get Recommendations", type="primary", use_container_width=True)

        st.markdown(f"**Model:** `{model_choice}`  \n**Top-K:** `{top_k}`")
        if available:
            st.success("Model loaded ✅")
        else:
            st.warning("⚠️ Model file not found. Run `train.py` to generate model artefacts.")

    with col_output:
        st.markdown("#### 🎯 Recommended Items")

        if run_btn:
            visitor_id = int(visitor_id_str) if visitor_id_str.strip() else None
            session_items = [int(x) for x in session_sel]

            if not visitor_id and not session_items:
                st.warning("⚠️ Please select at least a Visitor ID or one session item.")
            elif not available:
                st.info("**Demo mode** — model not loaded. Showing random placeholder items.")
                rng  = np.random.default_rng(seed=42)
                recs = rng.integers(100_000, 500_000, size=top_k).tolist()
                item_df = load_item_features()
                st.dataframe(_enrich_items(recs, item_df), use_container_width=True, hide_index=True)
            else:
                with st.spinner("Generating recommendations…"):
                    recs = _get_recommendations(model_choice, visitor_id, session_items, top_k)

                if recs:
                    item_df = load_item_features()
                    display_df = _enrich_items(recs, item_df)
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Items returned", len(recs))
                    avail_count = (display_df["Available"] == "Yes").sum() if "Available" in display_df.columns else "—"
                    m2.metric("In stock", avail_count)
                    n_price = (display_df["Price"] != "—").sum() if "Price" in display_df.columns else 0
                    m3.metric("Have price data", n_price)

                    st.caption(
                        "**How to read this table:** Rank 1 = strongest recommendation. "
                        "Relevance Score is normalised 0–1 (higher = stronger signal from the model). "
                        "Category Depth = hierarchy level (0 = top-level, e.g. Electronics; 2 = sub-sub-category). "
                        "Available = whether the item was last seen marked as in-stock."
                    )
                else:
                    st.info(
                        "No recommendations returned. This can happen when the visitor has no "
                        "matching history in the model. Try a different visitor or add session items."
                    )
        else:
            st.markdown(
                "<div style='background:#f0f4ff;border-radius:10px;padding:2rem;text-align:center;color:#444;'>"
                "<h4>👈 Fill in the inputs and click <b>▶ Get Recommendations</b></h4>"
                "<p style='font-size:0.9rem;'>"
                "Pick a visitor and/or session items on the left.<br>"
                "The model will return the top-K products most likely to be purchased next."
                "</p>"
                "</div>",
                unsafe_allow_html=True,
            )

    with st.expander("ℹ️ How do the 4 models work?"):
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(
            "**🔀 Hybrid**\n\n"
            "Blends all three models with weighted scores. Best for most situations — "
            "degrades gracefully when history is sparse."
        )
        c2.markdown(
            "**🤝 ALS (Collaborative Filtering)**\n\n"
            "Matrix factorisation on user-item interactions. "
            "Learns user 'taste' — finds users who bought similar things."
        )
        c3.markdown(
            "**🏷️ Content-Based (TF-IDF)**\n\n"
            "Finds items with similar metadata (category, price, availability). "
            "Works even for brand-new items with no purchase history."
        )
        c4.markdown(
            "**⚡ Session-Based (Item-KNN)**\n\n"
            "Co-occurrence patterns from 386k training sessions. "
            "Requires no user history — works from session alone."
        )

# ===========================================================================
# TAB 2 — MODEL METRICS
# ===========================================================================
with tab_metrics:

    st.markdown(
        "### Offline Evaluation Results\n"
        "All models were evaluated on a held-out test set constructed from the most recent "
        "sessions (leave-last-session-out split). Metrics measure how often the true next "
        "purchased item appeared in the top-K recommendations list."
    )

    eval_df = load_eval_report()

    if eval_df is None:
        st.warning(
            "⚠️ `outputs/reports/evaluation_report.csv` not found.  \n"
            "Run `python src/training/run_evaluation.py` to generate it."
        )
    else:
        k_values = sorted(eval_df["k"].unique()) if "k" in eval_df.columns else []
        col_k, _ = st.columns([1, 3])
        with col_k:
            selected_k = st.selectbox(
                "Evaluate at K =",
                options=k_values,
                index=len(k_values) - 1 if k_values else 0,
                help=(
                    "K = number of recommendations shown to the user. "
                    "Hit Rate@10 means: did the correct item appear in the top 10?"
                ),
            )

        filtered = eval_df[eval_df["k"] == selected_k].copy() if "k" in eval_df.columns else eval_df.copy()

        METRIC_DESCS = {
            "hit_rate":  ("Hit Rate",   "Fraction of users whose next item was found in top-K. **Higher = better.**"),
            "ndcg":      ("NDCG",       "Rewards finding the item at a higher rank. **Higher = better.**"),
            "mrr":       ("MRR",        "Mean Reciprocal Rank — 1 / rank of first match. **Higher = better.**"),
            "precision": ("Precision",  "What fraction of recommended items were relevant. **Higher = better.**"),
            "recall":    ("Recall",     "What fraction of all relevant items were captured. **Higher = better.**"),
            "coverage":  ("Coverage",   "Share of catalogue ever recommended — diversity signal. **Higher = more diverse.**"),
            "novelty":   ("Novelty",    "Avg. log-popularity of recommended items. **Higher = more niche / less obvious.**"),
        }

        metric_cols = [
            c for c in filtered.columns
            if c not in ("model", "k") and pd.api.types.is_numeric_dtype(filtered[c])
        ]

        st.markdown("#### 📋 All Results")
        st.dataframe(
            filtered.drop(columns=["k"], errors="ignore").set_index("model")
            .style.highlight_max(axis=0, props="background-color:#d4f1d4; font-weight:bold;"),
            use_container_width=True,
        )
        st.caption("🟢 Green highlight = best value for that metric. ALS and Content-Based show 0.0 due to temporal cold-start in the test set — see the explanation below.")

        st.divider()
        st.markdown("#### 📊 Visual Comparison")
        st.caption(
            "Each chart shows one metric across all four models. "
            "Taller bar = better performance on that aspect of recommendation quality."
        )

        primary_metrics = [m for m in ("hit_rate", "ndcg", "mrr") if m in metric_cols]
        if primary_metrics:
            cols = st.columns(len(primary_metrics))
            for col, metric in zip(cols, primary_metrics):
                label, desc = METRIC_DESCS.get(metric, (metric, ""))
                with col:
                    st.markdown(f"**{label} @ {selected_k}**")
                    st.bar_chart(filtered.set_index("model")[[metric]], use_container_width=True, color="#4c9be8")
                    st.caption(desc)

        secondary_metrics = [m for m in ("coverage", "novelty") if m in metric_cols]
        if secondary_metrics:
            st.markdown("#### Diversity & Novelty")
            cols2 = st.columns(len(secondary_metrics))
            for col, metric in zip(cols2, secondary_metrics):
                label, desc = METRIC_DESCS.get(metric, (metric, ""))
                with col:
                    st.markdown(f"**{label}**")
                    st.bar_chart(filtered.set_index("model")[[metric]], use_container_width=True, color="#8e6bbf")
                    st.caption(desc)

        st.divider()
        with st.expander("🔍 Why do ALS and Content-Based show 0.0? (click to understand)"):
            st.markdown(
                """
**Root cause: temporal cold-start in the test set.**

The test set is built from the *most recent* sessions in the dataset. 
Users in that time window were not seen during training — so ALS (which needs a user embedding) 
and Content-Based (which needs a session context) score 0.0. This is realistic: 
in production, ~70% of visitors are cold-start.

**Why does Session-Based perform best?**

Item-KNN only needs the items in the *current session* — it has no cold-start problem. 
A **Hit Rate of 11.9%** means: for roughly 1 in 8 visitors, the item they eventually purchased 
appeared in the top-10 list, using *only* what they browsed in that session.

**What about diversity metrics?**

- **Coverage** — higher means the model is recommending across a wider range of the catalogue 
  (avoids a "popularity trap" where only bestsellers get recommended).
- **Novelty** — higher means the model surfaces niche/less-popular items rather than just the obvious ones.
                """
            )

# ===========================================================================
# TAB 3 — DATA EXPLORER
# ===========================================================================
with tab_data:

    st.markdown(
        "### Dataset & Feature Explorer\n"
        "Browse the engineered features derived from the RetailRocket dataset. "
        "These are the inputs used to train the recommendation models."
    )

    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("Events",        "2.75 M")
    s2.metric("Visitors",      "1.41 M")
    s3.metric("Items",         "417 k")
    s4.metric("Sessions",      "1.73 M")
    s5.metric("Cold-Start",    "~71 %",   help="Visitors with ≤ 1 interaction")
    s6.metric("Purchase Rate", "0.5 %",   help="Transaction events as % of all events")

    st.divider()

    col_user, col_item = st.columns(2, gap="large")

    with col_user:
        st.markdown("#### 👤 User Features")
        st.caption(
            "Each row is one visitor. Features capture how active they are, "
            "how many unique items they viewed, and how likely they are to purchase."
        )
        user_df = load_user_features()
        if user_df is not None:
            st.dataframe(user_df, use_container_width=True, hide_index=False)

            with st.expander("📊 Interaction Score Distribution"):
                st.caption(
                    "**What this shows:** Number of users at each total interaction score. "
                    "Score = sum of (view×1 + addtocart×5 + transaction×10). "
                    "The spike near 0 confirms that most visitors have very few interactions — "
                    "the fundamental sparsity problem this system addresses. "
                    "Clipped at the 95th percentile to remove extreme outliers."
                )
                if "weighted_interaction_score" in user_df.columns:
                    clipped = user_df["weighted_interaction_score"].clip(
                        upper=user_df["weighted_interaction_score"].quantile(0.95)
                    )
                    st.bar_chart(clipped.value_counts().sort_index(), use_container_width=True, color="#4c9be8")

            with st.expander("📊 Cold-Start vs. Warm Users"):
                st.caption(
                    "**What this shows:** Split between cold-start visitors (≤ 1 total event) "
                    "and warm visitors (> 1 event). Cold-start users cannot be served "
                    "by Collaborative Filtering — they rely on Session-Based or Content-Based models."
                )
                if "is_cold_start" in user_df.columns:
                    cs_counts = user_df["is_cold_start"].map(
                        {True: "Cold-Start (<=1 event)", False: "Warm (>1 event)"}
                    ).value_counts()
                    st.bar_chart(cs_counts, use_container_width=True, color="#e8844c")
        else:
            st.info(
                "⚠️ User feature data not found.  \n"
                "Run notebook `02_feature_engineering.ipynb` to generate `data/processed/user_features.csv`."
            )

    with col_item:
        st.markdown("#### 🏷️ Item Features")
        st.caption(
            "Each row is one product. Features capture category placement, "
            "price band, and current stock availability."
        )
        item_df = load_item_features()
        if item_df is not None:
            st.dataframe(item_df.head(500), use_container_width=True, hide_index=False)

            with st.expander("📊 Category Depth Distribution"):
                st.caption(
                    "**What this shows:** How many items sit at each level of the product category hierarchy. "
                    "Depth 0 = top-level category (e.g. 'Electronics'). "
                    "Depth 1 = sub-category (e.g. 'Laptops'). "
                    "Depth 2 = sub-sub-category. "
                    "Most items sit at depth 1–2, meaning the hierarchy is relatively shallow — "
                    "so category is a useful but not deeply hierarchical feature."
                )
                if "category_depth" in item_df.columns:
                    depth_col = item_df["category_depth"].dropna()
                elif "categoryid" in item_df.columns:
                    depths = _load_category_depths()
                    depth_col = item_df["categoryid"].map(depths).dropna()
                else:
                    depth_col = None

                if depth_col is not None and len(depth_col) > 0:
                    depth_counts = depth_col.astype(int).value_counts().sort_index()
                    depth_counts.index = [f"Depth {d}" for d in depth_counts.index]
                    st.bar_chart(depth_counts, use_container_width=True, color="#4c9be8")
                else:
                    st.info("Category depth data not available (category_tree.csv required).")

            with st.expander("📊 Item Availability Breakdown"):
                st.caption(
                    "**What this shows:** Share of items marked as available vs. unavailable "
                    "in the most recent property snapshot. In a production system, unavailable "
                    "items would be filtered out of recommendations at serving time."
                )
                if "available" in item_df.columns:
                    avail_counts = item_df["available"].map(
                        {True: "Available", False: "Unavailable"}
                    ).value_counts()
                    st.bar_chart(avail_counts, use_container_width=True, color="#54b88e")
        else:
            st.info(
                "⚠️ Item feature data not found.  \n"
                "Run notebook `02_feature_engineering.ipynb` to generate `data/processed/item_features.csv`."
            )

    st.divider()

    st.markdown("#### 🔻 Purchase Funnel")
    st.caption(
        "**What this shows:** The share of each event type across all 2.75M events. "
        "The steep drop-off from Views to Purchases is the core challenge. "
        "With only 0.5% of events being transactions, models must learn from weak implicit signals "
        "(views, add-to-cart) rather than explicit purchases."
    )
    funnel_data = pd.DataFrame({
        "Event Type": ["Views", "Add-to-Cart", "Purchases"],
        "Share (%)":  [95.8,     2.7,            0.5],
    }).set_index("Event Type")
    st.bar_chart(funnel_data, use_container_width=True, color="#e8714c")
    st.caption(
        "💡 **Implication:** Interaction weights (view=1, addtocart=5, transaction=10) "
        "amplify the rare but high-value purchase signal so the models don't ignore it."
    )

    st.divider()

    st.markdown("#### 💡 Key Findings from EDA")
    i1, i2, i3 = st.columns(3)
    with i1:
        st.info(
            "**📉 Extreme Sparsity**\n\n"
            ">70% of visitors have ≤ 3 events total. Pure collaborative filtering "
            "would fail for the majority of users — hence the session-based "
            "and content-based fallbacks."
        )
    with i2:
        st.info(
            "**🕐 Peak Hours: 17–21h**\n\n"
            "User activity peaks in the evening. "
            "Time-of-day is a candidate feature for future re-ranking "
            "(not yet wired into the conversion ranker)."
        )
    with i3:
        st.info(
            "**📦 Metadata Coverage Gap**\n\n"
            "~230k items have metadata but no purchase events. "
            "Content-Based filtering ensures these cold items "
            "can still be recommended when contextually relevant."
        )

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.82rem;'>"
    "Built by <b>Biplob Gon</b> · "
    "<a href='https://github.com/biplobgon'>GitHub</a> · "
    "<a href='https://linkedin.com/in/biplobgon'>LinkedIn</a> · "
    "RetailRocket Dataset (Jun–Sep 2015) · Python 3.14 · Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
