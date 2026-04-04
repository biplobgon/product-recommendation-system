# 🛍️ Product Recommendation System

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Recommender](https://img.shields.io/badge/System-Hybrid_Recommendation-green)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit)
![Docker](https://img.shields.io/badge/Container-Docker-2496ED?logo=docker)
![HuggingFace](https://img.shields.io/badge/Models-HuggingFace_Hub-FFD21E?logo=huggingface&logoColor=black)
![Dataset](https://img.shields.io/badge/Dataset-RetailRocket-lightgrey)
![Events](https://img.shields.io/badge/Events-2.75M-yellow)
![Models](https://img.shields.io/badge/Models-4_Hybrid_Stack-blueviolet)

---

## 🚀 Try the Live App

<div align="center">

### [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://recomsys.streamlit.app/)

**[https://recomsys.streamlit.app/](https://recomsys.streamlit.app/)**

*Get real-time product recommendations · Compare 4 ML models · Explore the dataset*

</div>

---

<!--═══════════════════════════════════════════════════════════╗
    PART 1 — PRODUCT LINKS
╚══════════════════════════════════════════════════════════════-->

## 🔗 Product Links

| Surface | Link | Notes |
|---|---|---|
| 🌐 **Live Streamlit App** | [recomsys.streamlit.app](https://recomsys.streamlit.app/) | **Deployed on Streamlit Community Cloud** — recommendations, metrics & data explorer |
| 🤗 **Model Repository** | [biplobgon/product-recommendation-data](https://huggingface.co/datasets/biplobgon/product-recommendation-data) | HF Hub dataset repo hosting 4 model PKLs + processed data (~830 MB) |
| 🐳 **HF Docker Space** | [biplobgon/product-recommendation-system](https://huggingface.co/spaces/biplobgon/product-recommendation-system) | Alternate deployment via Docker on HF Spaces |
| **FastAPI Service** | [http://localhost:8000](http://localhost:8000) | Local REST endpoints — `/recommend/{visitor_id}`, `/similar/{item_id}`, `/popular` |
| **FastAPI Docs (Swagger)** | [http://localhost:8000/docs](http://localhost:8000/docs) | Auto-generated OpenAPI spec |

> **Run locally:**
> ```bash
> uvicorn src.app.api:app --host 0.0.0.0 --port 8000 --reload
> streamlit run src/app/dashboard.py --server.port 8501
> ```

---

<!--═══════════════════════════════════════════════════════════╗
    PART 2 — EXECUTIVE SUMMARY
╚══════════════════════════════════════════════════════════════-->

## 📌 Overview

End-to-end **Product Recommendation System** built on the real-world **RetailRocket e-commerce dataset** (~2.75 M user interaction events). The project covers the full ML lifecycle — EDA → feature engineering → model training → offline evaluation → REST API + interactive dashboard — using a **4-model hybrid architecture** that handles both cold-start and warm-start users.

> **Core prediction goal**: _Given a visitor's browsing session and historical interactions, predict the next N products they are most likely to purchase._

---

## 🎯 Executive Summary

### The Problem

E-commerce recommendation engines face a brutal sparsity problem. In this dataset, **>70 % of visitors have ≤ 3 interactions** and transactions represent only **0.5 %** of all events. A single model cannot solve this:

- Pure **Collaborative Filtering** fails for cold-start users (most of them).
- Pure **Content-Based** filtering ignores rich co-purchase signals.
- Pure **Session-Based** models lose long-term preference memory.

### The Solution

A **hybrid stack of four complementary models** working together — each covering the blind spots of the others:

| Model | Covers |
|---|---|
| ALS (Collaborative Filtering) | Warm users with ≥ 2 interactions — taste matching |
| TF-IDF Content-Based | Cold items and new visitors — property similarity |
| Item-KNN Session-Based | All visitors — within-session sequential intent |
| Hybrid Blender | Weighted merge of all three signals into top-K results |

### Results at a Glance

| Model | Hit Rate@10 | NDCG@10 | MRR@10 |
|---|---|---|---|
| ALS (Collaborative Filtering) | 0.000 | 0.000 | 0.000 |
| Content-Based (TF-IDF) | 0.000 | 0.000 | 0.000 |
| **Session-Based (Item-KNN)** | **0.119** | **0.058** | **0.039** |
| **Hybrid** | **0.115** | **0.056** | **0.038** |

> ALS and Content-Based register 0.0 on this evaluation because the test set is constructed from the most recent sessions — users in that window are not present in training (temporal cold-start). This is expected and realistic. The session model — which requires no user history — is the strongest performer.

### Why This Matters for a Business

- **Personalisation from session 1** — session-based model requires zero user history.
- **Full catalogue coverage** — content-based ensures every item (even those with no purchase history) can surface.
- **Modular, swappable stack** — each model is independently loadable and overridable via API parameter.
- **Sub-100 ms serving** — all models are loaded in-memory; no database joins at request time.

---

<!--═══════════════════════════════════════════════════════════╗
    PART 3 — TECHNICAL DEEP DIVE
╚══════════════════════════════════════════════════════════════-->

## 🔬 Technical Deep Dive

---

### 📊 Dataset

**Source**: [RetailRocket E-commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) (Jun–Sep 2015)

| File | Rows | Key stat |
|---|---|---|
| `events.csv` | ~2.75 M | view 95.8 %, addtocart 2.7 %, transaction 0.5 % |
| `item_properties_part1.csv` | ~11 M | ~185k unique items with metadata |
| `item_properties_part2.csv` | ~9.2 M | Extends Part 1; combined ~20 M property records |
| `category_tree.csv` | ~1,600 nodes | Depth 2–3 dominant; max depth 5 |

**Interaction signal weights used in training**: view = 1 · addtocart = 5 · transaction = 10

---

### 🧠 System Architecture

```
                ┌────────────────────────────────────────────┐
                │              Incoming Request               │
                │   visitor_id  ·  session_items  ·  top_k   │
                └──────────────────┬─────────────────────────┘
                                   │
         ┌─────────────────────────┼──────────────────────────┐
         ▼                         ▼                          ▼
  ┌─────────────┐         ┌───────────────┐         ┌──────────────────┐
  │  ALS (CF)   │         │  TF-IDF (CB)  │         │  Item-KNN (SB)   │
  │  weight=0.4 │         │  weight=0.2   │         │  weight=0.4      │
  │  warm users │         │  cold items   │         │  all visitors    │
  └──────┬──────┘         └──────┬────────┘         └────────┬─────────┘
         └──────────────────────▼──────────────────────────┘
                        ┌──────────────────┐
                        │  Weighted Blend  │
                        │  + min-max norm  │
                        └────────┬─────────┘
                                 ▼
                       Top-K Recommendations
```

**Session boundary**: 1-hour inactivity gap · Max sequence length: 20 items · Temporal train/test split: leave-last-session-out

---

### 🏗️ Model Implementations

#### ALS — Collaborative Filtering (`src/models/collaborative_filtering.py`)
- Pure **numpy / scipy** implementation of the Hu-Koren-Volinsky (2008) Alternating Least Squares algorithm — **no external library dependency** (the `implicit` package doesn't support Python 3.14).
- Weighted implicit feedback matrix with `factors=32, iterations=10, regularisation=0.01`.
- Trained on 31,880 warm users × 344,728 weighted interactions.

#### Content-Based — TF-IDF (`src/models/content_based.py`)
- Scikit-learn `TfidfVectorizer` over concatenated item property strings (categoryid + price bucket + availability).
- `max_features=500` to keep the matrix (417k × 500) tractable.
- Cosine similarity at query time; no pre-computed pairwise matrix (memory efficient).

#### Session-Based — Item-KNN (`src/models/session_based.py`)
- **Co-occurrence matrix** over 386,099 training sequences, recency-weighted (more recent sessions count more).
- No PyTorch dependency — pure numpy co-occurrence counting with exponential recency decay.
- At query time: sums co-occurrence scores for all items in the current session; returns top-K.

#### Hybrid (`src/models/hybrid.py`)
- Calls all three models, normalises scores to [0, 1], applies configurable weights, and merges.
- Gracefully degrades: if ALS has no embedding for a new user, only CB + Session contribute.

---

### ⚙️ Feature Engineering

| Feature Set | File | Key columns |
|---|---|---|
| User features | `data/processed/user_features.parquet` | `n_views`, `n_addtocart`, `n_transactions`, `n_unique_items`, `conversion_rate`, `is_cold_start` |
| Item features | `data/processed/item_features.parquet` | `categoryid`, `category_depth`, `price`, `available` |
| Session features | `data/processed/session_features.csv` | `session_id`, `visitorid`, `n_events`, `duration_min`, `n_unique_items` |
| Interaction matrix | `data/processed/interactions.csv` | `visitorid`, `itemid`, `score` |
| TF-IDF matrix | `data/processed/tfidf_matrix.npz` | Sparse (417,053 × 500) |

Generated by running `notebooks/02_feature_engineering.ipynb`.

---

### 📈 Evaluation Metrics

| Metric | What it measures |
|---|---|
| **Hit Rate@K** | Fraction of test users whose ground-truth item appears in top-K |
| **NDCG@K** | Ranks the true item higher → higher reward |
| **MRR@K** | Mean reciprocal rank of first relevant item |
| **Precision@K / Recall@K** | Standard IR metrics at cutoff K |
| **Coverage** | Fraction of item catalogue ever recommended (diversity signal) |
| **Novelty** | Avg. log-popularity of recommended items (higher = more niche) |

Results saved to `outputs/reports/evaluation_report.csv`. Run evaluation independently via:
```bash
python src/training/run_evaluation.py
```

---

### 🔌 API Reference

```bash
uvicorn src.app.api:app --host 0.0.0.0 --port 8000 --reload
```

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health + loaded models |
| `GET` | `/recommend/{visitor_id}` | Personalised top-K (all models) |
| `POST` | `/recommend/session` | Session-only recommendations (no history needed) |
| `GET` | `/similar/{item_id}` | Items similar to a given item (TF-IDF) |
| `GET` | `/popular` | Global popularity fallback |

**Example**:
```bash
curl "http://localhost:8000/recommend/12345?session_items=101,202&top_k=10"
```
```json
{
  "visitor_id": 12345,
  "recommended_items": [876, 205, 934, 412, 778, 66, 543, 188, 301, 407],
  "model": "hybrid",
  "latency_ms": 38
}
```

---

### 📁 Project Structure

```
product-recommendation-system/
│
├── data/
│   ├── raw/                            # Original RetailRocket CSVs (gitignored)
│   │   ├── events.csv
│   │   ├── category_tree.csv
│   │   ├── item_properties_part1.csv
│   │   └── item_properties_part2.csv
│   └── processed/                      # Generated feature files (gitignored — large)
│       ├── user_features.parquet       # 1,407,580 users × 12 features
│       ├── item_features.parquet       # 417,053 items × 4 features (incl. category_depth)
│       ├── interactions.csv            # 2,145,179 user-item weighted interactions
│       ├── session_features.csv        # 1,726,714 sessions × 9 features
│       ├── session_sequences.csv       # 386,099 training sequences
│       ├── tfidf_matrix.npz            # Sparse TF-IDF (417k × 500)
│       ├── tfidf_item_ids.csv          # Item ID → TF-IDF row index mapping
│       └── tfidf_vectorizer.pkl        # Fitted sklearn TfidfVectorizer
│
├── notebooks/
│   ├── eda.ipynb                       # 01 — Full EDA (25 cells, 25 visualisations)
│   ├── 02_feature_engineering.ipynb    # 02 — User / item / session feature generation
│   ├── 03_model_training.ipynb         # 03 — Train ALS, CB, Session, Hybrid
│   └── 04_model_evaluation.ipynb       # 04 — Offline metrics + comparison charts
│
├── src/
│   ├── utils/
│   │   ├── config.py                   # YAML loader with dot-notation access
│   │   └── logger.py                   # Centralised logging
│   │
│   ├── features/
│   │   ├── user_features.py            # Visitor-level aggregates
│   │   ├── item_features.py            # Item metadata + price/availability parsing
│   │   └── session_features.py         # Session segmentation + sequence building
│   │
│   ├── models/
│   │   ├── collaborative_filtering.py  # ALSRecommender (pure numpy/scipy)
│   │   ├── content_based.py            # ContentBasedRecommender (sklearn TF-IDF)
│   │   ├── session_based.py            # SessionBasedRecommender (Item-KNN, numpy)
│   │   └── hybrid.py                   # HybridRecommender (weighted blend)
│   │
│   ├── training/
│   │   ├── train.py                    # Full end-to-end training pipeline
│   │   ├── resume_training.py          # Skip CB re-training; reload saved ALS+CB
│   │   ├── run_evaluation.py           # Evaluate all models → evaluation_report.csv
│   │   └── evaluate.py                 # HR@K, NDCG@K, MRR@K, Coverage, Novelty
│   │
│   └── app/
│       ├── api.py                      # FastAPI service (5 endpoints)
│       ├── dashboard.py                # Streamlit dashboard (3 tabs)
│       └── hf_loader.py                # Downloads model/data artefacts from HF Hub
│
├── outputs/
│   ├── models/                         # Serialised PKL artefacts (gitignored — large)
│   │   ├── als_model.pkl               # 18.4 MB  — hosted on HF Hub
│   │   ├── content_based_model.pkl     # 354.7 MB — hosted on HF Hub
│   │   ├── session_based_model.pkl     # 14.7 MB  — hosted on HF Hub
│   │   └── hybrid_model.pkl            # 388.1 MB — hosted on HF Hub
│   └── reports/
│       └── evaluation_report.csv       # All model metrics across K=5,10
│
├── configs/
│   ├── model_config.yaml               # Hyperparameters & data paths
│   └── pipeline_config.yaml            # Training pipeline, MLflow, retrain schedule
│
├── app.py                              # HF Spaces / Streamlit Cloud entry-point shim
├── Dockerfile                          # Docker image for HF Spaces deployment
├── requirements.txt                    # Slim inference deps (no TF/PyTorch)
├── requirements-spaces.txt             # Alias — same slim deps
├── .streamlit/secrets.toml.example    # Template for local Streamlit secrets
├── assets/                             # Static images for README
├── .gitignore
└── README.md
```

---

### ☁️ Deployment

The app is deployed on two platforms simultaneously. Model artefacts (~830 MB total) live on **Hugging Face Hub** and are downloaded automatically on first startup — no Docker image bloat, no git LFS costs.

#### Streamlit Community Cloud *(primary)*
- **URL**: [recomsys.streamlit.app](https://recomsys.streamlit.app/)
- Deploys directly from the `main` branch of this GitHub repo.
- On cold start, `src/app/hf_loader.py` downloads the 4 model PKLs from `biplobgon/product-recommendation-data` and caches them for the session (~2–4 min first load).

#### HF Spaces — Docker *(alternate)*
- **URL**: [huggingface.co/spaces/biplobgon/product-recommendation-system](https://huggingface.co/spaces/biplobgon/product-recommendation-system)
- Uses the `Dockerfile` at the repo root; runs Streamlit on port 7860.
- Same `hf_loader.py` mechanism for model downloads.

#### Model Storage (Hugging Face Hub)
All large artefacts are stored at [`biplobgon/product-recommendation-data`](https://huggingface.co/datasets/biplobgon/product-recommendation-data):

| Path in repo | Size |
|---|---|
| `models/als_model.pkl` | 18.4 MB |
| `models/session_based_model.pkl` | 14.7 MB |
| `models/content_based_model.pkl` | 354.7 MB |
| `models/hybrid_model.pkl` | 388.1 MB |
| `processed/user_features.parquet` | 28.8 MB |
| `processed/item_features.parquet` | 3.5 MB |
| `processed/interactions.csv` | 36.4 MB |

#### Local development
```bash
# Copy the secrets template and fill in your HF token
cp .streamlit/secrets.toml.example .streamlit/secrets.toml

# Install slim inference deps
pip install -r requirements.txt

# Run dashboard (models load from local disk if present, else HF Hub)
streamlit run src/app/dashboard.py

# Run API
uvicorn src.app.api:app --reload
```

---

### 🔄 End-to-End Pipeline

```
Raw CSVs → EDA (01) → Feature Engineering (02)
                              │
              ┌───────────────▼──────────────────┐
              │  data/processed/                  │
              │  user_features  ·  item_features  │
              │  tfidf_matrix   ·  interactions   │
              │  session_sequences                │
              └───────────────┬──────────────────┘
                              │
              ┌───────────────▼──────────────────┐
              │  Training (train.py)              │
              │  ALS → CB (40 min) → Session      │
              │  → Hybrid                         │
              └───────────────┬──────────────────┘
                              │
              ┌───────────────▼──────────────────┐
              │  Evaluation (run_evaluation.py)   │
              │  HR@K · NDCG@K · MRR@K           │
              │  Coverage · Novelty               │
              └───────────────┬──────────────────┘
                              │
              ┌───────────────▼──────────────────────────────┐
              │  Serving                                      │
              │  FastAPI :8000  ·  Streamlit :8501 (local)   │
              │  Streamlit Cloud (recomsys.streamlit.app)     │
              │  HF Docker Space                              │
              └───────────────────────────────────────────────┘
```

**Quick start (full retrain from scratch):**
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull raw data (or place CSVs manually in data/raw/)
cp .env.example .env   # set GCS_BUCKET_NAME
python src/gcs_loader.py

# 3. Run feature engineering
jupyter nbconvert --to notebook --execute notebooks/02_feature_engineering.ipynb

# 4. Train models (full run ~50 min; CB is the bottleneck)
python src/training/train.py

# 5. Or, if models are already saved, run a fast retrain (session only, ~30 s)
python src/training/resume_training.py

# 6. Evaluate
python src/training/run_evaluation.py

# 7. Serve
uvicorn src.app.api:app --reload
streamlit run src/app/dashboard.py
```

---

### 🪲 Pitfalls Faced & How They Were Solved

| # | Pitfall | Root Cause | Solution |
|---|---|---|---|
| 1 | **OOM crash during TF-IDF** | `max_features=5000` on 417k items → 2 GB dense matrix | Reduced to `max_features=500` |
| 2 | **`implicit` / `torch` missing on Python 3.14** | No wheels available for Python 3.14 | Replaced ALS with pure numpy/scipy, replaced GRU4Rec with Item-KNN |
| 3 | **Duplicate class in `session_based.py`** | Old GRU4Rec class at line 162 silently overrode the new Item-KNN class | Truncated file to 160 lines to remove the ghost class |
| 4 | **`user_features.py` OOM from `mode()` lambda** | `.apply(lambda x: x.mode())` on 2.75 M rows with groupby is O(n²) | Rewrote with vectorised `pandas.groupby` aggregates — 16× faster |
| 5 | **`pyarrow` incompatible with Python 3.14** | `ArrowKeyError: No type extension named arrow.py_extension_type` | Switched notebook persist step from `.to_parquet()` → `.to_csv()` |
| 6 | **Dashboard showing wrong model files** | `MODEL_FILES` dict still referenced old filenames (`als_model.npz`, `gru4rec_model.pt`) | Updated all keys to actual `.pkl` filenames |
| 7 | **`_enrich_items` column misalignment** | Models return `list[tuple[int, float]]`; dashboard iterated as plain ints | Fixed unpacking to `(item_id, score)` and added a Score column |
| 8 | **`evaluation_report.csv` cached as `None`** | `@st.cache_data` cached before file existed on first run | Removed the decorator from `load_eval_report` |
| 9 | **Large files rejected by GitHub (>100 MB)** | Model PKLs (338–370 MB) and processed CSVs exceeded GitHub limit | Added to `.gitignore`; removed from git history; hosted on HF Hub |
| 10 | **HF Spaces rejected push (binary files in history)** | `gcloud-installer.exe` and architecture PNG found in git history by HF pre-receive hook | Used `git filter-repo` to purge all 3 files from every past commit; force-pushed clean branch |
| 11 | **Streamlit SDK removed from HF Spaces** | HF now only supports Gradio, Docker, Static SDKs | Switched to Docker SDK; added `Dockerfile` running Streamlit on port 7860 |

---

### 🚀 Future Enhancements

| Priority | Item |
|---|---|
| 🔴 High | **MLflow experiment tracking** — config exists; calls not yet wired into `train.py` |
| 🔴 High | **ConversionRanker (LightGBM)** — train on session features × `is_purchase` target to re-rank hybrid output |
| 🟡 Medium | **Real-time event streaming** — Kafka consumer to update session model with live clicks |
| 🟡 Medium | **A/B testing framework** — statistical comparison of hybrid vs. popularity baseline |
| 🟡 Medium | **Time-of-day feature** — inject hour-of-day signal (peak 17–21h) into re-ranker |
| 🟢 Low | **BERT4Rec / SASRec** — Transformer-based sequential model to replace Item-KNN |
| 🟢 Low | **NeuMF** — Neural Collaborative Filtering for richer user/item embeddings vs. ALS |
| 🟢 Low | **GNN over category hierarchy** — co-purchase + category graph for structural similarity |
| 🟢 Low | **Reinforcement learning bandit** — explore-exploit policy for dynamic recommendation |

---
## ☁️ Google Cloud Storage Setup

Raw files can be pulled from a GCS bucket automatically.

```bash
gcloud auth application-default login
cp .env.example .env   # set GCS_BUCKET_NAME
python src/gcs_loader.py
```

| Blob | Description |
|---|---|
| `events.csv` | User interaction events |
| `category_tree.csv` | Product category hierarchy |
| `item_properties_part1.csv` | Item metadata part 1 |
| `item_properties_part2.csv` | Item metadata part 2 |

---

## 👤 Author

**Biplob Gon** · Data Scientist | AI/ML | Recommender Systems

[![GitHub](https://img.shields.io/badge/GitHub-biplobgon-black?logo=github)](https://github.com/biplobgon)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Biplob%20Gon-blue?logo=linkedin)](https://linkedin.com/in/biplobgon)

---

## ⭐ Found this useful?

Give the repo a ⭐ — it helps others discover it.
