# 🛍️ Product Recommendation System

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Recommender](https://img.shields.io/badge/System-Hybrid_Recommendation-green)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit)
![Dataset](https://img.shields.io/badge/Dataset-RetailRocket-lightgrey)
![Events](https://img.shields.io/badge/Events-2.75M-yellow)
![Models](https://img.shields.io/badge/Models-4_Hybrid_Stack-blueviolet)


---

<!--═══════════════════════════════════════════════════════════╗
    PART 1 — PRODUCT LINKS
╚══════════════════════════════════════════════════════════════-->

## 🔗 Live Demos

| Surface | Link | Notes |
|---|---|---|
| **Streamlit Dashboard** | [http://localhost:8501](http://localhost:8501) | Interactive UI — get personalised recommendations, explore model metrics & data |
| **FastAPI Service** | [http://localhost:8000](http://localhost:8000) | REST endpoints — `/recommend/{visitor_id}`, `/similar/{item_id}`, `/popular` |
| **FastAPI Docs (Swagger)** | [http://localhost:8000/docs](http://localhost:8000/docs) | Auto-generated OpenAPI spec |

> **Run locally in two commands:**
> ```bash
> uvicorn src.app.api:app --host 0.0.0.0 --port 8000 --reload
> streamlit run src/app/dashboard.py --server.port 8501
> ```

---

<!--═══════════════════════════════════════════════════════════╗
    PART 2 — EXECUTIVE SUMMARY
╚══════════════════════════════════════════════════════════════-->

## 📌 Overview

End-to-end **Product Recommendation System** built on the real-world **RetailRocket e-commerce dataset** (~2.75M user interaction events).

The project covers the full ML lifecycle — exploratory data analysis → feature engineering → model training → offline evaluation → serving API — using a **4-model hybrid architecture** that tackles both cold-start and warm-start users.

> **Core prediction goal**: _Given a visitor's browsing session and historical interactions, predict the next N products they are most likely to purchase._

---

## 🎯 Business Problem

| Challenge | Impact |
|---|---|
| Sparse interaction data | >70 % of visitors have ≤3 events → weak collaborative signal |
| Cold-start users | New visitors have zero history → standard CF fails |
| Cold-start items | ~230k items have metadata but no purchase history |
| Conversion gap | Views >95 % but transactions <0.5 % → hard to learn purchase intent |
| Real-time serving | Recommendations must return in <200 ms at request time |

---

## 🚀 Business Impact

- 📈 Increase conversion rate by surfacing purchase-likely items
- 🛒 Improve Average Order Value through contextual upsells
- 🎯 Personalised experience from the first page view (session-based cold-start)
- 🔁 Re-engage returning users with collaborative taste-matching
- 📊 Measurable improvement tracked via Hit Rate, NDCG, and MRR at K

---

## 📊 Dataset — EDA Key Findings

**Source**: [RetailRocket E-commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) (Jun–Sep 2015)

| File | Rows | Key stat |
|---|---|---|
| `events.csv` | ~2.75M | view 95.8 %, addtocart 2.7 %, transaction 0.5 % |
| `item_properties_part1.csv` | ~11M | ~185k unique items with metadata |
| `item_properties_part2.csv` | ~9.2M | Extends Part 1; combined ~20M property records |
| `category_tree.csv` | ~1,600 nodes | Depth 2–3 dominant; max depth 5 |

**Visitor behaviour**
- Median events per visitor: **3** — severe sparsity
- Peak activity: **17:00 – 21:00** daily
- ~50k items have events but no metadata (cold items)
- ~230k items have metadata but no purchase events

**Interaction signal weights used in training**: view = 1 · addtocart = 5 · transaction = 10

---

## 🧠 Hybrid Architecture (4 Models)

```
                ┌─────────────────────────────────────────┐
                │           Incoming Request               │
                │  visitor_id · session_items · context    │
                └────────────────┬────────────────────────┘
                                 │
          ┌──────────────────────┼───────────────────────┐
          ▼                      ▼                       ▼
   ┌─────────────┐      ┌───────────────┐      ┌──────────────────┐
   │  ALS (CF)   │      │  TF-IDF (CB)  │      │  GRU4Rec (SB)    │
   │  weight=0.35│      │  weight=0.25  │      │  weight=0.40     │
   │  warm users │      │  cold items   │      │  all visitors    │
   └──────┬──────┘      └──────┬────────┘      └────────┬─────────┘
          └──────────────────── ▼ ──────────────────────┘
                       ┌────────────────┐
                       │ Weighted blend │
                       │ + min-max norm │
                       └───────┬────────┘
                               ▼
                  ┌────────────────────────┐
                  │  LightGBM Re-ranker    │
                  │  (Conversion Ranker)   │
                  │  purchase probability  │
                  └────────────┬───────────┘
                               ▼
                      Top-K Recommendations
```

| Model | Library | Purpose |
|---|---|---|
| ALS (Collaborative Filtering) | `implicit` | Taste-matching for users with ≥2 interactions |
| TF-IDF Content-Based | `scikit-learn` | Item similarity for cold items / new users |
| GRU4Rec (Session-Based) | `PyTorch` | Sequential pattern from current browsing session |
| LightGBM Conversion Ranker | `lightgbm` | Re-ranks by predicted add-to-cart / purchase probability |

**Session boundary**: 1-hour inactivity gap · Max sequence length: 20 items

---

## 📁 Project Structure

```
product-recommendation-system/
│
├── data/
│   ├── raw/                        # Original CSVs (downloaded from GCS or Kaggle)
│   │   ├── events.csv
│   │   ├── category_tree.csv
│   │   ├── item_properties_part1.csv
│   │   └── item_properties_part2.csv
│   └── processed/                  # Parquet / npz / pkl files from feature engineering
│
├── configs/
│   ├── model_config.yaml           # All model hyperparameters & data paths
│   └── pipeline_config.yaml        # Training pipeline, MLflow, API, retrain schedule
│
├── notebooks/
│   ├── eda.ipynb                   # Notebook 01 — full exploratory data analysis (25 cells)
│   ├── 02_feature_engineering.ipynb# Notebook 02 — user / item / session features
│   ├── 03_model_training.ipynb     # Notebook 03 — train ALS, CB, GRU4Rec, Hybrid
│   └── 04_model_evaluation.ipynb   # Notebook 04 — offline metrics & comparison charts
│
├── src/
│   ├── data_prep.py                # Raw data cleaning & validation
│   ├── create_sample.py            # Stratified sampling for fast iteration
│   ├── gcs_loader.py               # Download raw files from Google Cloud Storage
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py               # YAML loader with dot-notation access
│   │   └── logger.py               # Centralised logging (get_logger)
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── user_features.py        # Per-visitor aggregates + user-item matrix
│   │   ├── item_features.py        # Per-item metadata + TF-IDF matrix
│   │   └── session_features.py     # Session segmentation + GRU4Rec sequences
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── collaborative_filtering.py  # ALSRecommender (implicit)
│   │   ├── content_based.py            # ContentBasedRecommender (sklearn TF-IDF)
│   │   ├── session_based.py            # SessionBasedRecommender (PyTorch GRU)
│   │   ├── conversion_ranker.py        # ConversionRanker (LightGBM)
│   │   └── hybrid.py                   # HybridRecommender (weighted blend)
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py                # End-to-end training pipeline script
│   │   └── evaluate.py             # Offline evaluation (HR@K, NDCG@K, MRR@K, …)
│   │
│   └── app/                        # API & dashboard (FastAPI / Streamlit)
│
├── outputs/
│   ├── figures/                    # Generated EDA & evaluation plots (README inside)
│   ├── models/                     # Serialised model artefacts (README inside)
│   └── reports/                    # evaluation_report.csv & training logs (README inside)
│
├── assets/                         # Static images for README / presentation
├── .env.example                    # Environment variable template
├── requirements.txt
└── README.md
```

> Each `outputs/` sub-directory contains its own `README.md` documenting expected files and schemas.

---

## 🔄 Project Flow

```
Raw CSVs  ──▶  EDA (01)  ──▶  Feature Engineering (02)
                                        │
                          ┌─────────────▼──────────────┐
                          │  data/processed/            │
                          │  user_features.parquet      │
                          │  item_features.parquet      │
                          │  tfidf_matrix.npz           │
                          │  sessions.parquet           │
                          └─────────────┬──────────────┘
                                        │
                          ┌─────────────▼──────────────┐
                          │  Model Training (03)        │
                          │  ALS · CB · GRU4Rec · Hybrid│
                          └─────────────┬──────────────┘
                                        │
                          ┌─────────────▼──────────────┐
                          │  Offline Evaluation (04)    │
                          │  HR@K · NDCG@K · MRR@K     │
                          │  Coverage · Novelty         │
                          └─────────────┬──────────────┘
                                        │
                          ┌─────────────▼──────────────┐
                          │  FastAPI serving            │
                          │  GET /recommend/{user_id}   │
                          └────────────────────────────┘
```

**Temporal split strategy**: last 20 % of events per user held out as test set (leave-last-out).

---

## 📈 Evaluation Metrics

| Metric | What it measures |
|---|---|
| **Hit Rate@K** | Fraction of test users whose ground-truth item appears in top-K |
| **NDCG@K** | Ranks the true item higher → higher reward |
| **MRR@K** | Mean reciprocal rank of first relevant item |
| **Precision@K** | Fraction of K recommendations that are relevant |
| **Recall@K** | Fraction of all relevant items captured in top-K |
| **Coverage** | Fraction of catalogue ever recommended (diversity) |
| **Novelty** | Avg. log-popularity of recommended items (higher = more niche) |

Evaluation results are saved to `outputs/reports/evaluation_report.csv`.

---

## 📊 Key Insights from EDA

1. **Extreme sparsity** — >70 % of visitors have ≤3 events. Pure collaborative filtering would under-serve the majority of users.
2. **Purchase funnel is narrow** — transactions are <0.5 % of events. Weighted implicit feedback (view=1, addtocart=5, transaction=10) is essential to amplify purchase signal.
3. **Session patterns dominate** — users browse in concentrated bursts with clear inactivity gaps. A session-aware GRU model captures within-session intent that ALS cannot.
4. **Metadata coverage gap** — ~230k items have only properties (no interactions). Content-based filtering ensures these items can still be recommended when contextually relevant.
5. **Category hierarchy is shallow** — max depth 5, mode depth 2–3. Item category features are useful but not deeply hierarchical; TF-IDF over property text adds richer signal.
6. **Peak hours 17–21h** — time-of-day is a potential ranking feature for the conversion ranker.

---

## 🔌 API

### Start server

```bash
uvicorn src.app.api:app --reload --host 0.0.0.0 --port 8000
```

### Endpoint

```
GET /recommend/{user_id}?session_items=101,202,303&top_k=10
```

**Response**

```json
{
  "user_id": 12345,
  "recommended_items": [876, 205, 101, 934, 412, 778, 66, 290, 543, 188],
  "model": "hybrid",
  "latency_ms": 42
}
```

---

## 📊 Streamlit Dashboard

```bash
streamlit run src/app/dashboard.py
```

Features:
- Enter visitor ID or paste a browsing session to get live recommendations
- Toggle model (ALS / Content-Based / Session / Hybrid)
- View per-model metric comparison bar charts from `evaluation_report.csv`
- Browse top recommended items with category labels

---

## ☁️ Google Cloud Storage Setup

Raw files can be pulled from a GCS bucket automatically.

```bash
# Authenticate
gcloud auth application-default login

# Copy env template
cp .env.example .env   # set GCS_BUCKET_NAME

# Download all raw files
python src/gcs_loader.py

# Download specific files
python src/gcs_loader.py --files events.csv category_tree.csv
```

| Blob name | Description |
|---|---|
| `events.csv` | User interaction events |
| `category_tree.csv` | Product category hierarchy |
| `item_properties_part1.csv` | Item metadata part 1 |
| `item_properties_part2.csv` | Item metadata part 2 |

---

## ⚙️ How to Run End-to-End

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull raw data (or place CSVs manually in data/raw/)
cp .env.example .env   # set GCS_BUCKET_NAME
python src/gcs_loader.py

# 3. Run the training pipeline (all steps in one script)
python src/training/train.py

# 4. Or step through interactively
#    notebooks/eda.ipynb                    → EDA
#    notebooks/02_feature_engineering.ipynb → Features
#    notebooks/03_model_training.ipynb      → Train models
#    notebooks/04_model_evaluation.ipynb    → Evaluate

# 5. Serve recommendations
uvicorn src.app.api:app --reload
streamlit run src/app/dashboard.py
```

---

## 🔄 CI/CD (GitHub Actions)

Automated pipeline on push to `main`:

1. Install dependencies
2. Run `src/training/train.py`
3. Upload model artefacts to GCS
4. Run evaluation and post metrics summary to PR

MLflow experiment tracking configured in `configs/pipeline_config.yaml` (`mlruns/` directory).

---

## 🧩 Future Steps

| Priority | Enhancement |
|---|---|
| High | Wire **ConversionRanker training** into `03_model_training.ipynb` using session features as X and is_purchase as y |
| High | Add **MLflow logging** calls to `src/training/train.py` (config exists, calls not yet wired) |
| Medium | **Real-time streaming** with Kafka — ingest live click events and update session model |
| Medium | **A/B testing framework** — compare hybrid vs. baseline (popularity) with statistical significance |
| Medium | **Time-of-day feature** — inject hour-of-day (peak 17–21h from EDA) into ConversionRanker |
| Low | **Neural Collaborative Filtering** (NeuMF) or **BERT4Rec** to replace ALS for richer user embeddings |
| Low | **Graph Neural Network** (GNN) over category hierarchy + co-purchase graph |
| Low | **Reinforcement Learning** (bandits) for dynamic explore-exploit recommendation policy |

---

## 👤 Author

**Biplob Gon**
_Data Scientist | AI/ML | Recommender Systems_

[![GitHub](https://img.shields.io/badge/GitHub-biplobgon-black?logo=github)](https://github.com/biplobgon)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Biplob%20Gon-blue?logo=linkedin)](https://linkedin.com/in/biplobgon)

---

## ⭐ If you found this useful

Give this repo a ⭐ to support and share!