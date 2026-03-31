# product-recommendation-system
End-to-end product recommendation system using collaborative filtering, hybrid ML models, and MLOps pipeline with API &amp; dashboard.

# 🛍️ Product Recommendation System (End-to-End ML + MLOps)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Recommender](https://img.shields.io/badge/System-Recommendation-green)
![MLOps](https://img.shields.io/badge/MLOps-GitHub_Actions-orange)
![FastAPI](https://img.shields.io/badge/API-FastAPI-teal)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)

---

## 📌 Overview

End-to-end **Product Recommendation System** built using real-world e-commerce interaction data.

👉 Predicts **what a user is most likely to buy next**  
👉 Combines **Collaborative Filtering + Content-Based + Hybrid approaches**  
👉 Designed with **production-grade ML pipeline + API + dashboard**

---

## 🎯 Business Problem

**Core Question:**
> What products should we recommend to each user to maximize engagement, conversion, and revenue?

### Key Challenges:
- Sparse interaction data  
- Cold-start problem (new users/items)  
- Real-time recommendation requirements  
- Scalability for millions of users  

---

## 🚀 Business Impact

- 📈 Increase conversion rate (CTR uplift)  
- 🛒 Improve Average Order Value (AOV)  
- 🎯 Personalized shopping experience  
- 🔁 Improve retention and engagement  

---

## 📊 Dataset

### Primary Dataset:
- **Retailrocket E-commerce Dataset**
  - User events: clicks, add-to-cart, purchases  
  - Product metadata: item IDs, categories  

### Data Fields:
- `user_id`
- `item_id`
- `event_type`
- `timestamp`

---

## 🏗️ System Architecture

<p align="center">
  <img src="./assets/system_design_architecture.png" width="650"/>
</p>

### 🔄 End-to-End Flow

1. **User Events**
2. **Data Processing**
3. **Feature Engineering**
4. **Model Training**
5. **Model Serving (FastAPI)**
6. **Recommendation Output**
7. **Dashboard (Streamlit)**

---

## 🧠 Recommendation Approaches

### 1. Collaborative Filtering
- User-based similarity  
- Item-based similarity  
- Matrix Factorization (ALS)

### 2. Content-Based Filtering
- Product similarity (TF-IDF / embeddings)  
- Metadata-driven recommendations  

### 3. Hybrid Model (Final)
- Combines collaborative + content features  
- Handles cold-start scenarios  

---

## 📁 Project Structure

```
product-recommendation-system/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── src/
│ ├── data_prep.py
│ ├── feature_engineering.py
│ ├── train_model.py
│ ├── recommend.py
│ ├── evaluate.py
│ │
│ └── app/
│ ├── api.py
│ └── dashboard.py
│
├── outputs/
│ ├── model/
│ ├── plots/
│
├── assets/
│ ├── system_architecture.png
│ └── xgb_model.pkl
│
├── notebooks/
├── presentation/
│
├── README.md
├── requirements.txt
└── .gitignore
```
```markdown id="tip"
👉 Modular structure designed for scalability and easy transition to production environments
```
---

## 📈 Evaluation Metrics

### Offline Metrics:
- Precision@K  
- Recall@K  
- NDCG  

### Business Metrics:
- CTR uplift  
- Conversion rate  
- Revenue per user  

---

## 🔌 API (FastAPI)

### ▶️ Run API

```bash
uvicorn src.app.api:app --reload
```
---

## 📥 Endpoint

### GET /recommend/{user_id}

```json
{
  "user_id": 123,
  "recommended_items": [101, 205, 876]
}
```
---

## 📊 Streamlit Dashboard

### ▶️ Run Dashboard

```bash
streamlit run src/app/dashboard.py
```

### Features:

- 🔮 Personalized recommendations
- 📊 Top-N products
- 📈 Interactive visualization

---

## 📊 Key Insights

- Users with similar behavior patterns purchase similar products
- Item similarity significantly improves recommendation accuracy
- Hybrid models outperform standalone approaches
- Cold-start handled via content-based signals

---

## ☁️ Google Cloud Storage Setup

Raw dataset files are stored in a GCS bucket and downloaded automatically the first time you run the pipeline.

### Prerequisites

1. **Install the GCS dependency** (included in `requirements.txt`):
   ```bash
   pip install google-cloud-storage==2.10.0
   ```

2. **Authenticate** using one of the following methods:
   - *Application Default Credentials* (recommended for local development):
     ```bash
     gcloud auth application-default login
     ```
   - *Service-account key file*: Download a JSON key from the GCP console and set:
     ```bash
     export GCS_CREDENTIALS=/path/to/service-account-key.json
     ```

3. **Configure environment variables** – copy `.env.example` to `.env` and edit as needed:
   ```bash
   cp .env.example .env
   # Edit .env and set GCS_BUCKET_NAME (and optionally GCS_PROJECT_ID / GCS_CREDENTIALS)
   ```

### Download data manually

```bash
# Download all default dataset files into data/raw/
python src/gcs_loader.py

# Download specific files
python src/gcs_loader.py --files events.csv category_tree.csv

# Override bucket and destination
python src/gcs_loader.py --bucket my-bucket --dest /tmp/data
```

### Files expected in the bucket

| Blob name | Description |
|---|---|
| `events.csv` | User interaction events (clicks, add-to-cart, purchases) |
| `category_tree.csv` | Product category hierarchy |
| `item_properties_part1.csv` | Item metadata (part 1) |
| `item_properties_part2.csv` | Item metadata (part 2) |

---

## ⚙️ How to Run End-to-End

```bash
pip install -r requirements.txt

# Configure GCS (see ☁️ Google Cloud Storage Setup above)
cp .env.example .env  # then edit .env

# data_prep.py auto-downloads from GCS if data/raw/events.csv is missing
python src/data_prep.py
python src/feature_engineering.py
python src/train_model.py
python src/evaluate.py

uvicorn src/app/api.py --reload
streamlit run src/app/dashboard.py
```

---

## 🔄 CI/CD (GitHub Actions)
Automated pipeline includes:

- Data preprocessing
- Model training
- Evaluation
- Artifact generation

---
## 🧩 Future Enhancements

- Real-time streaming recommendations (Kafka)
- Deep Learning (Neural CF, Transformers)
- Graph-based recommendations (GNN)
- A/B testing framework
- Reinforcement learning for dynamic recommendations

---

## 👤 Author

**Biplob Gon**  
_Data Scientist | AI/ML | Recommender Systems_

[![GitHub](https://img.shields.io/badge/GitHub-biplobgon-black?logo=github)](https://github.com/biplobgon)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Biplob%20Gon-blue?logo=linkedin)](https://linkedin.com/in/biplobgon)

---

## ⭐ If you found this useful
Give this repo a ⭐ to support and share!
