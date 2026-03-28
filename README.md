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

## ⚙️ How to Run End-to-End

```bash
pip install -r requirements.txt

python src/data_prep.py
python src/feature_engineering.py
python src/train_model.py
python src/evaluate.py

uvicorn src/app/api.py --reload
streamlit run src/app/dashboard.py
```

---

## 🔄 CI/CD (GitHub Actions)


