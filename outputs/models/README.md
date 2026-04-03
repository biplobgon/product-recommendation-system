# outputs/models

Serialised model artefacts saved after training. Load these for inference or evaluation without re-training.

## Expected files

| File | Saved by | Description |
|------|----------|-------------|
| `als_model.npz` | `src/models/collaborative_filtering.py` | ALS factor matrices (user × item) via `implicit` |
| `als_model_meta.pkl` | `src/models/collaborative_filtering.py` | Visitor/item index mappings for ALS |
| `cb_similarity.npz` | `src/models/content_based.py` | Pre-computed TF-IDF cosine similarity matrix |
| `cb_item_ids.pkl` | `src/models/content_based.py` | Item ID list aligned to similarity matrix rows |
| `cb_vectorizer.pkl` | `src/models/content_based.py` | Fitted TF-IDF vectorizer |
| `gru4rec_model.pt` | `src/models/session_based.py` | GRU4Rec PyTorch state dict |
| `gru4rec_meta.pkl` | `src/models/session_based.py` | Item encoder/decoder dicts and model config |
| `conversion_ranker.pkl` | `src/models/conversion_ranker.py` | Trained LightGBM binary classifier |
| `hybrid_config.pkl` | `src/models/hybrid.py` | Blend weights and component model paths |

## Loading example

```python
from src.models.collaborative_filtering import ALSRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.session_based import SessionBasedRecommender
from src.models.conversion_ranker import ConversionRanker
from src.models.hybrid import HybridRecommender

als = ALSRecommender.load("outputs/models/als_model.npz")
cb  = ContentBasedRecommender.load("outputs/models/cb_similarity.npz")
sb  = SessionBasedRecommender.load("outputs/models/gru4rec_model.pt")
ranker = ConversionRanker.load("outputs/models/conversion_ranker.pkl")

hybrid = HybridRecommender(cf_model=als, cb_model=cb, sb_model=sb, ranker=ranker)
recs = hybrid.recommend(visitor_id=12345, session_items=[67890], top_k=10)
```

## Versioning

For experiment tracking, prefix artefacts with a run ID:  
`als_model_run20260404.npz`  
MLflow artefact logging is configured in `configs/pipeline_config.yaml`.
