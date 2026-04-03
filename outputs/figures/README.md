# outputs/figures

Generated plots and visualisations from EDA, feature engineering, and model evaluation.

## Expected files

| File | Produced by | Description |
|------|-------------|-------------|
| `eda_event_distribution.png` | `notebooks/01_eda.ipynb` | Bar chart of event type counts (view / addtocart / transaction) |
| `eda_category_depth.png` | `notebooks/01_eda.ipynb` | Category tree depth distribution |
| `eda_events_per_visitor.png` | `notebooks/01_eda.ipynb` | Histogram of events per unique visitor |
| `eda_hourly_volume.png` | `notebooks/01_eda.ipynb` | Hourly event volume (peak-hour analysis) |
| `eda_item_overlap.png` | `notebooks/01_eda.ipynb` | Venn-style bar: events-only / both / props-only |
| `feat_user_interaction_score_dist.png` | `notebooks/02_feature_engineering.ipynb` | Distribution of weighted interaction scores |
| `feat_item_category_depth_dist.png` | `notebooks/02_feature_engineering.ipynb` | Item category depth distribution |
| `feat_session_length_dist.png` | `notebooks/02_feature_engineering.ipynb` | Session length (number of items) histogram |
| `eval_metrics_comparison.png` | `notebooks/04_model_evaluation.ipynb` | Bar chart comparing HR@K, NDCG@K, MRR@K across models |
| `eval_coverage_novelty.png` | `notebooks/04_model_evaluation.ipynb` | Catalogue coverage and novelty per model |

## Naming convention

`{stage}_{description}.png`  
- stage: `eda` | `feat` | `train` | `eval`  
- description: snake_case, concise
