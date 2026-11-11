# End-to-End Recommendation Data Management Pipeline

A reproducible, modular movie recommendation pipeline built for **Data Management for Machine Learning – Assignment 2**. It uses clearly documented synthetic data, validates it, creates time-aware train/test sets and reusable features, trains two recommenders, evaluates them, tracks experiments, and orchestrates the workflow.

## Quick start

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
```

The first run creates synthetic CSV inputs under `data/raw/`. Pipeline outputs are written under `data/processed/`, `data/features/`, `data/errors/`, `models/`, and `reports/`.

To use MLflow tracking (optional but included in requirements):
```bash
mlflow ui --backend-store-uri ./mlruns
```
Open http://127.0.0.1:5000. The pipeline logs parameters, metrics and model artifacts when MLflow is installed; it also writes a portable JSON experiment record.

## Pipeline stages

1. **Ingestion** – loads users, items and interactions from CSV.
2. **Validation** – checks required columns, missing values, duplicates, data types, rating bounds, timestamps and user/item referential integrity. Invalid interactions are quarantined in `data/errors/`.
3. **Preparation / transformation** – normalizes timestamps, removes duplicate rows, and performs a chronological per-user holdout split (last interaction per eligible user is test data).
4. **Transformation** – standardizes identifiers, item text and UTC timestamps; derives interaction date, hour and rating bucket.\n5. **Feature management** – builds user, item and interaction features, plus a user-item matrix.
6. **Training** – popularity baseline and item-item collaborative filtering (cosine similarity).
7. **Evaluation** – Precision@5, Recall@5, NDCG@5, catalog coverage and intra-list diversity; also reports MAE/RMSE for the popularity model's rating estimate.
8. **Orchestration** – dependency-aware Python workflow in `dags/pipeline.py`, with an optional Apache Airflow entry point in `dags/recommendation_pipeline_dag.py` and `dags/pipeline.py`. Airflow is not required for the standard Windows run.

## Outputs

- `data/errors/validation_report.json`, `invalid_interactions.csv`
- `data/processed/train.csv`, `test.csv`, `transformed_train.csv`, `transformed_test.csv`, `transformed_items.csv`, `user_activity.csv`, `item_popularity.csv`
- `data/features/user_features.csv`, `item_features.csv`, `interaction_features.csv`, `user_item_matrix.csv`
- `models/popularity_model.json`, `item_item_model.npz`, `item_item_metadata.json`
- `reports/evaluation_results.csv`, `feature_catalog.csv`, `final_report.md`, `pipeline_run.json`
- `mlruns/` when MLflow is enabled

## Data and assumptions

The default data is synthetic, generated with a fixed random seed. It contains users, catalog items (with categories), and timestamped 1–5 ratings. A user's final chronological interaction is held out for evaluation; users need at least two interactions to contribute to the split. Relevance is defined as a held-out rating of at least 4. Metrics are averaged over users with at least one relevant held-out item. The collaborative model is item-item cosine similarity over the training user-item matrix. This compact educational implementation favors transparency and reproducibility over production-scale performance.

## Re-run / reset

```bash
python run_pipeline.py
```
The pipeline deterministically regenerates its synthetic source data and overwrites generated outputs. To use your own data, replace the three CSV files in `data/raw/` with files using the documented columns (`user_id`, `age`, `country`; `item_id`, `title`, `category`; `user_id`, `item_id`, `rating`, `timestamp`) and set `generate_synthetic: false` in `configs/config.json`.

## Architecture

See `docs/architecture.md` for the pipeline diagram and design notes. The `dags/` folder contains an optional Airflow DAG; the default reproducible execution remains `python run_pipeline.py`.
