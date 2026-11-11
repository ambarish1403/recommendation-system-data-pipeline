import json
from datetime import datetime, timezone
from src.utils import ROOT, ensure_dirs, load_config, write_json
from src.ingestion.ingest import ingest
from src.validation.validate import validate
from src.preparation.prepare import prepare
from src.transformation.transform import transform
from src.features.build_features import build_features
from src.training.train import train_models
from src.evaluation.evaluate import evaluate

def run_pipeline():
    ensure_dirs()
    started = datetime.now(timezone.utc)
    print("[1/7] Ingesting data")
    raw = ingest()
    print("[2/7] Validating data")
    valid, validation_report = validate(raw)
    print("[3/7] Preparing and splitting data")
    prepared = prepare(valid)
    print("[4/8] Transforming and standardizing data")
    transformed = transform(prepared)
    print("[5/8] Building reusable features")
    features = build_features(transformed)
    transformed["features"] = features
    print("[6/8] Training recommendation models")
    models = train_models(transformed)
    print("[7/8] Evaluating models")
    results = evaluate(transformed, models)
    print("[8/8] Tracking and writing report")
    cfg = load_config()
    tracking = {"experiment": cfg["mlflow_experiment"], "params": {"k": cfg["k"],
                "relevance_threshold": cfg["test_relevance_threshold"], "seed": cfg["random_seed"]},
                "metrics": results.to_dict(orient="records")}
    write_json(ROOT / "reports/experiment_tracking.json", tracking)
    try:
        import mlflow
        mlflow.set_experiment(cfg["mlflow_experiment"])
        with mlflow.start_run(run_name="recommendation-pipeline"):
            mlflow.log_params(tracking["params"])
            for row in tracking["metrics"]:
                prefix = "popularity" if row["model"] == "Popularity baseline" else "item_item"
                for metric in ["precision_at_k", "recall_at_k", "ndcg_at_k", "coverage", "diversity", "mae", "rmse"]:
                    value = row.get(metric)
                    if value is not None and value == value:
                        mlflow.log_metric(f"{prefix}_{metric}", float(value))
            mlflow.log_artifact(str(ROOT / "reports/evaluation_results.csv"))
            mlflow.log_artifact(str(ROOT / "data/errors/validation_report.json"))
    except ImportError:
        print("MLflow not installed; portable experiment_tracking.json was written.")
    except Exception as exc:
        print(f"MLflow tracking unavailable ({exc}); portable experiment record retained.")
    finished = datetime.now(timezone.utc)
    report = f'''# Final Report – Recommendation Pipeline

## Objective and design
This project implements the required end-to-end data management pipeline for a movie recommendation use case. Synthetic users, movie items and timestamped ratings are generated with a fixed random seed so the run is reproducible. CSV files are used for transparent intermediate storage.

## Data quality and preparation
Validation checks required fields, missing values, duplicates, numeric ratings in the 1–5 range, parseable UTC timestamps, and user/item referential integrity. Invalid interactions are quarantined rather than silently discarded. The last chronological interaction for each user with at least two interactions is held out as test data.

## Features
User features include activity count, mean rating, preferred category and recency. Item features include popularity, mean rating and rating count. Interaction features include rating deviation from the user's mean. The user-item matrix is reused by collaborative filtering.

## Models and evaluation
Two models are compared: a global popularity baseline and item-item collaborative filtering using cosine similarity. Relevance is defined as a held-out rating >= {cfg["test_relevance_threshold"]}; ranking metrics use K={cfg["k"]}. Diversity is measured as the proportion of distinct item categories in a recommendation list. MAE/RMSE are reported for the popularity model's global-mean rating estimate; the ranking-only collaborative model does not predict explicit ratings.

### Results
{results.to_markdown(index=False)}

## Orchestration and tracking
A dependency-ordered Python workflow executes ingestion, validation, preparation, feature generation, training, evaluation and reporting. Parameters and metrics are stored in `reports/experiment_tracking.json`; MLflow logging is enabled when installed.

## Limitations
The synthetic dataset is small and does not represent real-world behavior. Item-item similarity can be sparse for low-activity users/items. The time split holds out one event per eligible user rather than a global date cutoff. Diversity is category-based and does not capture semantic novelty or serendipity. A production deployment would require monitoring, incremental feature updates, privacy controls and online evaluation.
'''
    (ROOT / "reports/final_report.md").write_text(report, encoding="utf-8")
    run_info = {"status": "success", "started_at": started.isoformat(), "finished_at": finished.isoformat(),
                "validation": validation_report["status"], "train_rows": len(prepared["train"]),
                "test_rows": len(prepared["test"])}
    write_json(ROOT / "reports/pipeline_run.json", run_info)
    print(results.to_string(index=False))
    print(f"Pipeline completed. Outputs: {ROOT}")
    return results
