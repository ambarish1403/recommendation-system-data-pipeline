"""Optional Apache Airflow entry point for the recommendation pipeline.

This DAG is optional: the project also runs without Airflow via `python run_pipeline.py`.
Install Apache Airflow in a supported Linux/Docker environment before loading this DAG.
"""
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from dags.pipeline import run_pipeline


def execute_recommendation_pipeline():
    """Run the dependency-ordered end-to-end Python workflow."""
    run_pipeline()


with DAG(
    dag_id="dm4ml_recommendation_pipeline",
    description="End-to-end recommendation data management and model workflow",
    start_date=datetime(2026, 1, 1),
    schedule=None,  # Trigger manually; set a cron expression for scheduled runs.
    catchup=False,
    tags=["DM4ML", "recommendation", "machine-learning"],
) as dag:
    run_recommendation_pipeline = PythonOperator(
        task_id="run_recommendation_pipeline",
        python_callable=execute_recommendation_pipeline,
    )
