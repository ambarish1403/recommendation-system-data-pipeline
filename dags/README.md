# Optional Airflow DAG

`recommendation_pipeline_dag.py` exposes the end-to-end workflow as an Apache Airflow DAG named `dm4ml_recommendation_pipeline`.

The DAG calls `dags.pipeline.run_pipeline()`, which internally executes the dependency-ordered stages: ingestion → validation → preparation → transformation → features → training → evaluation → tracking/reporting.

## Important

Airflow is optional. The project runs on Windows without Airflow using:

```bash
python run_pipeline.py
```

For Airflow, use a supported Linux or Docker-based Airflow installation, place this repository on the Airflow worker's Python path, install the project's `requirements.txt` in that environment, and copy/symlink this `dags/` directory into Airflow's configured DAGs folder. Then refresh the DAG list and trigger `dm4ml_recommendation_pipeline`.

The DAG is manually triggerable (`schedule=None`) to avoid unexpected repeated training runs. A cron schedule can be configured in the DAG if recurring execution is required.
