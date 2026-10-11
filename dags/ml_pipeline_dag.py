"""Airflow DAG: end-to-end ML pipeline for loan-default prediction.

Runs monthly (1st of each month) and supports full backfill via catchup=True.

Task graph:
    train_model ──► run_inference ──► run_monitor

- train_model   : trains all candidate models and registers the best as champion.
                  Skipped automatically when a champion already exists so that
                  backfill runs don't retrain on every historical date.
- run_inference : scores every customer in the feature_store for that month's
                  snapshot_date and writes a gold predictions table.
- run_monitor   : computes AUC/Gini, bad-rate drift and PSI for that month,
                  writes a gold model_monitoring table and saves PNG plots.

To trigger a forced retrain:
    airflow dags trigger ml_pipeline --conf '{"force_retrain": true}'
"""

import logging
import os

import pendulum
from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

_APP_DIR = "/app"

# Ensure imports from /app work regardless of Airflow's working directory
os.environ.setdefault("PYTHONPATH", _APP_DIR)


# ── Task callables ────────────────────────────────────────────────────────────

def _train(**context):
    os.chdir(_APP_DIR)
    force = context["params"].get("force_retrain", False)
    champion_path = os.path.join(_APP_DIR, "model_store", "champion.json")

    if not force and os.path.exists(champion_path):
        logger.info("Champion model already exists — skipping training.  "
                    "Pass force_retrain=true to override.")
        return

    logger.info("Training all candidate models …")
    from pipelines.train_pipeline import main
    main()


def _inference(**context):
    os.chdir(_APP_DIR)
    snapshot_date = context["ds"]           # Airflow logical date as YYYY-MM-DD
    logger.info("Running inference for snapshot_date=%s", snapshot_date)
    from pipelines.inference_pipeline import run
    run(snapshot_date)


def _monitor(**context):
    os.chdir(_APP_DIR)
    snapshot_date = context["ds"]
    logger.info("Running monitor for snapshot_date=%s", snapshot_date)
    from pipelines.monitor_pipeline import run
    run(snapshot_date)


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="ml_pipeline",
    description="Monthly loan-default ML pipeline: train → inference → monitor",
    schedule="@monthly",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    # last feature-store partition; drop end_date when new months land in production
    end_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=True,       # enables full historical backfill
    max_active_runs=1,  # prevent concurrent backfill runs from colliding on the model store
    tags=["ml", "loan-default"],
    params={
        "force_retrain": Param(
            False, type="boolean",
            description="Set true to retrain even if a champion model already exists.",
        ),
    },
    default_args={
        "retries": 1,
        "retry_delay": pendulum.duration(minutes=5),
    },
) as dag:

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=_train,
    )

    run_inference = PythonOperator(
        task_id="run_inference",
        python_callable=_inference,
    )

    run_monitor = PythonOperator(
        task_id="run_monitor",
        python_callable=_monitor,
    )

    train_model >> run_inference >> run_monitor
