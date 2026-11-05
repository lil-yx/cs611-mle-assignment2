"""Airflow DAG: end-to-end ML pipeline for loan-default prediction.

Runs monthly (1st of each month) and supports full backfill via catchup=True.

Task graph:
    build_datamart ──► train_model ──► run_inference ──► run_monitor

- build_datamart : builds bronze -> silver -> gold from data/ when any gold
                   label_store / feature_store partition is missing; otherwise skips.
- train_model    : no champion yet -> trains LR / RF / XGBoost and promotes the best
                   ("bootstrap"). Champion exists -> skipped. Retrains go through the
                   model_training DAG and only produce a CHALLENGER; promotion is a human
                   decision made through the model_deploy DAG.
- run_inference  : scores every customer in the feature_store for that month's
                   snapshot_date with the champion and writes gold predictions.
- run_monitor    : computes AUC/Gini, bad-rate drift and PSI for that month,
                   writes a gold model_monitoring table and saves PNG plots.

To retrain a challenger: trigger the model_training DAG.
"""

import logging
import os

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

_APP_DIR = "/app"

# Ensure imports from /app work regardless of Airflow's working directory
os.environ.setdefault("PYTHONPATH", _APP_DIR)


# ── Task callables ────────────────────────────────────────────────────────────

def _build_datamart(**context):
    os.chdir(_APP_DIR)
    from main import build_if_missing
    build_if_missing()


def _train(**context):
    os.chdir(_APP_DIR)
    from utils.ml.model_store import champion_exists, load_champion_id

    if champion_exists():
        logger.info("Champion %s already in production — skipping training.  "
                    "Trigger the model_training DAG to train a challenger.", load_champion_id())
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
    description="Monthly loan-default ML pipeline: datamart → train → inference → monitor",
    schedule="@monthly",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    # last feature-store partition; drop end_date when new months land in production
    end_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=True,       # enables full historical backfill
    max_active_runs=1,  # prevent concurrent backfill runs from colliding on the model store
    tags=["ml", "loan-default"],
    default_args={
        "retries": 1,
        "retry_delay": pendulum.duration(minutes=5),
    },
) as dag:

    build_datamart = PythonOperator(
        task_id="build_datamart",
        python_callable=_build_datamart,
    )

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

    build_datamart >> train_model >> run_inference >> run_monitor
