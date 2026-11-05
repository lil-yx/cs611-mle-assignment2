"""Airflow DAG: retrain a challenger on demand (manual trigger only).

Used when monitoring turns RED or at the 6-monthly refresh (docs/governance_sop.png).
Roll train_cutoff forward in config/base.yaml first if new labelled months exist.

Task graph:
    build_datamart ──► train_challenger

- build_datamart   : same skip-if-complete datamart build as ml_pipeline.
- train_challenger : trains LR / RF / XGBoost, picks the best and writes
                     model_store/<model_id>/challenger_report.json comparing it with the
                     champion on the same out-of-time window. champion.json is NOT
                     changed; promote through the model_deploy DAG after review.
                     (If no champion exists yet, the best model is bootstrapped instead.)

This is a separate DAG because ml_pipeline has an end_date: a manual run of it today
falls outside that window and Airflow would create the run with no tasks.
"""

import logging
import os

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

_APP_DIR = "/app"
os.environ.setdefault("PYTHONPATH", _APP_DIR)


def _build_datamart(**context):
    os.chdir(_APP_DIR)
    from main import build_if_missing
    build_if_missing()


def _train_challenger(**context):
    os.chdir(_APP_DIR)
    from pipelines.train_pipeline import main
    main()


with DAG(
    dag_id="model_training",
    description="Retrain a challenger and gate it against the champion (manual trigger only)",
    schedule=None,
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "loan-default", "governance"],
) as dag:

    build_datamart = PythonOperator(task_id="build_datamart", python_callable=_build_datamart)
    train_challenger = PythonOperator(task_id="train_challenger", python_callable=_train_challenger)

    build_datamart >> train_challenger
