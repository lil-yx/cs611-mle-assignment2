"""Airflow DAG: retrain a challenger on demand (manual trigger only).

Used when monitoring turns RED or at the 6-monthly refresh (docs/governance_sop.png).
Roll train_cutoff forward in config/base.yaml first if new labelled months exist.

Task graph:
    check_gold ──► train_challenger

- check_gold       : read-only gate. Fails if any gold label_store / feature_store
                     partition is missing. This DAG never writes the datamart: building
                     data is ml_pipeline's job, which also avoids two DAGs writing the
                     same partition at once.
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
from airflow.exceptions import AirflowFailException
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

_APP_DIR = "/app"
os.environ.setdefault("PYTHONPATH", _APP_DIR)


def _check_gold(**context):
    os.chdir(_APP_DIR)
    from main import expected_partitions, missing_partitions
    missing = missing_partitions("gold")
    if missing:
        raise AirflowFailException(
            f"gold incomplete: {len(missing)} partitions missing (e.g. {missing[:3]}). "
            "Run ml_pipeline first so the data layers are built, then retrain.")
    logger.info("gold complete: %d partitions present — ready to train", len(expected_partitions("gold")))


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

    check_gold = PythonOperator(task_id="check_gold", python_callable=_check_gold)
    train_challenger = PythonOperator(task_id="train_challenger", python_callable=_train_challenger)

    check_gold >> train_challenger
