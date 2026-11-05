"""Airflow DAG: human-approved promotion and rollback of the champion model.

Manual only (no schedule). Triggering it with the default (empty) parameters does
nothing: check_request short-circuits and every later task is skipped.

Task graph:
    check_request ──► check_gate ──► promote ──► check_deployment

- check_request    : stops cleanly if no model_id was given; requires approved_by.
- check_gate       : promotion -> the model needs a challenger_report.json that says
                     PROMOTE and was computed against the CURRENT champion.
                     rollback  -> the model must have been a champion before.
- promote          : atomically repoints model_store/champion.json and appends the
                     decision (who, why, previous model) to champion_history.jsonl.
- check_deployment : loads the new champion and scores the latest feature_store
                     partition to prove it works before the next monthly run uses it.

Promote a challenger (after reading its challenger_report.json):
    airflow dags trigger model_deploy --conf '{"model_id": "<id>", "approved_by": "<name>", "reason": "..."}'
Roll back:
    airflow dags trigger model_deploy --conf '{"model_id": "<old id>", "approved_by": "<name>", "rollback": true}'
"""

import logging
import os

import pendulum
from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.models.param import Param
from airflow.operators.python import PythonOperator, ShortCircuitOperator

logger = logging.getLogger(__name__)

_APP_DIR = "/app"
os.environ.setdefault("PYTHONPATH", _APP_DIR)


def _check_request(**context):
    p = context["params"]
    model_id = (p.get("model_id") or "").strip()
    if not model_id:
        logger.info("No model_id given — nothing to deploy. Trigger with "
                    "{'model_id': ..., 'approved_by': ...} to promote or roll back.")
        return False
    if not (p.get("approved_by") or "").strip():
        raise AirflowFailException("approved_by is required: every promotion needs a named approver")

    os.chdir(_APP_DIR)
    from utils.ml.model_store import champion_exists, load_champion_id, model_exists
    if not model_exists(model_id):
        raise AirflowFailException(f"model_id {model_id} not found in model_store/")
    if champion_exists() and load_champion_id() == model_id:
        logger.info("%s is already the champion — nothing to do", model_id)
        return False
    return True


def _check_gate(**context):
    os.chdir(_APP_DIR)
    from utils.ml.model_store import load_challenger_report, load_champion_history, load_champion_id
    p = context["params"]
    model_id = p["model_id"].strip()

    if p.get("rollback"):
        history = load_champion_history()
        # previous_model_id also counts: covers champions set before the history log existed
        past = {r["model_id"] for r in history} | {r["previous_model_id"] for r in history}
        if model_id not in past:
            raise AirflowFailException(f"rollback target {model_id} was never a champion")
        logger.info("rollback to former champion %s allowed", model_id)
        return

    report = load_challenger_report(model_id)
    if report is None:
        raise AirflowFailException(f"{model_id} has no challenger_report.json — retrain with "
                                   "the model_training DAG so it is compared with the champion")
    current = load_champion_id()
    if report["champion_id"] != current:
        raise AirflowFailException(f"report compared against {report['champion_id']}, but the "
                                   f"champion is now {current}; retrain to refresh the comparison")
    for c in report["checks"]:
        logger.info("  %-16s passed=%s value=%s rule %s", c["name"], c["passed"], c["value"], c["rule"])
    if report["recommendation"] != "PROMOTE":
        raise AirflowFailException(f"gate says {report['recommendation']} for {model_id}")
    logger.info("gate passed for %s (Gini %s -> %s)", model_id,
                report["champion_gini"], report["challenger_gini"])


def _promote(**context):
    os.chdir(_APP_DIR)
    from utils.ml.model_store import save_champion
    p = context["params"]
    record = save_champion(
        p["model_id"].strip(),
        promoted_by="rollback" if p.get("rollback") else "manual",
        approved_by=p["approved_by"].strip(),
        reason=(p.get("reason") or "").strip() or None,
    )
    logger.info("champion.json -> %s (was %s)", record["model_id"], record["previous_model_id"])


def _check_deployment(**context):
    os.chdir(_APP_DIR)
    import glob

    import numpy as np
    import pandas as pd

    from utils.common.config import table_dir
    from utils.ml.model_store import load_champion_id, load_model

    expected = context["params"]["model_id"].strip()
    model, feature_cols, metadata = load_model()
    if load_champion_id() != expected or metadata["model_id"] != expected:
        raise AirflowFailException(f"champion is {load_champion_id()}, expected {expected}")

    latest = sorted(glob.glob(table_dir("gold", "feature_store") + "*.parquet"))[-1]
    X = pd.read_parquet(latest, columns=feature_cols).head(500).values
    scores = model.predict_proba(X)[:, 1]
    if not np.all(np.isfinite(scores)) or scores.min() < 0 or scores.max() > 1:
        raise AirflowFailException("new champion produced invalid scores")
    logger.info("smoke test OK: %d rows from %s, mean score %.3f — next scheduled run uses %s",
                len(scores), os.path.basename(latest), scores.mean(), expected)


with DAG(
    dag_id="model_deploy",
    description="Human-approved champion promotion / rollback (manual trigger only)",
    schedule=None,
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "loan-default", "governance"],
    params={
        "model_id":    Param("", type="string", description="model_store/<model_id> to make champion"),
        "approved_by": Param("", type="string", description="Name of the model-risk approver"),
        "reason":      Param("", type="string", description="Why this model is being promoted"),
        "rollback":    Param(False, type="boolean",
                             description="Repoint to a former champion without the challenger gate"),
    },
) as dag:

    check_request = ShortCircuitOperator(task_id="check_request", python_callable=_check_request)
    check_gate = PythonOperator(task_id="check_gate", python_callable=_check_gate)
    promote = PythonOperator(task_id="promote", python_callable=_promote)
    check_deployment = PythonOperator(task_id="check_deployment", python_callable=_check_deployment)

    check_request >> check_gate >> promote >> check_deployment
