"""Helpers for saving and loading model artefacts.

Directory layout:
    model_store/
        champion.json               <- points to the model_id used in production
        champion_history.jsonl      <- append-only audit log of every promotion / rollback
        optuna.db                   <- tuned Optuna studies (versioned in git)
        <YYYYMMDD_HHMMSS>_<name>/
            model.pkl               <- fitted sklearn Pipeline
            feature_cols.json       <- ordered list of feature column names
            metadata.json           <- training date, metrics, split info
            shap_summary.json       <- mean |SHAP| per feature (best model of each run)
            challenger_report.json  <- challenger vs champion gate result (retrains only)

Promotion rules (see docs/governance_sop.png):
    - bootstrap : the first training run promotes its best model, because there is no
                  champion to compare against.
    - manual    : every later promotion goes through the model_deploy DAG, which needs a
                  passing challenger_report.json and a named approver.
    - rollback  : model_deploy can repoint champion.json to any earlier champion.
"""

import json
import os
import pickle
from datetime import datetime, UTC

from utils.common.config import CONFIG

_STORE = CONFIG["ml"]["model_store_dir"]


def _model_dir(model_id):
    return os.path.join(_STORE, model_id)


def _write_json_atomic(path, payload):
    """Write to a temp file and rename, so readers never see a half-written file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def save_model(model_id, model, metadata, feature_cols):
    path = _model_dir(model_id)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(path, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)


def model_exists(model_id):
    return os.path.exists(os.path.join(_model_dir(model_id), "model.pkl"))


def save_shap_summary(model_id, importance):
    """Save mean |SHAP| feature importance to model_store/<model_id>/shap_summary.json."""
    path = _model_dir(model_id)
    os.makedirs(path, exist_ok=True)
    ranked = sorted(importance.items(), key=lambda x: -x[1])
    payload = {"ranked_features": [{"feature": k, "mean_abs_shap": v} for k, v in ranked]}
    with open(os.path.join(path, "shap_summary.json"), "w") as f:
        json.dump(payload, f, indent=2)


def load_shap_top(model_id, k=10):
    """Return the top-k SHAP feature names, or None when the model has no SHAP summary."""
    path = os.path.join(_model_dir(model_id), "shap_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [r["feature"] for r in json.load(f)["ranked_features"][:k]]


# ---------------------------------------------------------------------------
# Champion pointer + audit trail
# ---------------------------------------------------------------------------

def champion_exists():
    return os.path.exists(os.path.join(_STORE, "champion.json"))


def load_champion():
    """Return the full champion.json record."""
    with open(os.path.join(_STORE, "champion.json")) as f:
        return json.load(f)


def load_champion_id():
    return load_champion()["model_id"]


def save_champion(model_id, promoted_by, approved_by=None, reason=None):
    """Point champion.json at model_id and append the change to champion_history.jsonl.

    promoted_by: "bootstrap" | "manual" | "rollback"
    """
    if promoted_by not in ("bootstrap", "manual", "rollback"):
        raise ValueError(f"unknown promoted_by: {promoted_by}")
    os.makedirs(_STORE, exist_ok=True)
    record = {
        "model_id":          model_id,
        "previous_model_id": load_champion_id() if champion_exists() else None,
        "promoted_by":       promoted_by,
        "approved_by":       approved_by,
        "reason":            reason,
        "updated_at":        datetime.now(UTC).isoformat(),
    }
    _write_json_atomic(os.path.join(_STORE, "champion.json"), record)
    with open(os.path.join(_STORE, "champion_history.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def load_champion_history():
    path = os.path.join(_STORE, "champion_history.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Challenger reports
# ---------------------------------------------------------------------------

def save_challenger_report(model_id, report):
    path = _model_dir(model_id)
    os.makedirs(path, exist_ok=True)
    _write_json_atomic(os.path.join(path, "challenger_report.json"), report)


def load_challenger_report(model_id):
    """Return the challenger report for model_id, or None if it was never gated."""
    path = os.path.join(_model_dir(model_id), "challenger_report.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_model(model_id=None):
    """Load a model by id, or the current champion if model_id is None."""
    if model_id is None:
        model_id = load_champion_id()
    path = _model_dir(model_id)
    with open(os.path.join(path, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(path, "feature_cols.json")) as f:
        feature_cols = json.load(f)
    with open(os.path.join(path, "metadata.json")) as f:
        metadata = json.load(f)
    return model, feature_cols, metadata
