"""Helpers for saving and loading model artefacts.

Directory layout:
    model_store/
        champion.json               <- points to the current best model_id
        <YYYYMMDD_HHMMSS>_<name>/
            model.pkl               <- fitted sklearn Pipeline
            feature_cols.json       <- ordered list of feature column names
            metadata.json           <- training date, metrics, split info
"""

import json
import os
import pickle
from datetime import datetime, UTC

from utils.common.config import CONFIG

_STORE = CONFIG["ml"]["model_store_dir"]


def _model_dir(model_id):
    return os.path.join(_STORE, model_id)


def save_model(model_id, model, metadata, feature_cols):
    path = _model_dir(model_id)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(path, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)


def save_shap_summary(model_id, importance):
    """Save mean |SHAP| feature importance to model_store/<model_id>/shap_summary.json."""
    path = _model_dir(model_id)
    os.makedirs(path, exist_ok=True)
    ranked = sorted(importance.items(), key=lambda x: -x[1])
    payload = {"ranked_features": [{"feature": k, "mean_abs_shap": v} for k, v in ranked]}
    with open(os.path.join(path, "shap_summary.json"), "w") as f:
        json.dump(payload, f, indent=2)


def save_champion(model_id):
    os.makedirs(_STORE, exist_ok=True)
    with open(os.path.join(_STORE, "champion.json"), "w") as f:
        json.dump({"model_id": model_id, "updated_at": datetime.now(UTC).isoformat()}, f, indent=2)


def load_champion_id():
    with open(os.path.join(_STORE, "champion.json")) as f:
        return json.load(f)["model_id"]


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
