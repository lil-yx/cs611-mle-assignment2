"""Unit tests for the new train_pipeline functions: Optuna tuning, CV retraining and SHAP.

Runs without Spark — all tests use tiny synthetic numpy arrays.

    python -m pytest tests/unit/test_train_pipeline.py -v
"""

import json
import os
import tempfile

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# patch config before importing the pipeline module so we can override trial count
import utils.common.config as _cfg
_cfg.CONFIG.setdefault("ml", {})
# assign (not setdefault): base.yaml already defines these, and the tests must never
# read or write the production Optuna studies in model_store/optuna.db
_cfg.CONFIG["ml"]["optuna_trials"] = 3
_cfg.CONFIG["ml"]["cv_folds"] = 3
_cfg.CONFIG["ml"]["optuna_storage"] = os.path.join(tempfile.mkdtemp(), "optuna_test.db")
_cfg.CONFIG["ml"].setdefault("random_state", 42)
_cfg.CONFIG["ml"].setdefault("train_cutoff", "2024-07-01")
_cfg.CONFIG["ml"].setdefault("model_store_dir", "model_store/")

from pipelines.train_pipeline import (
    _build_tuned_pipeline,
    _compute_shap,
    _cv_auc,
    _evaluate,
    _tune,
)
from utils.ml.model_store import save_shap_summary

# ---------------------------------------------------------------------------
# Shared fixture: tiny balanced dataset
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_data():
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_redundant=2, random_state=42,
    )
    # inject a few NaNs to exercise the imputer
    rng = np.random.default_rng(0)
    mask = rng.random(X.shape) < 0.05
    X[mask] = np.nan
    feature_cols = [f"feat_{i}" for i in range(X.shape[1])]
    return X, y, feature_cols


# ---------------------------------------------------------------------------
# _build_tuned_pipeline
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params,expected_steps", [
    (
        "logistic_regression",
        {"C": 1.0},
        ["imputer", "scaler", "clf"],
    ),
    (
        "random_forest",
        {"n_estimators": 50, "max_depth": 4, "min_samples_leaf": 2},
        ["imputer", "clf"],
    ),
    (
        "gradient_boosting",
        {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1,
         "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
         "scale_pos_weight": 2.5},
        ["imputer", "clf"],
    ),
])
def test_build_tuned_pipeline_step_names(name, params, expected_steps):
    pipe = _build_tuned_pipeline(name, params)
    assert list(pipe.named_steps.keys()) == expected_steps


def test_build_tuned_pipeline_lr_uses_given_C():
    pipe = _build_tuned_pipeline("logistic_regression", {"C": 0.01})
    assert pipe.named_steps["clf"].C == pytest.approx(0.01)


def test_build_tuned_pipeline_gbm_uses_given_lr():
    params = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.05,
              "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
              "scale_pos_weight": 2.5}
    pipe = _build_tuned_pipeline("gradient_boosting", params)
    assert pipe.named_steps["clf"].learning_rate == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# _evaluate
# ---------------------------------------------------------------------------

def test_evaluate_returns_expected_keys(tiny_data):
    X, y, _ = tiny_data
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf",     LogisticRegression(max_iter=200, random_state=42)),
    ])
    pipe.fit(X, y)
    metrics = _evaluate(pipe, X, y)
    required = {"auc_roc", "gini", "f1", "precision", "recall", "accuracy",
                "test_size", "bad_rate_test"}
    assert required <= set(metrics.keys())


def test_evaluate_gini_equals_two_auc_minus_one(tiny_data):
    X, y, _ = tiny_data
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf",     LogisticRegression(max_iter=200, random_state=42)),
    ])
    pipe.fit(X, y)
    m = _evaluate(pipe, X, y)
    assert m["gini"] == pytest.approx(2 * m["auc_roc"] - 1, abs=1e-4)


def test_evaluate_auc_in_valid_range(tiny_data):
    X, y, _ = tiny_data
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf",     XGBClassifier(objective="binary:logistic", n_estimators=50,
                                  random_state=42, eval_metric="logloss", verbosity=0)),
    ])
    pipe.fit(X, y)
    m = _evaluate(pipe, X, y)
    assert 0.5 <= m["auc_roc"] <= 1.0


# ---------------------------------------------------------------------------
# _cv_auc
# ---------------------------------------------------------------------------

def test_cv_auc_returns_mean_and_std(tiny_data):
    X, y, _ = tiny_data
    pipe = _build_tuned_pipeline(
        "logistic_regression", {"C": 1.0}
    )
    mean, std = _cv_auc(pipe, X, y)
    assert 0.5 <= mean <= 1.0
    assert 0.0 <= std < 0.2


def test_cv_auc_std_is_non_negative(tiny_data):
    X, y, _ = tiny_data
    pipe = _build_tuned_pipeline("random_forest",
        {"n_estimators": 20, "max_depth": 3, "min_samples_leaf": 2})
    _, std = _cv_auc(pipe, X, y)
    assert std >= 0.0


# ---------------------------------------------------------------------------
# _tune (fast: N_TRIALS patched to 3 above)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected_param", [
    ("logistic_regression", "C"),
    ("random_forest",       "n_estimators"),
    ("gradient_boosting",   "learning_rate"),
])
def test_tune_returns_expected_param_key(tiny_data, name, expected_param):
    X, y, _ = tiny_data
    best_params, best_cv_auc = _tune(name, X, y)
    assert expected_param in best_params
    assert 0.5 <= best_cv_auc <= 1.0


def test_tune_gbm_params_within_search_space(tiny_data):
    X, y, _ = tiny_data
    params, _ = _tune("gradient_boosting", X, y)
    assert 100 <= params["n_estimators"] <= 400
    assert 2 <= params["max_depth"] <= 6
    assert 0.01 <= params["learning_rate"] <= 0.2
    assert 0.6 <= params["subsample"] <= 1.0
    assert 1e-4 <= params["reg_lambda"] <= 10.0


# ---------------------------------------------------------------------------
# _compute_shap
# ---------------------------------------------------------------------------

def _fitted_pipeline(name, X, y):
    pipe = _build_tuned_pipeline(
        name,
        {"C": 1.0} if name == "logistic_regression"
        else {"n_estimators": 30, "max_depth": 3, "min_samples_leaf": 2}
        if name == "random_forest"
        else {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.1,
              "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
              "scale_pos_weight": 2.5},
    )
    pipe.fit(X, y)
    return pipe


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest", "gradient_boosting"])
def test_compute_shap_returns_all_features(tiny_data, name):
    X, y, feature_cols = tiny_data
    pipe = _fitted_pipeline(name, X, y)
    importance = _compute_shap(pipe, X, feature_cols, name)
    assert set(importance.keys()) == set(feature_cols)


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest", "gradient_boosting"])
def test_compute_shap_values_are_non_negative(tiny_data, name):
    X, y, feature_cols = tiny_data
    pipe = _fitted_pipeline(name, X, y)
    importance = _compute_shap(pipe, X, feature_cols, name)
    assert all(v >= 0.0 for v in importance.values())


# ---------------------------------------------------------------------------
# save_shap_summary (model_store helper)
# ---------------------------------------------------------------------------

def test_save_shap_summary_creates_file(tmp_path, monkeypatch):
    import utils.ml.model_store as ms
    monkeypatch.setattr(ms, "_STORE", str(tmp_path))

    importance = {"feat_a": 0.5, "feat_b": 0.3, "feat_c": 0.1}
    save_shap_summary("test_model_001", importance)

    out = tmp_path / "test_model_001" / "shap_summary.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert "ranked_features" in data
    assert data["ranked_features"][0]["feature"] == "feat_a"  # highest importance first


def test_save_shap_summary_ranked_descending(tmp_path, monkeypatch):
    import utils.ml.model_store as ms
    monkeypatch.setattr(ms, "_STORE", str(tmp_path))

    importance = {"x": 0.1, "y": 0.9, "z": 0.5}
    save_shap_summary("test_model_002", importance)

    out = tmp_path / "test_model_002" / "shap_summary.json"
    data = json.loads(out.read_text())
    vals = [r["mean_abs_shap"] for r in data["ranked_features"]]
    assert vals == sorted(vals, reverse=True)
