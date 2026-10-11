"""ML training pipeline with Optuna tuning, cross-validated retraining and SHAP explainability.

For each of the three candidate model types:
  1. Optuna (TPE, N_TRIALS) finds best hyperparameters using 5-fold StratifiedKFold CV on the
     training set.
  2. The tuned pipeline is then cross-validated on the full training set to produce a
     stable AUC estimate (CV mean ± std).
  3. It is fitted on the full training set and evaluated on the temporal holdout.
After all candidates are compared, the champion's SHAP values are computed and saved.

Usage (from project root, inside Docker):
    python -m pipelines.train_pipeline
"""

import os
from datetime import datetime, date, UTC

import numpy as np
import optuna
import shap
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.common.config import CONFIG
from utils.common.log import get_logger
from utils.common.spark_session import get_spark
from utils.features.feature_store import get_training_data
from utils.ml.model_store import save_champion, save_model, save_shap_summary

logger = get_logger("train_pipeline")

TRAIN_CUTOFF = date.fromisoformat(CONFIG["ml"]["train_cutoff"])
RANDOM_STATE = CONFIG["ml"]["random_state"]
N_TRIALS        = CONFIG["ml"].get("optuna_trials", 50)
CV_FOLDS        = CONFIG["ml"].get("cv_folds", 5)
OPTUNA_STORAGE  = "sqlite:///" + CONFIG["ml"].get("optuna_storage", "model_store/optuna.db")

# columns that are keys / metadata, not model inputs
_NON_FEATURE = {
    "loan_id", "Customer_ID", "loan_start_date",
    "label", "label_def", "snapshot_date", "feature_snapshot_date",
}

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Optuna objectives
# ---------------------------------------------------------------------------

def _make_objective(model_name, X_train, y_train):
    # XGBoost uses scale_pos_weight for imbalance: ratio of negatives to positives
    spw = float((y_train == 0).sum()) / float((y_train == 1).sum())

    def objective(trial):
        if model_name == "logistic_regression":
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
                ("clf",     LogisticRegression(
                    C=trial.suggest_float("C", 1e-3, 10.0, log=True),
                    max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced",
                )),
            ])
        elif model_name == "random_forest":
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf",     RandomForestClassifier(
                    n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
                    max_depth=trial.suggest_int("max_depth", 4, 12),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
                    random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1,
                )),
            ])
        else:  # gradient_boosting
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf",     XGBClassifier(
                    objective="binary:logistic",
                    n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
                    max_depth=trial.suggest_int("max_depth", 2, 6),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                    scale_pos_weight=spw,
                    tree_method="hist", random_state=RANDOM_STATE,
                    eval_metric="logloss", verbosity=0,
                )),
            ])
        return _cv_auc(pipe, X_train, y_train)[0]

    return objective


def _log_trial(model_name):
    """Build an Optuna callback that logs each completed trial's progress."""
    def callback(study, trial):
        done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if trial.state != optuna.trial.TrialState.COMPLETE:
            logger.warning("    %s trial %d %s", model_name, trial.number, trial.state.name)
            return
        is_best = study.best_trial.number == trial.number
        logger.info(
            "    %s trial %d/%d  cv_auc=%.4f  best=%.4f%s",
            model_name, done, N_TRIALS, trial.value, study.best_value,
            "  <- new best" if is_best else "",
        )
    return callback


def _tune(model_name, X_train, y_train):
    os.makedirs(CONFIG["ml"].get("model_store_dir", "model_store/"), exist_ok=True)
    study = optuna.create_study(
        study_name=model_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        storage=OPTUNA_STORAGE,
        load_if_exists=True,
    )
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, N_TRIALS - done)
    if remaining == 0:
        logger.info("  %s: all %d trials already done, loading best params", model_name, N_TRIALS)
    else:
        logger.info("  tuning %s (%d/%d trials remaining) …", model_name, remaining, N_TRIALS)
        study.optimize(
            _make_objective(model_name, X_train, y_train),
            n_trials=remaining, show_progress_bar=False,
            callbacks=[_log_trial(model_name)],
        )
    logger.info("  best params: %s  (cv_auc=%.4f)", study.best_params, study.best_value)
    return study.best_params, study.best_value


# ---------------------------------------------------------------------------
# Build a fitted pipeline from tuned params
# ---------------------------------------------------------------------------

def _build_tuned_pipeline(model_name, params):
    if model_name == "logistic_regression":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("clf",     LogisticRegression(
                C=params["C"], max_iter=1000,
                random_state=RANDOM_STATE, class_weight="balanced",
            )),
        ])
    elif model_name == "random_forest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf",     RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1,
            )),
        ])
    else:  # gradient_boosting
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf",     XGBClassifier(
                objective="binary:logistic",
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                reg_lambda=params["reg_lambda"],
                scale_pos_weight=params["scale_pos_weight"],
                tree_method="hist", random_state=RANDOM_STATE,
                eval_metric="logloss", verbosity=0,
            )),
        ])


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_prob)
    return {
        "auc_roc":       round(float(auc), 4),
        "gini":          round(float(2 * auc - 1), 4),
        "f1":            round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "precision":     round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":        round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "accuracy":      round(float(accuracy_score(y_test, y_pred)), 4),
        "test_size":     int(len(y_test)),
        "bad_rate_test": round(float(y_test.mean()), 4),
    }


def _cv_auc(pipeline, X, y):
    """Return (mean, std) AUC across CV_FOLDS folds on the given data."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for tr, val in cv.split(X, y):
        p = clone(pipeline)
        p.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[val], p.predict_proba(X[val])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


# ---------------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------------

def _compute_shap(pipeline, X_train, feature_cols, model_name):
    """Compute mean |SHAP| per feature for the champion and return as dict."""
    imputer = pipeline.named_steps["imputer"]
    X_imp   = imputer.transform(X_train)
    clf     = pipeline.named_steps["clf"]

    if model_name in ("gradient_boosting", "random_forest"):  # tree-based: TreeExplainer
        explainer   = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_imp)
        if isinstance(shap_values, list):
            # older SHAP: list of [class_0_array, class_1_array]
            shap_values = shap_values[1]
        elif shap_values.ndim == 3:
            # newer SHAP (0.44+): single array (n_samples, n_features, n_classes)
            shap_values = shap_values[:, :, 1]
        # else: 2D (n_samples, n_features) for binary GBM — use as-is
    else:
        scaler      = pipeline.named_steps["scaler"]
        X_scaled    = scaler.transform(X_imp)
        explainer   = shap.LinearExplainer(clf, shap.maskers.Independent(X_scaled))
        shap_values = explainer.shap_values(X_scaled)

    mean_abs   = np.abs(shap_values).mean(axis=0)
    importance = {feature_cols[i]: round(float(mean_abs[i]), 6) for i in range(len(feature_cols))}

    top20 = sorted(importance.items(), key=lambda x: -x[1])[:20]
    logger.info("  top SHAP features (mean |SHAP|):")
    for feat, val in top20:
        logger.info("    %-40s  %.5f", feat, val)
    return importance


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    spark = get_spark("train_pipeline")
    logger.info("loading gold feature + label stores …")
    pdf = get_training_data(spark).toPandas()
    spark.stop()
    logger.info("full dataset: %d rows, %d columns", *pdf.shape)

    feature_cols = [c for c in pdf.columns if c not in _NON_FEATURE]
    logger.info("feature columns: %d", len(feature_cols))

    # temporal split — never shuffle; future data must not appear in training
    train = pdf[pdf["loan_start_date"] < TRAIN_CUTOFF]
    test  = pdf[pdf["loan_start_date"] >= TRAIN_CUTOFF]
    logger.info(
        "train: %d rows (loan_start_date < %s) | test: %d rows (>= %s)",
        len(train), TRAIN_CUTOFF, len(test), TRAIN_CUTOFF,
    )
    logger.info(
        "bad rate — train: %.1f%%  test: %.1f%%",
        train["label"].mean() * 100, test["label"].mean() * 100,
    )

    X_train = train[feature_cols].values
    y_train = train["label"].values
    X_test  = test[feature_cols].values
    y_test  = test["label"].values

    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    # scores: name -> (model_id, holdout_auc, fitted_pipeline)
    scores = {}

    for name in ("logistic_regression", "random_forest", "gradient_boosting"):
        logger.info("=== %s ===", name)

        # 1. Optuna hyperparameter search
        best_params, _ = _tune(name, X_train, y_train)
        if name == "gradient_boosting":
            best_params["scale_pos_weight"] = float((y_train == 0).sum()) / float((y_train == 1).sum())

        # 2. Cross-validated retraining: full CV report on training set
        pipeline = _build_tuned_pipeline(name, best_params)
        cv_mean, cv_std = _cv_auc(pipeline, X_train, y_train)
        logger.info("  cross-val AUC on training set: %.4f ± %.4f", cv_mean, cv_std)

        # 3. Fit on full training set, evaluate on temporal holdout
        pipeline.fit(X_train, y_train)
        metrics = _evaluate(pipeline, X_test, y_test)
        metrics["cv_auc_mean"] = round(cv_mean, 4)
        metrics["cv_auc_std"]  = round(cv_std, 4)
        logger.info(
            "  holdout AUC %.4f  Gini %.4f  CV %.4f±%.4f",
            metrics["auc_roc"], metrics["gini"], cv_mean, cv_std,
        )

        model_id = f"{run_id}_{name}"
        save_model(
            model_id=model_id,
            model=pipeline,
            feature_cols=feature_cols,
            metadata={
                "model_id":     model_id,
                "model_type":   name,
                "trained_at":   datetime.now(UTC).isoformat(),
                "train_cutoff": str(TRAIN_CUTOFF),
                "train_size":   int(len(train)),
                "best_params":  best_params,
                "metrics":      metrics,
            },
        )
        scores[name] = (model_id, metrics["auc_roc"], pipeline)

    best_name, (best_id, best_auc, best_pipeline) = max(
        scores.items(), key=lambda kv: kv[1][1],
    )
    save_champion(best_id)
    logger.info(
        "champion registered: %s  model_id=%s  AUC=%.4f",
        best_name, best_id, best_auc,
    )

    # 4. SHAP explainability for champion
    logger.info("computing SHAP values for champion (%s) …", best_name)
    shap_importance = _compute_shap(best_pipeline, X_train, feature_cols, best_name)
    save_shap_summary(best_id, shap_importance)
    logger.info("SHAP summary saved → model_store/%s/shap_summary.json", best_id)

    logger.info("\n%-25s  %s", "model", "AUC-ROC")
    for name, (mid, auc, _) in sorted(scores.items(), key=lambda kv: -kv[1][1]):
        flag = " ← champion" if mid == best_id else ""
        logger.info("%-25s  %.4f%s", name, auc, flag)


if __name__ == "__main__":
    main()
