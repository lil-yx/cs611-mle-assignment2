"""ML training pipeline with Optuna tuning, cross-validated retraining and SHAP explainability.

For each of the three candidate model types:
  1. Optuna (TPE, N_TRIALS complete trials) finds hyperparameters with TUNING_CV_FOLDS-fold CV,
     a median pruner that drops weak trials after a fold, and early stopping for XGBoost
     (tree count chosen on an inner split of each training fold). Studies are named by a
     fingerprint of features + tuning settings, so stale studies are never reused.
  2. The tuned pipeline is then cross-validated on the full training set to produce a
     stable AUC estimate (CV mean ± std).
  3. It is fitted on the full training set and evaluated on the temporal holdout.
After all candidates are compared, the best one's SHAP values are computed and saved.

Promotion (see utils/ml/promotion.py):
  - no champion yet -> the best candidate is promoted automatically ("bootstrap")
  - champion exists -> the best candidate is a challenger: it is compared with the
    champion on the same out-of-time window and a challenger_report.json is written.
    champion.json is NOT changed; a person promotes through the model_deploy DAG.

Usage (from project root, inside Docker):
    python -m pipelines.train_pipeline
"""

import hashlib
import json
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
from utils.ml.model_store import (
    champion_exists, load_champion_id, load_model, load_shap_top,
    save_challenger_report, save_champion, save_model, save_shap_summary,
)
from utils.ml.promotion import build_report

logger = get_logger("train_pipeline")

TRAIN_CUTOFF = date.fromisoformat(CONFIG["ml"]["train_cutoff"])
RANDOM_STATE = CONFIG["ml"]["random_state"]
N_TRIALS        = CONFIG["ml"].get("optuna_trials", 25)
CV_FOLDS        = CONFIG["ml"].get("cv_folds", 5)              # final CV report
TUNING_CV_FOLDS = CONFIG["ml"].get("tuning_cv_folds", 3)       # Optuna objective (ranking only)
N_JOBS          = CONFIG["ml"].get("n_jobs", 4)                # threads per model fit
MAX_TREES       = CONFIG["ml"].get("max_trees", 1000)          # XGBoost ceiling; early stopping picks
EARLY_STOPPING  = CONFIG["ml"].get("early_stopping_rounds", 30)
SEARCH_SPACE_VERSION = 2   # bump when a search space below changes
OPTUNA_STORAGE  = "sqlite:///" + CONFIG["ml"].get("optuna_storage", "model_store/optuna.db")

# columns that are keys / metadata, not model inputs
_NON_FEATURE = {
    "loan_id", "Customer_ID", "loan_start_date",
    "label", "label_def", "snapshot_date", "feature_snapshot_date",
}

# Gold columns kept in the feature store but not used as model inputs (config ml.exclude_features).
# Clickstream fe_1..fe_20 exist for 100% of loans before Jul 2024 and 0% after (see notebooks/01_eda),
# so they lift in-period CV AUC (~0.81 -> ~0.90) but are blank for every loan scored out of time
# (OOT AUC 0.810 without vs 0.804 with). has_clickstream is then a pure time-period flag.
EXCLUDED_FEATURES = CONFIG["ml"].get("exclude_features", [])


def select_features(columns):
    """Model inputs: every gold column except keys/labels and configured exclusions.
    An exclusion ending in '*' is a prefix (e.g. 'fe_*')."""
    def excluded(c):
        return any(c.startswith(e[:-1]) if e.endswith("*") else c == e for e in EXCLUDED_FEATURES)
    return [c for c in columns if c not in _NON_FEATURE and not excluded(c)]


optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Optuna objectives
# ---------------------------------------------------------------------------

def _suggest_estimator(model_name, trial):
    """Unfitted classifier for one trial (no imputer: the objective imputes per fold)."""
    if model_name == "logistic_regression":
        return LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 10.0, log=True),
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced",
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
            max_depth=trial.suggest_int("max_depth", 4, 12),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            random_state=RANDOM_STATE, class_weight="balanced", n_jobs=N_JOBS,
        )
    # gradient_boosting: no n_estimators in the search — early stopping picks the tree count
    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=MAX_TREES,
        early_stopping_rounds=EARLY_STOPPING,
        max_depth=trial.suggest_int("max_depth", 2, 6),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        tree_method="hist", random_state=RANDOM_STATE,
        eval_metric="logloss", verbosity=0, n_jobs=N_JOBS,
    )


def _make_objective(model_name, X_train, y_train):
    # XGBoost uses scale_pos_weight for imbalance: ratio of negatives to positives
    spw = float((y_train == 0).sum()) / float((y_train == 1).sum())
    cv = StratifiedKFold(n_splits=TUNING_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        clf = _suggest_estimator(model_name, trial)
        aucs, trees = [], []
        for step, (tr, val) in enumerate(cv.split(X_train, y_train)):
            imputer = SimpleImputer(strategy="median").fit(X_train[tr])
            X_tr, X_val = imputer.transform(X_train[tr]), imputer.transform(X_train[val])
            model = clone(clf)
            if model_name == "logistic_regression":
                model = Pipeline([("scaler", StandardScaler()), ("clf", model)])
                model.fit(X_tr, y_train[tr])
            elif model_name == "gradient_boosting":
                # early stopping on an inner split of the TRAINING fold, so the
                # validation fold that is scored never chooses the tree count
                inner_tr, inner_es = next(StratifiedKFold(
                    n_splits=5, shuffle=True, random_state=RANDOM_STATE,
                ).split(X_tr, y_train[tr]))
                model.set_params(scale_pos_weight=spw)
                model.fit(X_tr[inner_tr], y_train[tr][inner_tr],
                          eval_set=[(X_tr[inner_es], y_train[tr][inner_es])], verbose=False)
                trees.append(model.best_iteration + 1)
            else:
                model.fit(X_tr, y_train[tr])
            aucs.append(roc_auc_score(y_train[val], model.predict_proba(X_val)[:, 1]))

            # pruning: stop a trial early when its running mean is clearly below others
            trial.report(float(np.mean(aucs)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if trees:
            trial.set_user_attr("n_estimators", int(round(np.mean(trees))))
        return float(np.mean(aucs))

    return objective


def _log_trial(model_name):
    """Build an Optuna callback that logs each finished trial's progress."""
    def callback(study, trial):
        done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if trial.state == optuna.trial.TrialState.PRUNED:
            logger.info("    %s trial %d pruned after %d folds", model_name, trial.number,
                        len(trial.intermediate_values))
            return
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


def _study_name(model_name, feature_cols):
    """Study name tied to the data and tuning setup, so a study tuned on different
    features or settings is never silently reused."""
    fingerprint = hashlib.sha1(json.dumps({
        "features": list(feature_cols), "folds": TUNING_CV_FOLDS,
        "max_trees": MAX_TREES, "early_stopping": EARLY_STOPPING,
        "train_cutoff": str(TRAIN_CUTOFF), "search_space": SEARCH_SPACE_VERSION,
    }, sort_keys=True).encode()).hexdigest()[:10]
    return f"{model_name}__{fingerprint}"


def _tune(model_name, X_train, y_train, feature_cols=None):
    os.makedirs(CONFIG["ml"].get("model_store_dir", "model_store/"), exist_ok=True)
    if feature_cols is None:
        feature_cols = [f"f{i}" for i in range(X_train.shape[1])]
    study = optuna.create_study(
        study_name=_study_name(model_name, feature_cols),
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        # wait for 5 complete trials, and always score at least one fold, before pruning
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        storage=OPTUNA_STORAGE,
        load_if_exists=True,
    )
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, N_TRIALS - done)
    if remaining == 0:
        logger.info("  %s: all %d trials already done (study %s), loading best params",
                    model_name, N_TRIALS, study.study_name)
    else:
        logger.info("  tuning %s: %d/%d trials remaining, %d-fold CV, n_jobs=%d (study %s) …",
                    model_name, remaining, N_TRIALS, TUNING_CV_FOLDS, N_JOBS, study.study_name)
        # count only COMPLETE trials toward the budget; pruned ones are cheap extras,
        # capped so a run can never loop indefinitely
        def _complete():
            return len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        while _complete() < N_TRIALS and len(study.trials) < 3 * N_TRIALS:
            study.optimize(
                _make_objective(model_name, X_train, y_train),
                n_trials=1, show_progress_bar=False,
                callbacks=[_log_trial(model_name)],
            )
    best_params = {**study.best_params, **study.best_trial.user_attrs}
    logger.info("  best params: %s  (cv_auc=%.4f)", best_params, study.best_value)
    return best_params, study.best_value


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
                random_state=RANDOM_STATE, class_weight="balanced", n_jobs=N_JOBS,
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
                eval_metric="logloss", verbosity=0, n_jobs=N_JOBS,
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

    feature_cols = select_features(pdf.columns)
    logger.info("excluded from model inputs by config: %s", EXCLUDED_FEATURES)
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
        best_params, _ = _tune(name, X_train, y_train, feature_cols)
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

    # 4. the run's best candidate (selection within a run is always automatic)
    best_name, (best_id, best_auc, best_pipeline) = max(
        scores.items(), key=lambda kv: kv[1][1],
    )
    logger.info("best candidate this run: %s  model_id=%s  AUC=%.4f", best_name, best_id, best_auc)

    # 5. SHAP explainability for the best candidate
    logger.info("computing SHAP values for %s …", best_name)
    shap_importance = _compute_shap(best_pipeline, X_train, feature_cols, best_name)
    save_shap_summary(best_id, shap_importance)
    logger.info("SHAP summary saved → model_store/%s/shap_summary.json", best_id)

    logger.info("\n%-25s  %s", "model", "AUC-ROC")
    for name, (mid, auc, _) in sorted(scores.items(), key=lambda kv: -kv[1][1]):
        flag = " <- best" if mid == best_id else ""
        logger.info("%-25s  %.4f%s", name, auc, flag)

    # 6. promotion policy
    if not champion_exists():
        # cold start: nothing to compare against, so the best candidate goes live
        save_champion(best_id, promoted_by="bootstrap",
                      reason="first training run: no champion in the model store")
        logger.info("champion registered (bootstrap): %s", best_id)
        return

    # retrain: register as challenger only; a person promotes via the model_deploy DAG
    champion_id = load_champion_id()
    champion_model, champion_cols, _ = load_model(champion_id)
    report = build_report(
        champion_id, champion_model, champion_cols, load_shap_top(champion_id),
        best_id, best_pipeline, feature_cols, load_shap_top(best_id),
        oot_df=test, window=f"loan_start_date >= {TRAIN_CUTOFF}",
    )
    save_challenger_report(best_id, report)
    logger.info("challenger %s vs champion %s on %s (%d loans)",
                best_id, champion_id, report["window"], report["n_loans"])
    logger.info("  Gini  champion %s  challenger %.4f", report["champion_gini"], report["challenger_gini"])
    for c in report["checks"]:
        status = {True: "PASS", False: "FAIL", None: "n/a"}[c["passed"]]
        logger.info("  %-16s %-6s value=%s  rule %s", c["name"], status, c["value"], c["rule"])
    if report["recommendation"] == "PROMOTE":
        logger.info("recommendation: PROMOTE — champion.json NOT changed. After review, trigger "
                    "model_deploy with model_id=%s and approved_by=<name>", best_id)
    else:
        logger.info("recommendation: KEEP_CHAMPION — %s stays in production; challenger %s is "
                    "archived with its report for audit", champion_id, best_id)


if __name__ == "__main__":
    main()
