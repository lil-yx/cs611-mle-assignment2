"""Challenger-vs-champion promotion gate.

A retrained model never replaces the champion on its own. Training writes a
challenger_report.json with the checks below and a recommendation; a person then
promotes it (or not) through the model_deploy DAG, which re-reads this report.

Both models are scored on the SAME out-of-time window (loans on or after the current
train_cutoff), so the comparison is like for like.

Checks (thresholds in config/base.yaml -> ml.promotion):
    gini_uplift     challenger Gini - champion Gini >= min_gini_uplift
                    (0.02 ~ one CV standard deviation of Gini, so a "win" is not just noise)
    score_stability PSI of the challenger's scores, early vs late half of the window,
                    < max_score_psi (the challenger is stable across time)
    shap_sanity     at least min_shap_top10_overlap of the champion's top-10 SHAP
                    features are also in the challenger's top 10 (no surprise drivers)
"""

from datetime import datetime, UTC

import numpy as np
from sklearn.metrics import roc_auc_score

from utils.common.config import CONFIG
from utils.ml.stability import psi

_DEFAULTS = {"min_gini_uplift": 0.02, "max_score_psi": 0.10, "min_shap_top10_overlap": 5}


def thresholds():
    return {**_DEFAULTS, **CONFIG["ml"].get("promotion", {})}


def _gini(y, scores):
    return round(float(2 * roc_auc_score(y, scores) - 1), 4)


def decide(champion_gini, challenger_gini, challenger_psi, shap_overlap, limits=None):
    """Apply the gate. Returns (checks, recommendation).

    A check whose value is None could not be computed: gini_uplift then fails (the
    gate cannot be argued without it), shap_sanity is reported but does not block.
    """
    limits = limits or thresholds()
    delta = None if champion_gini is None else round(challenger_gini - champion_gini, 4)
    checks = [
        {
            "name": "gini_uplift", "value": delta,
            "rule": f">= {limits['min_gini_uplift']}",
            "passed": delta is not None and delta >= limits["min_gini_uplift"],
        },
        {
            "name": "score_stability", "value": challenger_psi,
            "rule": f"< {limits['max_score_psi']}",
            "passed": challenger_psi is not None and challenger_psi < limits["max_score_psi"],
        },
        {
            "name": "shap_sanity", "value": shap_overlap,
            "rule": f">= {limits['min_shap_top10_overlap']} of top 10",
            "passed": None if shap_overlap is None else shap_overlap >= limits["min_shap_top10_overlap"],
        },
    ]
    blocking = [c for c in checks if c["passed"] is not None]
    recommendation = "PROMOTE" if all(c["passed"] for c in blocking) else "KEEP_CHAMPION"
    return checks, recommendation


def build_report(champion_id, champion_model, champion_cols, champion_shap_top,
                 challenger_id, challenger_model, challenger_cols, challenger_shap_top,
                 oot_df, window):
    """Score champion and challenger on oot_df (pandas, with 'label' and
    'loan_start_date') and return the challenger_report.json payload."""
    y = oot_df["label"].values
    challenger_scores = challenger_model.predict_proba(oot_df[challenger_cols].values)[:, 1]
    challenger_gini = _gini(y, challenger_scores)

    champion_gini, champion_note = None, None
    missing = [c for c in champion_cols if c not in oot_df.columns]
    if missing:
        champion_note = f"champion needs columns no longer in the feature store: {missing[:5]}"
    else:
        champion_gini = _gini(y, champion_model.predict_proba(oot_df[champion_cols].values)[:, 1])

    # stability: early half of the window is the reference, late half the current period
    order = np.argsort(oot_df["loan_start_date"].values, kind="stable")
    half = len(order) // 2
    challenger_psi = round(psi(challenger_scores[order[:half]], challenger_scores[order[half:]]), 4)

    overlap = None
    if champion_shap_top and challenger_shap_top:
        overlap = len(set(champion_shap_top) & set(challenger_shap_top))

    checks, recommendation = decide(champion_gini, challenger_gini, challenger_psi, overlap)
    return {
        "challenger_id":   challenger_id,
        "champion_id":     champion_id,
        "window":          window,
        "n_loans":         int(len(y)),
        "champion_gini":   champion_gini,
        "challenger_gini": challenger_gini,
        "checks":          checks,
        "recommendation":  recommendation,
        "note":            champion_note,
        "generated_at":    datetime.now(UTC).isoformat(),
    }
