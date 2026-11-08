"""Feature drift and the monthly traffic-light status (docs/governance_sop.png).

Feature PSI
    Each model input is compared with the training window. Bins are deciles of the
    reference (training) values; low-cardinality features use one bin per value; missing
    values get their own bin, so a jump in missingness counts as drift too.

Traffic light (thresholds in config/base.yaml -> ml.monitoring)
    RED    score PSI >= 0.25, or decision Gini < 0.50,
           or a second consecutive AMBER
    AMBER  score PSI 0.10-0.25, or decision Gini 0.50-0.57,
           or any model input with feature PSI >= 0.25,
           or the 3-month flag gap (predicted minus actual bad rate) above +0.10
           or below -0.05
    GREEN  otherwise
    The decision Gini and the flag gap are means of the last 3 months with labels, so one
    noisy month does not trigger a retrain. The flag-gap band is asymmetric on purpose:
    flagging a few too many loans costs some good customers, but missing bad loans costs
    credit losses, so under-flagging alerts sooner. Status never retrains anything: RED means "open a retrain
    ticket", and a person runs the model_training DAG.

Refresh
    refresh_due is set once the model's training data is refresh_months old
    (snapshot month - train_cutoff), independent of the colour.
"""

import math

import numpy as np

from utils.common.config import CONFIG

_DEFAULTS = {
    "psi_amber": 0.10, "psi_red": 0.25,
    "gini_amber": 0.57, "gini_red": 0.50,
    "feature_psi_major": 0.25, "refresh_months": 6,
    "flag_gap_over_amber": 0.10, "flag_gap_under_amber": 0.05,
}


def limits():
    return {**_DEFAULTS, **CONFIG["ml"].get("monitoring", {})}


def _isnan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def feature_psi(reference, current, n_bins=10, eps=1e-4):
    """PSI of one feature, current vs reference. NaN if either side is empty."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    ref_nan, cur_nan = np.isnan(ref), np.isnan(cur)
    ref_val, cur_val = ref[~ref_nan], cur[~cur_nan]

    values = np.unique(ref_val)
    if len(values) <= n_bins:
        # discrete: one bin per reference value, plus a bin for values never seen in training
        def counts(v):
            return np.array([(v == u).sum() for u in values] + [(~np.isin(v, values)).sum()])
    else:
        edges = np.unique(np.quantile(ref_val, np.linspace(0, 1, n_bins + 1)[1:-1]))

        def counts(v):
            return np.bincount(np.searchsorted(edges, v, side="right"), minlength=len(edges) + 1)

    ref_counts = np.append(counts(ref_val), ref_nan.sum())
    cur_counts = np.append(counts(cur_val), cur_nan.sum())
    ref_pct = np.maximum(ref_counts / len(ref), eps)
    cur_pct = np.maximum(cur_counts / len(cur), eps)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def feature_drift(reference_df, current_df, features):
    """{feature: psi} for every model input, highest first."""
    out = {f: round(feature_psi(reference_df[f].values, current_df[f].values), 4) for f in features}
    return dict(sorted(out.items(), key=lambda kv: -(kv[1] if not _isnan(kv[1]) else -1)))


def decision_gini(gini_history):
    """Mean of the last 3 non-missing monthly Ginis (oldest first), or NaN."""
    recent = [g for g in gini_history if not _isnan(g)][-3:]
    return round(float(np.mean(recent)), 4) if recent else float("nan")


def decision_flag_gap(gap_history):
    """Mean of the last 3 non-missing monthly flag gaps (predicted minus actual bad rate)."""
    return decision_gini(gap_history)


def classify(score_psi, gini_3m, n_features_major, top_feature, prev_status, lim=None,
             flag_gap_3m=float("nan")):
    """Return (status, reason) for one month."""
    lim = lim or limits()
    level, reasons = 0, []

    def flag(new_level, reason):
        nonlocal level
        level = max(level, new_level)
        reasons.append(reason)

    if not _isnan(score_psi):
        if score_psi >= lim["psi_red"]:
            flag(2, f"score PSI {score_psi:.3f} >= {lim['psi_red']}")
        elif score_psi >= lim["psi_amber"]:
            flag(1, f"score PSI {score_psi:.3f} >= {lim['psi_amber']}")
    if not _isnan(gini_3m):
        if gini_3m < lim["gini_red"]:
            flag(2, f"3-month Gini {gini_3m:.3f} < {lim['gini_red']}")
        elif gini_3m < lim["gini_amber"]:
            flag(1, f"3-month Gini {gini_3m:.3f} < {lim['gini_amber']}")
    if not _isnan(flag_gap_3m):
        if flag_gap_3m > lim["flag_gap_over_amber"]:
            flag(1, f"3-month flag gap {flag_gap_3m:+.3f} > +{lim['flag_gap_over_amber']} (over-flagging)")
        elif flag_gap_3m < -lim["flag_gap_under_amber"]:
            flag(1, f"3-month flag gap {flag_gap_3m:+.3f} < -{lim['flag_gap_under_amber']} (under-flagging)")
    if n_features_major:
        flag(1, f"{n_features_major} input(s) with feature PSI >= {lim['feature_psi_major']} "
                f"(top: {top_feature})")

    status = ("GREEN", "AMBER", "RED")[level]
    if status == "AMBER" and prev_status == "AMBER":
        status = "RED"
        reasons.append("second consecutive AMBER")
    return status, "; ".join(reasons) or "all checks within limits"


def model_age_months(snapshot_date, train_cutoff):
    """Whole months between the end of the training data and the scored month."""
    s_year, s_month = int(snapshot_date[:4]), int(snapshot_date[5:7])
    c_year, c_month = int(str(train_cutoff)[:4]), int(str(train_cutoff)[5:7])
    return (s_year - c_year) * 12 + (s_month - c_month)
