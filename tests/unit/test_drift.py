"""Unit tests for feature drift and the SOP traffic light (utils/ml/drift.py). No Spark."""

import math

import numpy as np
import pandas as pd

from utils.ml.drift import (
    classify, decision_flag_gap, decision_gini, feature_drift, feature_psi, model_age_months,
)

LIM = {"psi_amber": 0.10, "psi_red": 0.25, "gini_amber": 0.57, "gini_red": 0.50,
       "feature_psi_major": 0.25, "refresh_months": 6,
       "flag_gap_over_amber": 0.10, "flag_gap_under_amber": 0.05}
rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# feature_psi
# ---------------------------------------------------------------------------

def test_feature_psi_same_distribution_is_small():
    a, b = rng.normal(size=5000), rng.normal(size=5000)
    assert feature_psi(a, b) < 0.02


def test_feature_psi_shifted_distribution_is_large():
    a, b = rng.normal(size=5000), rng.normal(loc=1.0, size=5000)
    assert feature_psi(a, b) > 0.25


def test_feature_psi_detects_jump_in_missingness():
    a = rng.normal(size=5000)
    b = rng.normal(size=5000)
    b[:2500] = np.nan                     # half the values suddenly missing
    assert feature_psi(a, b) > 0.25


def test_feature_psi_discrete_feature_and_unseen_value():
    a = rng.integers(0, 2, size=4000).astype(float)
    assert feature_psi(a, rng.integers(0, 2, size=4000).astype(float)) < 0.02
    assert feature_psi(a, np.full(4000, 5.0)) > 1.0   # value never seen in training


def test_feature_psi_empty_side_is_nan():
    assert math.isnan(feature_psi([], [1.0, 2.0]))


def test_feature_drift_sorted_highest_first():
    ref = pd.DataFrame({"stable": rng.normal(size=3000), "moved": rng.normal(size=3000)})
    cur = pd.DataFrame({"stable": rng.normal(size=3000), "moved": rng.normal(2, 1, size=3000)})
    out = feature_drift(ref, cur, ["stable", "moved"])
    assert list(out) == ["moved", "stable"]


# ---------------------------------------------------------------------------
# decision_gini / model_age_months
# ---------------------------------------------------------------------------

def test_decision_gini_uses_last_three_labelled_months():
    assert decision_gini([0.9, 0.60, float("nan"), 0.62, 0.64]) == round((0.60 + 0.62 + 0.64) / 3, 4)
    assert math.isnan(decision_gini([float("nan")]))


def test_model_age_months():
    assert model_age_months("2025-01-01", "2024-07-01") == 6
    assert model_age_months("2024-07-01", "2024-07-01") == 0


# ---------------------------------------------------------------------------
# classify (SOP traffic light)
# ---------------------------------------------------------------------------

def test_green_when_all_within_limits():
    assert classify(0.03, 0.63, 0, None, None, LIM)[0] == "GREEN"


def test_amber_on_minor_score_shift():
    status, reason = classify(0.14, 0.63, 0, None, None, LIM)
    assert status == "AMBER" and "score PSI" in reason


def test_amber_on_gini_band_and_on_feature_drift():
    assert classify(0.03, 0.55, 0, None, None, LIM)[0] == "AMBER"
    status, reason = classify(0.03, 0.63, 2, "income 0.31", None, LIM)
    assert status == "AMBER" and "income" in reason


def test_red_on_major_shift_or_low_gini():
    assert classify(0.30, 0.63, 0, None, None, LIM)[0] == "RED"
    assert classify(0.03, 0.45, 0, None, None, LIM)[0] == "RED"


def test_second_consecutive_amber_escalates_to_red():
    status, reason = classify(0.14, 0.63, 0, None, "AMBER", LIM)
    assert status == "RED" and "consecutive" in reason


def test_missing_metrics_are_skipped_not_alarmed():
    # first OOT month: no PSI reference yet; labels not matured: no Gini
    assert classify(float("nan"), float("nan"), 0, None, None, LIM)[0] == "GREEN"


# ---------------------------------------------------------------------------
# flag gap (calibration): predicted bad rate at 0.5 minus actual bad rate
# ---------------------------------------------------------------------------

def test_small_over_flagging_stays_green():
    # the champion over-flags by about 6 points: safer than missing bad loans, so no alert
    assert classify(0.03, 0.63, 0, None, None, LIM, flag_gap_3m=0.06)[0] == "GREEN"
    assert decision_flag_gap([0.07, float("nan"), 0.04, 0.06, 0.08]) == 0.06


def test_flag_gap_band_is_asymmetric():
    status, reason = classify(0.03, 0.63, 0, None, None, LIM, flag_gap_3m=0.12)
    assert status == "AMBER" and "over-flagging" in reason
    # a 6-point miss alerts while a 6-point over-flag does not: missing bad loans costs more
    status, reason = classify(0.03, 0.63, 0, None, None, LIM, flag_gap_3m=-0.06)
    assert status == "AMBER" and "under-flagging" in reason
