"""Unit tests for the promotion gate and the champion audit trail.

Runs without Spark.

    python -m pytest tests/unit/test_promotion.py -v
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import utils.ml.model_store as ms
from utils.ml.promotion import build_report, decide

LIMITS = {"min_gini_uplift": 0.02, "max_score_psi": 0.10, "min_shap_top10_overlap": 5}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "_STORE", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# decide()
# ---------------------------------------------------------------------------

def test_decide_promotes_when_all_checks_pass():
    checks, rec = decide(0.60, 0.63, 0.02, 7, LIMITS)
    assert rec == "PROMOTE"
    assert all(c["passed"] for c in checks)


def test_decide_keeps_champion_on_noise_level_uplift():
    # +0.01 Gini is inside one CV std: not enough to replace the champion
    _, rec = decide(0.60, 0.61, 0.02, 7, LIMITS)
    assert rec == "KEEP_CHAMPION"


def test_decide_keeps_champion_when_challenger_unstable():
    _, rec = decide(0.60, 0.70, 0.30, 7, LIMITS)
    assert rec == "KEEP_CHAMPION"


def test_decide_keeps_champion_when_drivers_change():
    _, rec = decide(0.60, 0.70, 0.02, 2, LIMITS)
    assert rec == "KEEP_CHAMPION"


def test_decide_missing_shap_does_not_block():
    checks, rec = decide(0.60, 0.63, 0.02, None, LIMITS)
    assert rec == "PROMOTE"
    assert next(c for c in checks if c["name"] == "shap_sanity")["passed"] is None


def test_decide_unscorable_champion_blocks():
    _, rec = decide(None, 0.70, 0.02, 7, LIMITS)
    assert rec == "KEEP_CHAMPION"


# ---------------------------------------------------------------------------
# build_report()
# ---------------------------------------------------------------------------

def test_build_report_scores_both_models_on_same_window():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["label"] = (df["a"] + 0.3 * rng.normal(size=n) > 0).astype(int)
    df["loan_start_date"] = pd.date_range("2024-07-01", periods=n, freq="D").date

    weak = LogisticRegression().fit(df[["b"]].values, df["label"].values)
    strong = LogisticRegression().fit(df[["a", "b"]].values, df["label"].values)
    report = build_report("champ", weak, ["b"], None, "chal", strong, ["a", "b"], None,
                          oot_df=df, window="test")

    assert report["n_loans"] == n
    assert report["challenger_gini"] > report["champion_gini"]
    assert report["champion_id"] == "champ" and report["challenger_id"] == "chal"
    assert report["recommendation"] in ("PROMOTE", "KEEP_CHAMPION")


def test_build_report_flags_champion_with_missing_columns():
    df = pd.DataFrame({"a": [0.1, 0.9, 0.2, 0.8], "label": [0, 1, 0, 1],
                       "loan_start_date": pd.date_range("2024-07-01", periods=4).date})
    model = LogisticRegression().fit(df[["a"]].values, df["label"].values)
    report = build_report("champ", model, ["gone"], None, "chal", model, ["a"], None,
                          oot_df=df, window="test")
    assert report["champion_gini"] is None
    assert report["recommendation"] == "KEEP_CHAMPION"
    assert "gone" in report["note"]


# ---------------------------------------------------------------------------
# champion pointer + history
# ---------------------------------------------------------------------------

def test_save_champion_writes_pointer_and_history(store):
    ms.save_champion("m1", promoted_by="bootstrap", reason="first run")
    ms.save_champion("m2", promoted_by="manual", approved_by="risk team")

    assert ms.load_champion_id() == "m2"
    champ = ms.load_champion()
    assert champ["previous_model_id"] == "m1"
    assert champ["approved_by"] == "risk team"
    history = ms.load_champion_history()
    assert [r["model_id"] for r in history] == ["m1", "m2"]
    assert history[0]["promoted_by"] == "bootstrap"


def test_save_champion_rejects_unknown_source(store):
    with pytest.raises(ValueError):
        ms.save_champion("m1", promoted_by="auto")


def test_rollback_restores_previous_champion(store):
    ms.save_champion("m1", promoted_by="bootstrap")
    ms.save_champion("m2", promoted_by="manual", approved_by="x")
    ms.save_champion("m1", promoted_by="rollback", approved_by="x")
    assert ms.load_champion_id() == "m1"
    assert ms.load_champion()["previous_model_id"] == "m2"


def test_challenger_report_round_trip(store):
    assert ms.load_challenger_report("m9") is None
    ms.save_challenger_report("m9", {"recommendation": "PROMOTE"})
    assert ms.load_challenger_report("m9") == {"recommendation": "PROMOTE"}
    assert not (store / "m9" / "challenger_report.json.tmp").exists()


def test_old_style_champion_json_still_loads(store):
    # champion.json files written before the audit fields existed
    (store / "champion.json").write_text(json.dumps({"model_id": "legacy", "updated_at": "x"}))
    assert ms.load_champion_id() == "legacy"
    assert ms.save_champion("m2", promoted_by="manual", approved_by="x")["previous_model_id"] == "legacy"
