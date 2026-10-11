"""Unit tests for silver cleaning rules, using values seen in the raw data.

    python -m pytest tests
"""

import pytest

from utils.transformation.cleaning import (
    clean_numeric, null_outside, null_if_invalid_category, duration_to_months, split_loan_types,
)
from utils.transformation.validation import validate_schema, validate_unique_key
from utils.features.feature_defs import loan_type_feature
from utils.common.dates import month_starts


def test_month_starts_is_inclusive_and_crosses_years():
    assert month_starts("2024-11-01", "2025-02-01") == ["2024-11-01", "2024-12-01", "2025-01-01", "2025-02-01"]


def _apply(spark, values, expr_fn):
    df = spark.createDataFrame([(v,) for v in values], "x string")
    return [row[0] for row in df.select(expr_fn("x")).collect()]


def test_clean_numeric_strips_underscores_and_nulls_junk(spark):
    assert _apply(spark, ["52312.68_", "40_", "-5", "_", "__10000__", None], clean_numeric) == \
        [pytest.approx(52312.68), 40.0, -5.0, None, 10000.0, None]


def test_null_outside_keeps_only_plausible_range(spark):
    df = spark.createDataFrame([(-500,), (14,), (33,), (100,), (8678,)], "x int")
    assert [r[0] for r in df.select(null_outside("x", 14, 100)).collect()] == [None, 14, 33, 100, None]


def test_null_if_invalid_category_removes_placeholders(spark):
    assert _apply(spark, ["Lawyer", "Media_Manager", "_______", "_", "!@9#%8"], null_if_invalid_category) == \
        ["Lawyer", "Media_Manager", None, None, None]


def test_duration_to_months(spark):
    assert _apply(spark, ["10 Years and 9 Months", "31 Years and 0 Months", None], duration_to_months) == [129, 372, None]


def test_split_loan_types(spark):
    result = _apply(spark, ["Auto Loan, and Payday Loan", "Not Specified", None], split_loan_types)
    assert result == [["Auto Loan", "Payday Loan"], ["Not Specified"], []]


def test_loan_type_feature_names():
    assert loan_type_feature("Credit-Builder Loan") == "loan_type_credit_builder"
    assert loan_type_feature("Debt Consolidation Loan") == "loan_type_debt_consolidation"


def test_validate_schema_rejects_drift(spark):
    df = spark.createDataFrame([("CUS_1", 30)], "Customer_ID string, Age int")
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_schema(df, "silver", "attributes")


def test_validate_unique_key_rejects_duplicates(spark):
    df = spark.createDataFrame([("CUS_1", "2024-01-01"), ("CUS_1", "2024-01-01")], "Customer_ID string, snapshot_date string")
    with pytest.raises(ValueError, match="duplicate keys"):
        validate_unique_key(df, ["Customer_ID", "snapshot_date"], "test")
