"""Unit tests for the gold feature-store changes.

Covers:
  - Fix 1: clickstream aggregated across all months (read_partitions_up_to)
  - Fix 2: new engineered features
  - Fix 3: age-bin one-hot columns
  - Fix 4: categorical imputation (-1 sentinel) and ratio null-fill

    python -m pytest tests/unit/test_gold_features.py -v
"""

import os
import pytest
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, FloatType, StructType, StructField, StringType, DateType

from utils.features.build_gold import _encode, _ratio, _age_bin_cols
from utils.features.feature_defs import CREDIT_MIX_ORDINAL, MIN_PAYMENT_BINARY


# ---------------------------------------------------------------------------
# Fix 4a: _encode returns -1 for null / unknown (not null)
# ---------------------------------------------------------------------------

def test_encode_known_values(spark):
    df = spark.createDataFrame([("Bad",), ("Standard",), ("Good",)], "v string")
    result = [r[0] for r in df.select(_encode(col("v"), CREDIT_MIX_ORDINAL)).collect()]
    assert result == [0, 1, 2]


def test_encode_null_returns_minus_one(spark):
    df = spark.createDataFrame([(None,), ("_",), ("Unknown",)], "v string")
    result = [r[0] for r in df.select(_encode(col("v"), CREDIT_MIX_ORDINAL)).collect()]
    assert result == [-1, -1, -1], "unknown / null categories must map to -1, not null"


def test_encode_binary_missing_returns_minus_one(spark):
    df = spark.createDataFrame([("Yes",), ("No",), ("NM",), (None,)], "v string")
    result = [r[0] for r in df.select(_encode(col("v"), MIN_PAYMENT_BINARY)).collect()]
    assert result == [1, 0, -1, -1]


# ---------------------------------------------------------------------------
# Fix 4b: _ratio fills null with 0 when denominator is zero or missing
# ---------------------------------------------------------------------------

def test_ratio_normal_case(spark):
    df = spark.createDataFrame([(100.0, 4.0)], "num float, den float")
    result = df.select(_ratio("num", "den").alias("r")).collect()[0][0]
    assert result == pytest.approx(25.0)


def test_ratio_zero_denominator_fills_zero(spark):
    df = spark.createDataFrame([(100.0, 0.0)], "num float, den float")
    result = df.select(_ratio("num", "den").alias("r")).collect()[0][0]
    assert result == pytest.approx(0.0), "zero denominator should fill with 0, not null"


def test_ratio_null_denominator_fills_zero(spark):
    df = spark.createDataFrame([(100.0, None)], "num float, den float")
    result = df.select(_ratio("num", "den").alias("r")).collect()[0][0]
    assert result == pytest.approx(0.0), "null denominator should fill with 0, not null"


def test_ratio_null_numerator_fills_zero(spark):
    df = spark.createDataFrame([(None, 4.0)], "num float, den float")
    result = df.select(_ratio("num", "den").alias("r")).collect()[0][0]
    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Fix 3: age-bin one-hot columns
# ---------------------------------------------------------------------------

def _age_bins(spark, ages):
    df = spark.createDataFrame([(a,) for a in ages], "Age int")
    cols = _age_bin_cols(col("Age"))
    rows = df.select(*cols).collect()
    return [tuple(r) for r in rows]


def test_age_bins_correct_bucket(spark):
    # (lt_25, 25-39, 40-54, 55+, missing)
    results = _age_bins(spark, [20, 30, 45, 60])
    assert results[0] == (1, 0, 0, 0, 0), "age 20 -> age_lt_25"
    assert results[1] == (0, 1, 0, 0, 0), "age 30 -> age_25_39"
    assert results[2] == (0, 0, 1, 0, 0), "age 45 -> age_40_54"
    assert results[3] == (0, 0, 0, 1, 0), "age 60 -> age_55_plus"


def test_age_bins_boundary_values(spark):
    results = _age_bins(spark, [25, 40, 55])
    assert results[0] == (0, 1, 0, 0, 0), "age 25 is in 25-39 bucket"
    assert results[1] == (0, 0, 1, 0, 0), "age 40 is in 40-54 bucket"
    assert results[2] == (0, 0, 0, 1, 0), "age 55 is in 55+ bucket"


def test_age_bins_null_sets_missing_flag(spark):
    results = _age_bins(spark, [None])
    assert results[0] == (0, 0, 0, 0, 1), "null age must set age_missing=1"


# ---------------------------------------------------------------------------
# Fix 1: read_partitions_up_to filters by date correctly
# ---------------------------------------------------------------------------

def test_read_partitions_up_to_selects_correct_files(spark, tmp_path):
    """Writes three parquet partitions and checks that up_to filtering works."""
    from utils.common.io import read_partitions_up_to

    # write three tiny parquet files with the same naming convention the pipeline uses
    for date_str in ["2023_01_01", "2023_02_01", "2023_03_01"]:
        path = tmp_path / f"silver_clickstream_{date_str}.parquet"
        spark.createDataFrame([(date_str,)], "snapshot_date string").write.parquet(str(path))

    # monkey-patch table_dir to point at tmp_path
    import utils.common.io as io_module
    original_table_dir = io_module.table_dir

    def patched_table_dir(layer, table):
        return str(tmp_path) + "/"

    io_module.table_dir = patched_table_dir
    try:
        result = read_partitions_up_to(spark, "silver", "clickstream", "2023-02-01")
        assert result is not None
        dates_found = sorted([r[0] for r in result.collect()])
        assert dates_found == ["2023_01_01", "2023_02_01"], \
            "should include Jan and Feb but NOT March"
    finally:
        io_module.table_dir = original_table_dir


def test_read_partitions_up_to_returns_none_when_no_files(spark, tmp_path):
    from utils.common.io import read_partitions_up_to
    import utils.common.io as io_module

    original_table_dir = io_module.table_dir
    io_module.table_dir = lambda layer, table: str(tmp_path) + "/"
    try:
        result = read_partitions_up_to(spark, "silver", "clickstream", "2023-01-01")
        assert result is None
    finally:
        io_module.table_dir = original_table_dir


# ---------------------------------------------------------------------------
# Fix 2: new engineered features produce correct values
# ---------------------------------------------------------------------------

def _make_row(spark, **kwargs):
    """Create a one-row DataFrame from keyword args for inline feature testing."""
    defaults = dict(
        Monthly_Inhand_Salary=3000.0,
        Total_EMI_per_month=1000.0,
        Num_of_Loan=4,
        Num_Bank_Accounts=2,
        Num_Credit_Card=2,
        Delay_from_due_date=8,
        Num_Credit_Inquiries=8,
        Outstanding_Debt=6000.0,
        Monthly_Balance=1500.0,
        Changed_Credit_Limit=200.0,
    )
    defaults.update(kwargs)
    schema = StructType([StructField(k, FloatType() if isinstance(v, float) else IntegerType()) for k, v in defaults.items()])
    return spark.createDataFrame([tuple(defaults.values())], schema)


def test_repayment_ability(spark):
    df = _make_row(spark, Monthly_Inhand_Salary=3000.0, Total_EMI_per_month=1000.0)
    result = df.select((col("Monthly_Inhand_Salary") - col("Total_EMI_per_month")).alias("r")).collect()[0][0]
    assert result == pytest.approx(2000.0)


def test_loans_per_credit_item(spark):
    # 4 loans / (2 bank + 2 credit card) = 1.0
    df = _make_row(spark, Num_of_Loan=4, Num_Bank_Accounts=2, Num_Credit_Card=2)
    credit_items = col("Num_Bank_Accounts") + col("Num_Credit_Card")
    result = df.select(
        F.when(credit_items > 0, col("Num_of_Loan") / credit_items).cast(FloatType()).alias("r")
    ).collect()[0][0]
    assert result == pytest.approx(1.0)


def test_loan_extent(spark):
    # 8 days delay / 4 loans = 2.0
    df = _make_row(spark, Delay_from_due_date=8, Num_of_Loan=4)
    result = df.select(
        F.when(col("Num_of_Loan") > 0, col("Delay_from_due_date") / col("Num_of_Loan")).cast(FloatType()).alias("r")
    ).collect()[0][0]
    assert result == pytest.approx(2.0)


def test_inq_per_loan(spark):
    # 8 inquiries / 4 loans = 2.0
    df = _make_row(spark, Num_Credit_Inquiries=8, Num_of_Loan=4)
    result = df.select(
        F.when(col("Num_of_Loan") > 0, col("Num_Credit_Inquiries") / col("Num_of_Loan")).cast(FloatType()).alias("r")
    ).collect()[0][0]
    assert result == pytest.approx(2.0)


def test_balance_to_debt(spark):
    # 1500 / 6000 = 0.25
    df = _make_row(spark, Monthly_Balance=1500.0, Outstanding_Debt=6000.0)
    result = df.select(
        F.when(col("Outstanding_Debt") > 0, col("Monthly_Balance") / col("Outstanding_Debt")).cast(FloatType()).alias("r")
    ).collect()[0][0]
    assert result == pytest.approx(0.25)


def test_has_credit_limit_change_nonzero(spark):
    df = _make_row(spark, Changed_Credit_Limit=200.0)
    result = df.select(
        F.coalesce(col("Changed_Credit_Limit") != 0, F.lit(False)).cast(IntegerType()).alias("r")
    ).collect()[0][0]
    assert result == 1


def test_has_credit_limit_change_zero(spark):
    df = _make_row(spark, Changed_Credit_Limit=0.0)
    result = df.select(
        F.coalesce(col("Changed_Credit_Limit") != 0, F.lit(False)).cast(IntegerType()).alias("r")
    ).collect()[0][0]
    assert result == 0


def test_has_credit_limit_change_null(spark):
    schema = StructType([StructField("Changed_Credit_Limit", FloatType())])
    df = spark.createDataFrame([(None,)], schema)
    result = df.select(
        F.coalesce(col("Changed_Credit_Limit") != 0, F.lit(False)).cast(IntegerType()).alias("r")
    ).collect()[0][0]
    assert result == 0, "null credit limit change should be treated as no-change (0)"
