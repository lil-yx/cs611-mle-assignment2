"""Silver -> Gold: label store and ML-ready feature store."""

import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, FloatType

from utils.common.io import read_parquet, read_partitions_up_to
from utils.transformation.validation import validate_and_write
from utils.features.feature_defs import (
    OCCUPATIONS, LOAN_TYPES, CREDIT_MIX_ORDINAL, MIN_PAYMENT_BINARY, SPEND_LEVEL_ORDINAL,
    PAYMENT_VALUE_ORDINAL, CLICKSTREAM_COLUMNS, NUMERIC_FEATURES,
    occupation_feature, loan_type_feature,
)

KEYS = ["Customer_ID", "snapshot_date"]


def _encode(column, mapping):
    """Map category -> integer; unknown or null -> -1 (explicit missing sentinel)."""
    lookup = F.create_map(*[F.lit(x) for pair in mapping.items() for x in pair])
    return F.coalesce(lookup[column], F.lit(-1)).cast(IntegerType())


def _ratio(numerator, denominator, fill=0.0):
    """Ratio of two column names; fills null (zero/missing denominator) with fill."""
    return F.coalesce(
        F.when(col(denominator) > 0, col(numerator) / col(denominator)).cast(FloatType()),
        F.lit(float(fill)).cast(FloatType()),
    )


def process_labels_gold_table(snapshot_date, spark, dpd, mob):
    df = read_parquet(spark, "silver", "lms_loan_daily", snapshot_date).filter(col("mob") == mob)

    df = df.select(
        "loan_id",
        "Customer_ID",
        "loan_start_date",  # join key to the feature store
        (col("dpd") >= dpd).cast(IntegerType()).alias("label"),
        F.lit(f"{dpd}dpd_{mob}mob").alias("label_def"),
        "snapshot_date",
    )

    validate_and_write(df, "gold", "label_store", snapshot_date, ["loan_id"])
    return df


def _load_clickstream(spark, snapshot_date, customers):
    """Aggregate all clickstream partitions up to snapshot_date per customer.

    Takes the mean of fe_1..fe_20 across every available month, so customers
    with multiple months of history contribute richer behavioural signals than
    those who only appeared in the application month.  The has_clickstream flag
    is 1 for any customer with at least one month of data.
    """
    clickstream_hist = read_partitions_up_to(spark, "silver", "clickstream", snapshot_date)
    if clickstream_hist is not None:
        agg_exprs = [F.mean(c).cast(FloatType()).alias(c) for c in CLICKSTREAM_COLUMNS]
        clickstream_agg = (
            clickstream_hist
            .join(customers.select("Customer_ID").distinct(), on="Customer_ID", how="inner")
            .groupBy("Customer_ID")
            .agg(*agg_exprs, F.lit(1).alias("has_clickstream"))
        )
        df = customers.join(clickstream_agg, on="Customer_ID", how="left")
    else:
        df = customers.select(
            *customers.columns,
            *[F.lit(None).cast(FloatType()).alias(c) for c in CLICKSTREAM_COLUMNS],
            F.lit(0).alias("has_clickstream"),
        )
    return df.withColumn("has_clickstream", F.coalesce(col("has_clickstream"), F.lit(0)).cast(IntegerType()))


def _age_bin_cols(age_col):
    """One-hot age group indicators plus an explicit missing flag."""
    return [
        F.coalesce(age_col < 25, F.lit(False)).cast(IntegerType()).alias("age_lt_25"),
        F.coalesce((age_col >= 25) & (age_col < 40), F.lit(False)).cast(IntegerType()).alias("age_25_39"),
        F.coalesce((age_col >= 40) & (age_col < 55), F.lit(False)).cast(IntegerType()).alias("age_40_54"),
        F.coalesce(age_col >= 55, F.lit(False)).cast(IntegerType()).alias("age_55_plus"),
        age_col.isNull().cast(IntegerType()).alias("age_missing"),
    ]


def process_features_gold_table(snapshot_date, spark):
    """One partition: every customer's features as known at snapshot_date."""
    attributes = read_parquet(spark, "silver", "attributes", snapshot_date)
    financials = read_parquet(spark, "silver", "financials", snapshot_date)
    df = _load_clickstream(spark, snapshot_date, attributes.join(financials, on=KEYS, how="inner"))

    behaviour = F.split(col("Payment_Behaviour"), "_")
    occupation_is = lambda o: F.coalesce(col("Occupation") == o, F.lit(False)).cast(IntegerType())
    loan_type_count = lambda t: F.size(F.filter("Loan_Type_List", lambda x: x == t))
    age = col("Age")
    credit_items = col("Num_Bank_Accounts") + col("Num_Credit_Card")

    df = df.select(
        *KEYS,

        # --- raw numerics ---
        *[col(src).alias(dst) for src, dst in NUMERIC_FEATURES.items()],

        # --- ordinal / binary categoricals (null -> -1 missing sentinel) ---
        _encode(col("Credit_Mix"), CREDIT_MIX_ORDINAL).alias("credit_mix"),
        _encode(col("Payment_of_Min_Amount"), MIN_PAYMENT_BINARY).alias("pays_min_amount_only"),
        _encode(behaviour.getItem(0), SPEND_LEVEL_ORDINAL).alias("spend_level"),
        _encode(behaviour.getItem(2), PAYMENT_VALUE_ORDINAL).alias("payment_value"),

        # --- financial ratios (null filled with 0: conservative no-burden default) ---
        _ratio("Outstanding_Debt", "Annual_Income").alias("debt_to_annual_income"),
        _ratio("Total_EMI_per_month", "Monthly_Inhand_Salary").alias("emi_to_monthly_salary"),
        _ratio("Amount_invested_monthly", "Monthly_Inhand_Salary").alias("invested_to_monthly_salary"),

        # --- new engineered features ---
        # absolute cash left after debt service; null when salary or EMI unknown
        (col("Monthly_Inhand_Salary") - col("Total_EMI_per_month")).cast(FloatType()).alias("repayment_ability"),
        # how many loans per credit product; signals concentration of debt
        F.when(credit_items > 0, col("Num_of_Loan") / credit_items).cast(FloatType()).alias("loans_per_credit_item"),
        # average delay normalised by number of loans; flags systemic payment strain
        F.when(col("Num_of_Loan") > 0, col("Delay_from_due_date") / col("Num_of_Loan")).cast(FloatType()).alias("loan_extent"),
        # credit enquiries per loan; signals credit-seeking behaviour
        F.when(col("Num_of_Loan") > 0, col("Num_Credit_Inquiries") / col("Num_of_Loan")).cast(FloatType()).alias("inq_per_loan"),
        # liquidity relative to debt
        F.when(col("Outstanding_Debt") > 0, col("Monthly_Balance") / col("Outstanding_Debt")).cast(FloatType()).alias("balance_to_debt"),
        # binary: any credit-limit change detected (financial volatility signal)
        F.coalesce(col("Changed_Credit_Limit") != 0, F.lit(False)).cast(IntegerType()).alias("has_credit_limit_change"),

        # --- age bins (one-hot, raw age retained in NUMERIC_FEATURES above) ---
        *_age_bin_cols(age),

        # --- loan type counts ---
        *[loan_type_count(t).alias(loan_type_feature(t)) for t in LOAN_TYPES],

        # --- occupation one-hots ---
        *[occupation_is(o).alias(occupation_feature(o)) for o in OCCUPATIONS],
        col("Occupation").isNull().cast(IntegerType()).alias("occupation_missing"),

        # --- clickstream (mean across all months up to snapshot_date; float) ---
        *CLICKSTREAM_COLUMNS,
        "has_clickstream",
    )

    validate_and_write(df, "gold", "feature_store", snapshot_date, KEYS)
    return df
