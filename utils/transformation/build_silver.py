"""Bronze -> Silver: enforce types, clean dirty values and derive loan status."""

import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from utils.common.io import read_bronze
from utils.transformation.cleaning import (
    clean_numeric, null_outside, null_if_invalid_category, duration_to_months, split_loan_types,
)
from utils.transformation.validation import validate_and_write

CUSTOMER_KEYS = ["Customer_ID", "snapshot_date"]

AGE_VALID_RANGE = (14, 100)
FINANCIALS_VALID_RANGES = {
    "Num_Bank_Accounts": (0, 20),
    "Num_Credit_Card": (0, 20),
    "Interest_Rate": (0, 50),
    "Num_of_Delayed_Payment": (0, 50),
    "Num_Credit_Inquiries": (0, 30),
    "Amount_invested_monthly": (0, 9999),  # 10000 is a sentinel
    "Monthly_Balance": (-1e6, 1e6),
}
FINANCIALS_NUMERIC = [
    "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card",
    "Interest_Rate", "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
    "Num_Credit_Inquiries", "Outstanding_Debt", "Credit_Utilization_Ratio",
    "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
]
FINANCIALS_COUNTS = [
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Delayed_Payment",
    "Num_Credit_Inquiries", "Delay_from_due_date",
]
LOAN_DAILY_TYPES = {
    "loan_id": StringType(),
    "Customer_ID": StringType(),
    "loan_start_date": DateType(),
    "tenure": IntegerType(),
    "installment_num": IntegerType(),
    "loan_amt": FloatType(),
    "due_amt": FloatType(),
    "paid_amt": FloatType(),
    "overdue_amt": FloatType(),
    "balance": FloatType(),
    "snapshot_date": DateType(),
}


def process_silver_loan_daily(snapshot_date, spark):
    df = read_bronze(spark, "lms_loan_daily", snapshot_date)
    df = df.select(*[col(c).cast(t).alias(c) for c, t in LOAN_DAILY_TYPES.items()])

    df = df.withColumn("mob", col("installment_num"))
    df = df.withColumn("installments_missed", F.coalesce(F.ceil(F.try_divide(col("overdue_amt"), col("due_amt"))).cast(IntegerType()), F.lit(0)))
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -col("installments_missed"))))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0))

    validate_and_write(df, "silver", "lms_loan_daily", snapshot_date, ["loan_id", "snapshot_date"])
    return df


def process_silver_attributes(snapshot_date, spark):
    df = read_bronze(spark, "attributes", snapshot_date)

    # Name and SSN (PII) are dropped here
    df = df.select(
        col("Customer_ID"),
        clean_numeric("Age").cast(IntegerType()).alias("Age"),
        null_if_invalid_category("Occupation").alias("Occupation"),
        col("snapshot_date").cast(DateType()),
    ).withColumn("Age", null_outside("Age", *AGE_VALID_RANGE))

    validate_and_write(df, "silver", "attributes", snapshot_date, CUSTOMER_KEYS)
    return df


def process_silver_financials(snapshot_date, spark):
    df = read_bronze(spark, "financials", snapshot_date)

    for c in FINANCIALS_NUMERIC:
        df = df.withColumn(c, clean_numeric(c))
    for c, (low, high) in FINANCIALS_VALID_RANGES.items():
        df = df.withColumn(c, null_outside(c, low, high))
    for c in FINANCIALS_COUNTS:
        df = df.withColumn(c, col(c).cast(IntegerType()))

    # cross-checks against take-home pay: income ~12x monthly salary; EMI <= monthly salary
    salary = col("Monthly_Inhand_Salary")
    df = df.withColumn("Annual_Income", F.when(col("Annual_Income") <= 36 * salary, col("Annual_Income")))
    df = df.withColumn("Total_EMI_per_month", F.when(col("Total_EMI_per_month") <= salary, col("Total_EMI_per_month")))

    df = df.withColumn("Credit_History_Age_Months", duration_to_months("Credit_History_Age"))

    # source Num_of_Loan is corrupt (-100, 1495); recount it from the loan-type list
    df = df.withColumn("Loan_Type_List", split_loan_types("Type_of_Loan"))
    df = df.withColumn("Num_of_Loan", F.size("Loan_Type_List"))

    for c in ["Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]:
        df = df.withColumn(c, null_if_invalid_category(c))
    df = df.withColumn("Payment_of_Min_Amount", F.when(col("Payment_of_Min_Amount") != "NM", col("Payment_of_Min_Amount")))  # NM = not mentioned

    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType())).drop("Type_of_Loan", "Credit_History_Age")

    validate_and_write(df, "silver", "financials", snapshot_date, CUSTOMER_KEYS)
    return df


def process_silver_clickstream(snapshot_date, spark):
    df = read_bronze(spark, "clickstream", snapshot_date)

    df = df.select(
        col("Customer_ID"),
        *[col(f"fe_{i}").cast(IntegerType()) for i in range(1, 21)],
        col("snapshot_date").cast(DateType()),
    ).dropDuplicates(CUSTOMER_KEYS)

    validate_and_write(df, "silver", "clickstream", snapshot_date, CUSTOMER_KEYS)
    return df


SILVER_PROCESSORS = {
    "lms_loan_daily": process_silver_loan_daily,
    "attributes": process_silver_attributes,
    "financials": process_silver_financials,
    "clickstream": process_silver_clickstream,
}


def process_silver_table(source, snapshot_date, spark):
    return SILVER_PROCESSORS[source](snapshot_date, spark)
