"""Column-cleaning rules for silver.

Rules are fixed domain limits, never statistics from the data, so they cannot leak
test information into training. Imputation happens later, in the model pipeline.
"""

import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType, IntegerType


def clean_numeric(column):
    """'52312.68_' -> 52312.68; non-numeric junk -> null."""
    stripped = F.regexp_replace(F.trim(col(column)), "_", "")
    return F.when(stripped.rlike(r"^-?\d+(\.\d+)?$"), stripped.cast(FloatType()))


def null_outside(column, low, high):
    return F.when(col(column).between(low, high), col(column))


def null_if_invalid_category(column):
    """Placeholders such as '_', '_______' or '!@9#%8' -> null."""
    return F.when(col(column).rlike(r"^[A-Za-z]"), F.trim(col(column)))


def duration_to_months(column):
    """'10 Years and 9 Months' -> 129."""
    years = F.regexp_extract(col(column), r"(\d+)\s+Years", 1)
    months = F.regexp_extract(col(column), r"(\d+)\s+Months", 1)
    return F.when(years != "", years.cast(IntegerType()) * 12 + months.cast(IntegerType()))


def split_loan_types(column):
    """'Auto Loan, and Payday Loan' -> ['Auto Loan', 'Payday Loan']; null -> []."""
    loan_list = F.split(F.regexp_replace(col(column), r"\band\s+", ""), r",\s*")
    return F.when(col(column).isNotNull(), loan_list).otherwise(F.array().cast("array<string>"))
