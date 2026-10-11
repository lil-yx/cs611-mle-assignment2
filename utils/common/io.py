import glob
import os

from utils.common.config import table_dir, table_path
from utils.common.log import get_logger

logger = get_logger(__name__)


def _ensure_parent_dir(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)


def write_bronze(df, table, snapshot_date):
    """Write raw rows as CSV, keeping every value exactly as received."""
    filepath = table_path("bronze", table, snapshot_date)
    _ensure_parent_dir(filepath)
    pdf = df.toPandas()
    pdf.to_csv(filepath, index=False)
    logger.info("%s rows -> %s", len(pdf), filepath)


def read_bronze(spark, table, snapshot_date):
    # all columns stay strings; silver owns the schema
    return spark.read.csv(table_path("bronze", table, snapshot_date), header=True, inferSchema=False)


def write_parquet(df, layer, table, snapshot_date):
    filepath = table_path(layer, table, snapshot_date)
    _ensure_parent_dir(filepath)
    df.write.mode("overwrite").parquet(filepath)
    # count from the written file (parquet metadata) instead of recomputing df
    logger.info("%s rows -> %s", df.sparkSession.read.parquet(filepath).count(), filepath)


def read_parquet(spark, layer, table, snapshot_date):
    return spark.read.parquet(table_path(layer, table, snapshot_date))


def parquet_exists(layer, table, snapshot_date):
    return os.path.exists(table_path(layer, table, snapshot_date))


def read_all_partitions(spark, layer, table):
    return spark.read.parquet(*sorted(glob.glob(table_dir(layer, table) + "*.parquet")))


def read_partitions_up_to(spark, layer, table, up_to_date):
    """Read all parquet partitions for a table where partition date <= up_to_date.

    Filename format: <layer>_<table>_YYYY_MM_DD.parquet — the last 10 chars before
    the extension encode the date, so lexicographic comparison equals date comparison.
    Returns None when no eligible partitions exist.
    """
    date_str = up_to_date.replace("-", "_")
    all_files = sorted(glob.glob(table_dir(layer, table) + "*.parquet"))
    eligible = [
        f for f in all_files
        if os.path.splitext(os.path.basename(f))[0][-10:] <= date_str
    ]
    return spark.read.parquet(*eligible) if eligible else None
