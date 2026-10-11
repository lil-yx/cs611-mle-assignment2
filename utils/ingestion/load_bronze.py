"""Raw -> Bronze: land one snapshot month of a source as-is, with no cleaning."""

from pyspark.sql.functions import col

from utils.common.config import SOURCES
from utils.common.io import write_bronze


def process_bronze_table(source, snapshot_date, spark):
    df = spark.read.csv(SOURCES[source]["csv_path"], header=True, inferSchema=False) \
        .filter(col("snapshot_date") == snapshot_date)
    write_bronze(df, source, snapshot_date)
    return df
