"""Medallion datamart: bronze -> silver -> gold.

    python main.py                      full rebuild of every partition (overwrites)

Airflow runs one task per layer through build_layer(), which processes only the
partitions that are missing, using the same per-month functions as the single-month
jobs in pipelines/ (ingest_pipeline, transform_pipeline, feature_pipeline):
    - first run on a fresh checkout  -> backfills all history from data/
    - later runs                     -> skip in about a second
    - a new month of data            -> only that month is built
Each layer ends with a gate: if any expected partition is still missing the task
fails, so nothing downstream runs on an incomplete datamart.
"""

from utils.common.config import CONFIG, SOURCES
from utils.common.dates import month_starts
from utils.common.io import parquet_exists
from utils.common.log import get_logger
from utils.common.spark_session import get_spark
from utils.ingestion.load_bronze import process_bronze_table
from utils.transformation.build_silver import process_silver_table
from utils.features.build_gold import process_labels_gold_table, process_features_gold_table
from utils.features.feature_store import load_label_store, load_feature_store, get_training_data

logger = get_logger("main")

LAYERS = ("bronze", "silver", "gold")


def source_months(source):
    return month_starts(SOURCES[source]["start_date"], SOURCES[source]["end_date"])


def _feature_months():
    window = CONFIG["feature_store"]
    return month_starts(window["start_date"], window["end_date"])


def expected_partitions(layer):
    """(table, month) pairs the config says this layer must contain."""
    if layer in ("bronze", "silver"):
        return [(source, m) for source in SOURCES for m in source_months(source)]
    if layer == "gold":
        return ([("label_store", m) for m in source_months("lms_loan_daily")]
                + [("feature_store", m) for m in _feature_months()])
    raise ValueError(f"unknown layer: {layer}")


def missing_partitions(layer):
    return [(t, m) for t, m in expected_partitions(layer) if not parquet_exists(layer, t, m)]


def _process(layer, table, month, spark):
    if layer == "bronze":
        process_bronze_table(table, month, spark)
    elif layer == "silver":
        process_silver_table(table, month, spark)
    elif table == "label_store":
        label = CONFIG["label"]
        process_labels_gold_table(month, spark, dpd=label["dpd"], mob=label["mob"])
    else:
        process_features_gold_table(month, spark)


def build_layer(layer):
    """Airflow entry point: build the missing partitions of one layer, then gate it."""
    missing = missing_partitions(layer)
    if not missing:
        logger.info("%s complete (%d partitions) — nothing to build", layer, len(expected_partitions(layer)))
        return 0

    logger.info("%s: %d partitions missing — building", layer, len(missing))
    spark = get_spark(f"build_{layer}")
    try:
        for table, month in missing:      # months ascending within each table
            _process(layer, table, month, spark)
    finally:
        spark.stop()

    still_missing = missing_partitions(layer)
    if still_missing:
        raise RuntimeError(f"{layer} gate failed: {len(still_missing)} partitions still missing, "
                           f"e.g. {still_missing[:5]}")
    logger.info("%s gate passed: all %d partitions present", layer, len(expected_partitions(layer)))
    return len(missing)


def main():
    """Full rebuild of every partition (used by `make pipeline`)."""
    spark = get_spark()
    for layer in LAYERS:
        for table, month in expected_partitions(layer):
            _process(layer, table, month, spark)

    features = load_feature_store(spark)
    logger.info("label_store: %s rows", load_label_store(spark).count())
    logger.info("feature_store: %s rows, %s columns", features.count(), len(features.columns))
    # every label should match exactly one point-in-time feature row
    logger.info("labels joined to features: %s", get_training_data(spark).count())

    spark.stop()


if __name__ == "__main__":
    main()
