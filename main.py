"""Backfill the full medallion pipeline into datamart/: bronze -> silver -> gold."""

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


def source_months(source):
    return month_starts(SOURCES[source]["start_date"], SOURCES[source]["end_date"])


def main():
    spark = get_spark()

    for source in SOURCES:
        for month in source_months(source):
            process_bronze_table(source, month, spark)

    for source in SOURCES:
        for month in source_months(source):
            process_silver_table(source, month, spark)

    label = CONFIG["label"]
    for month in source_months("lms_loan_daily"):
        process_labels_gold_table(month, spark, dpd=label["dpd"], mob=label["mob"])

    feature_window = CONFIG["feature_store"]
    for month in month_starts(feature_window["start_date"], feature_window["end_date"]):
        process_features_gold_table(month, spark)

    features = load_feature_store(spark)
    logger.info("label_store: %s rows", load_label_store(spark).count())
    logger.info("feature_store: %s rows, %s columns", features.count(), len(features.columns))
    # every label should match exactly one point-in-time feature row
    logger.info("labels joined to features: %s", get_training_data(spark).count())

    spark.stop()


def missing_gold_partitions():
    """Gold partitions the config expects that are not on disk yet."""
    label_months = source_months("lms_loan_daily")
    window = CONFIG["feature_store"]
    feature_months = month_starts(window["start_date"], window["end_date"])
    missing = [("label_store", m) for m in label_months if not parquet_exists("gold", "label_store", m)]
    missing += [("feature_store", m) for m in feature_months if not parquet_exists("gold", "feature_store", m)]
    return missing


def build_if_missing():
    """Airflow entry point: build bronze -> silver -> gold only when gold is incomplete,
    so scheduled runs reuse the datamart and a fresh checkout rebuilds it from data/."""
    missing = missing_gold_partitions()
    if not missing:
        logger.info("gold label_store and feature_store complete — datamart build skipped")
        return
    logger.info("%d gold partitions missing (first: %s %s) — building datamart from data/",
                len(missing), *missing[0])
    main()
    still_missing = missing_gold_partitions()
    if still_missing:
        raise RuntimeError(f"datamart build finished but gold is incomplete: {still_missing[:5]}")


if __name__ == "__main__":
    main()
