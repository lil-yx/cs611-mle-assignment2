"""Gold job for one month: python -m pipelines.feature_pipeline --snapshotdate 2024-01-01 [--table label_store]"""

import argparse

from utils.common.config import CONFIG
from utils.common.spark_session import get_spark
from utils.features.build_gold import process_labels_gold_table, process_features_gold_table


def main(snapshot_date, tables, dpd, mob):
    spark = get_spark()
    if "label_store" in tables:
        process_labels_gold_table(snapshot_date, spark, dpd=dpd, mob=mob)
    if "feature_store" in tables:
        process_features_gold_table(snapshot_date, spark)
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gold tables for one snapshot month")
    parser.add_argument("--snapshotdate", required=True, help="YYYY-MM-DD")
    parser.add_argument("--table", choices=["label_store", "feature_store"], help="default: both")
    parser.add_argument("--dpd", type=int, default=CONFIG["label"]["dpd"], help="days past due that counts as default")
    parser.add_argument("--mob", type=int, default=CONFIG["label"]["mob"], help="month on book when the label is taken")
    args = parser.parse_args()
    main(args.snapshotdate, [args.table] if args.table else ["label_store", "feature_store"], args.dpd, args.mob)
