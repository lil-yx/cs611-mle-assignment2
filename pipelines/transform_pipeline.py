"""Silver job for one month: python -m pipelines.transform_pipeline --snapshotdate 2024-01-01 [--source attributes]"""

import argparse

from utils.common.config import SOURCES
from utils.common.spark_session import get_spark
from utils.transformation.build_silver import process_silver_table


def main(snapshot_date, sources):
    spark = get_spark()
    for source in sources:
        process_silver_table(source, snapshot_date, spark)
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="silver transformation for one snapshot month")
    parser.add_argument("--snapshotdate", required=True, help="YYYY-MM-DD")
    parser.add_argument("--source", choices=list(SOURCES), help="default: all sources")
    args = parser.parse_args()
    main(args.snapshotdate, [args.source] if args.source else list(SOURCES))
