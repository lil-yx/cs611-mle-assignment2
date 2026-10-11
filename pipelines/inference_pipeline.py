"""Inference pipeline: score all customers for a given snapshot month.

Loads the champion model (or a named model), reads the feature store partition
for the given snapshot_date, generates a default-probability score for every
customer, and writes the results as a gold predictions table.

Usage (from project root, inside Docker):
    python -m pipelines.inference_pipeline --snapshotdate 2024-01-01
    python -m pipelines.inference_pipeline --snapshotdate 2024-01-01 --modelid <model_id>
"""

import argparse

import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType, IntegerType, StringType

from utils.common.io import parquet_exists, read_parquet
from utils.common.log import get_logger
from utils.common.spark_session import get_spark
from utils.ml.model_store import load_model
from utils.transformation.validation import validate_and_write

logger = get_logger("inference_pipeline")

# customers with score >= THRESHOLD are flagged as predicted defaults
THRESHOLD = 0.5


def run(snapshot_date, model_id=None):
    if not parquet_exists("gold", "feature_store", snapshot_date):
        logger.warning("no feature store partition for %s — skipping", snapshot_date)
        return

    model, feature_cols, metadata = load_model(model_id)
    used_model_id = metadata["model_id"]
    logger.info("model: %s", used_model_id)

    spark = get_spark("inference_pipeline")

    # load feature partition and select only the columns the model expects (in order)
    features_df = read_parquet(spark, "gold", "feature_store", snapshot_date)
    missing = [c for c in feature_cols if c not in features_df.columns]
    if missing:
        raise ValueError(f"feature store is missing columns used in training: {missing}")

    pdf = features_df.select("Customer_ID", "snapshot_date", *feature_cols).toPandas()
    logger.info("scoring %d customers for %s …", len(pdf), snapshot_date)

    X = pdf[feature_cols].values
    scores = model.predict_proba(X)[:, 1].astype(float)
    predictions = (scores >= THRESHOLD).astype(int)

    pdf["score"]      = scores
    pdf["prediction"] = predictions
    pdf["model_id"]   = used_model_id

    result_df = (
        spark.createDataFrame(pdf[["Customer_ID", "snapshot_date", "score", "prediction", "model_id"]])
        .withColumn("snapshot_date", col("snapshot_date").cast("date"))
        .withColumn("score",         col("score").cast(FloatType()))
        .withColumn("prediction",    col("prediction").cast(IntegerType()))
        .withColumn("model_id",      col("model_id").cast(StringType()))
        .withColumn("scored_at",     F.current_timestamp())
    )

    validate_and_write(result_df, "gold", "predictions", snapshot_date, ["Customer_ID", "snapshot_date"])
    logger.info(
        "predictions written — bad rate: %.1f%%  (threshold=%.2f)",
        predictions.mean() * 100, THRESHOLD,
    )
    spark.stop()


def main():
    parser = argparse.ArgumentParser(description="score one snapshot month")
    parser.add_argument("--snapshotdate", required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelid", default=None, help="model_id to use (default: champion)")
    args = parser.parse_args()
    run(args.snapshotdate, args.modelid)


if __name__ == "__main__":
    main()
