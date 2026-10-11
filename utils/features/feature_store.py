"""Read API for the gold stores.

Labels are joined to features point-in-time, on the loan's application date, so a
label observed at MOB 6 only ever sees features known when the loan was approved.
"""

from utils.common.io import read_all_partitions


def load_label_store(spark):
    return read_all_partitions(spark, "gold", "label_store")


def load_feature_store(spark):
    return read_all_partitions(spark, "gold", "feature_store")


def get_training_data(spark):
    labels = load_label_store(spark)
    features = load_feature_store(spark).withColumnRenamed("snapshot_date", "feature_snapshot_date")
    return labels.join(
        features,
        (labels.Customer_ID == features.Customer_ID) & (labels.loan_start_date == features.feature_snapshot_date),
        "inner",
    ).drop(features.Customer_ID)
