import os
import sys

import pyspark


def get_spark(app_name="dev"):
    # Windows has no 'python3'; point Spark at the current interpreter
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    spark = pyspark.sql.SparkSession.builder.appName(app_name).master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark
