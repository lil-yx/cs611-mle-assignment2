import pytest

from utils.common.spark_session import get_spark


@pytest.fixture(scope="session")
def spark():
    spark = get_spark("tests")
    yield spark
    spark.stop()
