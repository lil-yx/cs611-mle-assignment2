"""Data contracts: every silver and gold table is checked before it is written.

Expected schemas live in config/schemas/<layer>/<table>.yaml. Bronze has no contract.
"""

import os

import yaml

from utils.common.config import SCHEMA_DIR
from utils.common.io import write_parquet


def load_expected_schema(layer, table):
    with open(os.path.join(SCHEMA_DIR, layer, f"{table}.yaml")) as f:
        return yaml.safe_load(f)["columns"]


def validate_schema(df, layer, table):
    expected = load_expected_schema(layer, table)
    actual = {field.name: field.dataType.simpleString() for field in df.schema.fields}
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        wrong_type = sorted(c for c in set(expected) & set(actual) if expected[c] != actual[c])
        raise ValueError(
            f"{layer}.{table} schema mismatch - missing: {missing}, unexpected: {unexpected}, wrong type: {wrong_type}"
        )


def validate_unique_key(df, key_columns, table):
    duplicates = df.groupBy(*key_columns).count().filter("count > 1").count()
    if duplicates:
        raise ValueError(f"{table}: {duplicates} duplicate keys on {key_columns}")


def validate_and_write(df, layer, table, snapshot_date, key_columns):
    validate_schema(df, layer, table)
    validate_unique_key(df, key_columns, f"{layer}.{table}")
    write_parquet(df, layer, table, snapshot_date)
