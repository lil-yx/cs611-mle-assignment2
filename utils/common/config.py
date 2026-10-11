import os

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open(os.path.join(PROJECT_ROOT, "config", "base.yaml")) as f:
    CONFIG = yaml.safe_load(f)

SOURCES = CONFIG["sources"]
DATAMART_DIR = CONFIG["paths"]["datamart_dir"]
SCHEMA_DIR = os.path.join(PROJECT_ROOT, CONFIG["paths"]["schema_dir"])


def table_dir(layer, table):
    return f"{DATAMART_DIR}{layer}/{table}/"


def table_path(layer, table, snapshot_date):
    """datamart/<layer>/<table>/<layer>_<table>_YYYY_MM_DD.<csv|parquet>"""
    extension = "csv" if layer == "bronze" else "parquet"
    return f"{table_dir(layer, table)}{layer}_{table}_{snapshot_date.replace('-', '_')}.{extension}"
