"""Unit tests for the incremental, gated layer builds in main.py (no Spark started)."""

import pytest

import main


def test_expected_partitions_cover_every_source_month():
    bronze = main.expected_partitions("bronze")
    assert bronze == main.expected_partitions("silver")
    assert ("lms_loan_daily", "2023-01-01") in bronze
    assert len(bronze) == sum(len(main.source_months(s)) for s in main.SOURCES)


def test_expected_gold_has_labels_and_features():
    gold = main.expected_partitions("gold")
    tables = {t for t, _ in gold}
    assert tables == {"label_store", "feature_store"}


def test_unknown_layer_rejected():
    with pytest.raises(ValueError):
        main.expected_partitions("platinum")


def test_build_layer_skips_when_complete(monkeypatch):
    monkeypatch.setattr(main, "parquet_exists", lambda *a: True)
    monkeypatch.setattr(main, "get_spark", lambda *a: pytest.fail("Spark must not start"))
    assert main.build_layer("gold") == 0


def test_build_layer_builds_only_missing_then_gates(monkeypatch):
    built = set()
    gap = {("silver", "attributes", "2023-03-01"), ("silver", "attributes", "2023-04-01")}
    monkeypatch.setattr(main, "parquet_exists", lambda l, t, m: (l, t, m) not in gap or (t, m) in built)

    class FakeSpark:
        def stop(self):
            pass

    monkeypatch.setattr(main, "get_spark", lambda *a: FakeSpark())
    monkeypatch.setattr(main, "_process", lambda layer, t, m, spark: built.add((t, m)))
    assert main.build_layer("silver") == 2
    assert built == {("attributes", "2023-03-01"), ("attributes", "2023-04-01")}


def test_build_layer_gate_fails_when_partition_not_written(monkeypatch):
    monkeypatch.setattr(main, "parquet_exists", lambda l, t, m: m != "2023-02-01")

    class FakeSpark:
        def stop(self):
            pass

    monkeypatch.setattr(main, "get_spark", lambda *a: FakeSpark())
    monkeypatch.setattr(main, "_process", lambda *a: None)   # "succeeds" but writes nothing
    with pytest.raises(RuntimeError, match="bronze gate failed"):
        main.build_layer("bronze")
