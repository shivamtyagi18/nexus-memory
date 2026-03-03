"""Tests for nexus.metrics — counters, gauges, histograms, export."""

import pytest
from nexus.metrics import NexusMetrics


class TestCounter:
    def test_increment(self, metrics):
        metrics.encode_count.inc()
        metrics.encode_count.inc()
        assert metrics.encode_count.value == 2

    def test_increment_by_amount(self, metrics):
        metrics.encode_count.inc(5)
        assert metrics.encode_count.value == 5


class TestGauge:
    def test_set_value(self, metrics):
        metrics.memory_count.set(42)
        assert metrics.memory_count.value == 42

    def test_set_overwrites(self, metrics):
        metrics.memory_count.set(10)
        metrics.memory_count.set(20)
        assert metrics.memory_count.value == 20


class TestHistogram:
    def test_observe(self, metrics):
        metrics.encode_latency.observe(10.0)
        metrics.encode_latency.observe(20.0)
        metrics.encode_latency.observe(30.0)
        snap = metrics.encode_latency.snapshot()
        assert snap["count"] == 3
        assert snap["sum"] == pytest.approx(60.0)
        assert snap["min"] == pytest.approx(10.0)
        assert snap["max"] == pytest.approx(30.0)

    def test_percentiles(self, metrics):
        for i in range(100):
            metrics.encode_latency.observe(float(i))
        snap = metrics.encode_latency.snapshot()
        assert snap["p50"] == pytest.approx(49.5, abs=1)
        assert snap["p95"] >= 90
        assert snap["p99"] >= 95


class TestSnapshot:
    def test_snapshot_structure(self, metrics):
        metrics.encode_count.inc()
        metrics.memory_count.set(5)
        metrics.encode_latency.observe(100.0)

        snap = metrics.snapshot()
        assert "operations" in snap
        assert "state" in snap
        assert snap["operations"]["encode"]["total"] == 1
        assert snap["state"]["memories"] == 5

    def test_empty_snapshot(self, metrics):
        snap = metrics.snapshot()
        assert snap["operations"]["encode"]["total"] == 0
        assert snap["state"]["memories"] == 0


class TestPrometheusExport:
    def test_prometheus_format(self, metrics):
        metrics.encode_count.inc(3)
        metrics.memory_count.set(10)
        metrics.encode_latency.observe(50.0)

        text = metrics.prometheus()
        assert "nexus_encode_total 3" in text
        assert "nexus_memories 10" in text
        assert "nexus_encode_latency_ms_count 1" in text

    def test_empty_prometheus(self, metrics):
        text = metrics.prometheus()
        assert "nexus_encode_total 0" in text
