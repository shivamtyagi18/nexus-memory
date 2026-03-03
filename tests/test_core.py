"""Tests for nexus.core — end-to-end encode/recall, context manager, metrics."""

import os
import pytest
import threading

from nexus.models import NexusConfig, MemorySource
from nexus import NEXUS, NexusMetrics


@pytest.fixture
def nexus(tmp_dir, mock_llm):
    """NEXUS instance with mock LLM for testing (no real API calls)."""
    config = NexusConfig(storage_path=os.path.join(tmp_dir, "nexus_db"))
    n = NEXUS(config=config)
    # Replace LLM with mock to avoid real API calls
    n.llm = mock_llm
    n.attention_gate.llm = mock_llm
    n.consolidation_engine.llm = mock_llm
    yield n
    n.close()


class TestEncode:
    def test_encode_returns_id(self, nexus):
        mid = nexus.encode("Python is a programming language", use_llm=True)
        assert mid is not None

    def test_encode_empty_rejected(self, nexus):
        mid = nexus.encode("")
        assert mid is None

    def test_encode_whitespace_rejected(self, nexus):
        mid = nexus.encode("   ")
        assert mid is None

    def test_encode_truncates_long_content(self, nexus):
        long_content = "x" * 200000
        mid = nexus.encode(long_content, use_llm=False)
        # Should not crash, content should be truncated
        if mid:
            mem = nexus.palace.get_memory(mid)
            assert len(mem.content) <= nexus.config.max_content_length


class TestRecall:
    def test_recall_after_encode(self, nexus):
        nexus.encode("cats are furry domesticated animals", use_llm=True)
        results = nexus.recall("what are cats?")
        assert len(results) > 0

    def test_recall_empty_returns_nothing(self, nexus):
        results = nexus.recall("nonexistent topic")
        assert results == []


class TestMetrics:
    def test_encode_tracked(self, nexus):
        nexus.encode("trackable memory", use_llm=True)
        metrics = nexus.get_metrics()
        assert metrics["operations"]["encode"]["total"] >= 1

    def test_recall_tracked(self, nexus):
        nexus.encode("test", use_llm=True)
        nexus.recall("test")
        metrics = nexus.get_metrics()
        assert metrics["operations"]["recall"]["total"] >= 1

    def test_prometheus_format(self, nexus):
        nexus.encode("test", use_llm=True)
        text = nexus.get_metrics_prometheus()
        assert "nexus_encode_total" in text


class TestContextManager:
    def test_context_manager(self, tmp_dir, mock_llm):
        config = NexusConfig(storage_path=os.path.join(tmp_dir, "cm_test"))
        with NEXUS(config=config) as n:
            n.llm = mock_llm
            n.attention_gate.llm = mock_llm
            n.encode("inside context manager", use_llm=True)
        # Should not crash after exit


class TestClose:
    def test_close_saves_state(self, nexus):
        nexus.encode("save me", use_llm=True)
        nexus.close()
        # Should not crash on double close
        nexus.close()


class TestConcurrency:
    def test_concurrent_encode(self, nexus):
        errors = []

        def encode(i):
            try:
                nexus.encode(f"concurrent fact number {i}", use_llm=True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=encode, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
