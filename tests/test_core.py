"""Tests for smriti.core — end-to-end encode/recall, context manager, metrics."""

import os
import pytest
import threading

from smriti_memcore.models import SmritiConfig, MemorySource
from smriti_memcore import SMRITI, SmritiMetrics


@pytest.fixture
def smriti(tmp_dir, mock_llm):
    """SMRITI instance with mock LLM for testing (no real API calls)."""
    config = SmritiConfig(storage_path=os.path.join(tmp_dir, "smriti_db"))
    n = SMRITI(config=config)
    # Replace LLM with mock to avoid real API calls
    n.llm = mock_llm
    n.attention_gate.llm = mock_llm
    n.consolidation_engine.llm = mock_llm
    yield n
    n.close()


class TestEncode:
    def test_encode_returns_id(self, smriti):
        mid = smriti.encode("Python is a programming language", use_llm=True)
        assert mid is not None

    def test_encode_empty_rejected(self, smriti):
        mid = smriti.encode("")
        assert mid is None

    def test_encode_whitespace_rejected(self, smriti):
        mid = smriti.encode("   ")
        assert mid is None

    def test_encode_truncates_long_content(self, smriti):
        long_content = "x" * 200000
        mid = smriti.encode(long_content, use_llm=False)
        # Should not crash, content should be truncated
        if mid:
            mem = smriti.palace.get_memory(mid)
            assert len(mem.content) <= smriti.config.max_content_length


class TestRecall:
    def test_recall_after_encode(self, smriti):
        smriti.encode("cats are furry domesticated animals", use_llm=True)
        results = smriti.recall("what are cats?")
        assert len(results) > 0

    def test_recall_empty_returns_nothing(self, smriti):
        results = smriti.recall("nonexistent topic")
        assert results == []


class TestMetrics:
    def test_encode_tracked(self, smriti):
        smriti.encode("trackable memory", use_llm=True)
        metrics = smriti.get_metrics()
        assert metrics["operations"]["encode"]["total"] >= 1

    def test_recall_tracked(self, smriti):
        smriti.encode("test", use_llm=True)
        smriti.recall("test")
        metrics = smriti.get_metrics()
        assert metrics["operations"]["recall"]["total"] >= 1

    def test_prometheus_format(self, smriti):
        smriti.encode("test", use_llm=True)
        text = smriti.get_metrics_prometheus()
        assert "smriti_encode_total" in text


class TestContextManager:
    def test_context_manager(self, tmp_dir, mock_llm):
        config = SmritiConfig(storage_path=os.path.join(tmp_dir, "cm_test"))
        with SMRITI(config=config) as n:
            n.llm = mock_llm
            n.attention_gate.llm = mock_llm
            n.encode("inside context manager", use_llm=True)
        # Should not crash after exit


class TestClose:
    def test_close_saves_state(self, smriti):
        smriti.encode("save me", use_llm=True)
        smriti.close()
        # Should not crash on double close
        smriti.close()


class TestConcurrency:
    def test_concurrent_encode(self, smriti):
        errors = []

        def encode(i):
            try:
                smriti.encode(f"concurrent fact number {i}", use_llm=True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=encode, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
