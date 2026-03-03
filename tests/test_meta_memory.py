"""Tests for nexus.meta_memory — confidence mapping, gap tracking."""

import pytest
from nexus.models import ConfidenceLevel, DecisionType
from nexus.meta_memory import MetaMemory, _topic_overlap


class TestConfidenceMap:
    def test_unknown_for_empty_palace(self, palace):
        mm = MetaMemory(palace=palace)
        conf = mm.confidence_map("anything")
        assert conf.is_unknown

    def test_known_after_placement(self, palace, make_memory):
        mm = MetaMemory(palace=palace)
        palace.place_memory(make_memory("Python is great for data science"))
        conf = mm.confidence_map("Python")
        # Should have some confidence now
        assert conf.coverage > 0


class TestShouldRecallOrAsk:
    def test_ask_when_unknown(self, palace):
        mm = MetaMemory(palace=palace)
        decision = mm.should_recall_or_ask("quantum physics")
        assert decision == DecisionType.ADMIT_GAP_AND_ASK


class TestGapTracking:
    def test_register_gap(self, palace):
        mm = MetaMemory(palace=palace)
        mm.register_gap("quantum physics", "user asked about it")
        gaps = mm.knowledge_gaps()
        assert len(gaps) == 1
        assert gaps[0]["topic"] == "quantum physics"
        assert gaps[0]["resolved"] is False

    def test_resolve_gap(self, palace):
        mm = MetaMemory(palace=palace)
        mm.register_gap("quantum physics")
        mm.resolve_gap("quantum physics")
        gaps = mm.knowledge_gaps()
        assert gaps[0]["resolved"] is True

    def test_gap_from_failed_retrievals(self, palace):
        mm = MetaMemory(palace=palace)
        mm.register_failed_retrieval("quantum entanglement physics")
        mm.register_failed_retrieval("quantum entanglement explained")
        mm.register_failed_retrieval("quantum entanglement definition")
        # After 3 similar failures (overlapping words) → auto-registers gap
        gaps = [g for g in mm.knowledge_gaps() if not g["resolved"]]
        assert len(gaps) >= 1


class TestTopicOverlap:
    def test_identical(self):
        assert _topic_overlap("hello world", "hello world") is True

    def test_partial_overlap(self):
        assert _topic_overlap("machine learning algorithms", "learning algorithms") is True

    def test_no_overlap(self):
        assert _topic_overlap("cats and dogs", "quantum physics") is False

    def test_empty_strings(self):
        assert _topic_overlap("", "") is False


class TestConfidenceSummary:
    def test_summary_empty(self, palace):
        mm = MetaMemory(palace=palace)
        s = mm.get_confidence_summary()
        assert "No knowledge" in s

    def test_summary_with_data(self, palace, make_memory):
        mm = MetaMemory(palace=palace)
        palace.place_memory(make_memory("Python facts"))
        s = mm.get_confidence_summary()
        assert "confidence map" in s.lower() or "Knowledge" in s


class TestStats:
    def test_stats(self, palace):
        mm = MetaMemory(palace=palace)
        mm.register_gap("topic1")
        s = mm.stats()
        assert s["active_gaps"] == 1
        assert s["total_rooms"] == 0
