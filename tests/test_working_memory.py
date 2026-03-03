"""Tests for nexus.working_memory — capacity, dedup, eviction, priority."""

import pytest
from nexus.models import Memory, SalienceScore, MemorySource
from nexus.working_memory import WorkingMemory, EvictionRecord


class TestAdmit:
    def test_admit_single(self, working_memory, make_memory):
        m = make_memory("important fact")
        working_memory.admit(m)
        assert working_memory.size == 1
        assert working_memory.contains(m.id)

    def test_admit_deduplicates(self, working_memory, make_memory):
        m = make_memory("same memory")
        working_memory.admit(m)
        working_memory.admit(m)  # Same memory again
        assert working_memory.size == 1

    def test_admit_evicts_at_capacity(self, make_memory):
        wm = WorkingMemory(max_slots=3)
        memories = [make_memory(f"mem {i}") for i in range(4)]
        for m in memories[:3]:
            wm.admit(m)
        assert wm.size == 3
        assert wm.is_full

        eviction = wm.admit(memories[3])
        assert wm.size == 3  # Still at capacity
        assert eviction is not None
        assert isinstance(eviction, EvictionRecord)
        assert eviction.reason == "capacity"

    def test_eviction_logged(self, make_memory):
        wm = WorkingMemory(max_slots=2)
        m1 = make_memory("first")
        m2 = make_memory("second")
        m3 = make_memory("third")
        wm.admit(m1)
        wm.admit(m2)
        wm.admit(m3)
        assert len(wm.eviction_log) == 1


class TestPriority:
    def test_priority_ordering(self, make_memory):
        wm = WorkingMemory(max_slots=5)
        low = make_memory("low", salience=SalienceScore(relevance=0.1, utility=0.1))
        low.strength = 0.1
        low.confidence = 0.1
        high = make_memory("high", salience=SalienceScore(relevance=0.9, utility=0.9))
        high.strength = 3.0
        high.confidence = 1.0

        wm.admit(low)
        wm.admit(high)

        all_mems = wm.get_all()
        assert all_mems[0].id == high.id  # Higher priority first

    def test_update_priority(self, working_memory, make_memory):
        m = make_memory("test")
        working_memory.admit(m, priority=0.5)
        working_memory.update_priority(m.id, 0.9)
        p = working_memory._get_priority(m.id)
        assert p == pytest.approx(0.9)


class TestContext:
    def test_active_vs_peripheral(self, make_memory):
        wm = WorkingMemory(max_slots=7, active_chunks=4)
        for i in range(7):
            wm.admit(make_memory(f"mem {i}"))

        active = wm.get_active_context()
        peripheral = wm.get_peripheral_context()
        assert len(active) == 4
        assert len(peripheral) == 3

    def test_format_for_llm(self, working_memory, make_memory):
        working_memory.admit(make_memory("a fact"))
        text = working_memory.format_for_llm()
        assert "Active Context" in text
        assert "a fact" in text


class TestRemoveAndContains:
    def test_remove(self, working_memory, make_memory):
        m = make_memory("removable")
        working_memory.admit(m)
        working_memory.remove(m.id)
        assert not working_memory.contains(m.id)
        assert working_memory.size == 0


class TestSuggestions:
    def test_surface_suggestion(self, working_memory, make_memory):
        m = make_memory("suggestion")
        working_memory.surface_suggestion(m)
        assert len(working_memory.get_suggestions()) == 1

    def test_surface_warning(self, working_memory):
        working_memory.surface_warning("this failed before")
        assert len(working_memory.get_warnings()) == 1

    def test_clear_suggestions(self, working_memory, make_memory):
        working_memory.surface_suggestion(make_memory("s"))
        working_memory.surface_warning("w")
        working_memory.clear_suggestions()
        assert len(working_memory.get_suggestions()) == 0
        assert len(working_memory.get_warnings()) == 0

    def test_suggestions_bounded(self, working_memory, make_memory):
        for i in range(10):
            working_memory.surface_suggestion(make_memory(f"s{i}"))
        assert len(working_memory.get_suggestions()) <= 3


class TestStats:
    def test_stats(self, working_memory, make_memory):
        working_memory.admit(make_memory("test"))
        s = working_memory.stats()
        assert s["slots_used"] == 1
        assert s["max_slots"] == 7
