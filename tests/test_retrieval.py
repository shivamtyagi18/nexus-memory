"""Tests for nexus.retrieval — multi-hop search, spreading activation."""

import pytest
from nexus.models import NexusConfig, Memory, SalienceScore
from nexus.retrieval import RetrievalEngine


class TestBasicRetrieval:
    def test_retrieve_finds_memories(self, palace, vector_store, working_memory, make_memory):
        config = NexusConfig()
        engine = RetrievalEngine(
            palace=palace, working_memory=working_memory,
            vector_store=vector_store, config=config,
        )

        m = make_memory("Python is a programming language")
        palace.place_memory(m)

        results = engine.retrieve("what is Python?", top_k=5)
        assert len(results) > 0
        assert any("Python" in r.content for r in results)

    def test_retrieve_empty_palace(self, palace, vector_store, working_memory):
        config = NexusConfig()
        engine = RetrievalEngine(
            palace=palace, working_memory=working_memory,
            vector_store=vector_store, config=config,
        )
        results = engine.retrieve("nothing here", top_k=5)
        assert results == []


class TestRetrievalStats:
    def test_stats(self, palace, vector_store, working_memory):
        config = NexusConfig()
        engine = RetrievalEngine(
            palace=palace, working_memory=working_memory,
            vector_store=vector_store, config=config,
        )
        s = engine.stats()
        assert "total_retrievals" in s


class TestMultipleMemories:
    def test_top_k_respected(self, palace, vector_store, working_memory, make_memory):
        config = NexusConfig()
        engine = RetrievalEngine(
            palace=palace, working_memory=working_memory,
            vector_store=vector_store, config=config,
        )

        for i in range(10):
            palace.place_memory(make_memory(f"memory about topic {i}"))

        results = engine.retrieve("topic", top_k=3)
        assert len(results) <= 3
