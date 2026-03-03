"""Tests for nexus.consolidation — scheduling, forgetting, conflict resolution."""

import pytest
from collections import deque
from datetime import datetime, timedelta

from nexus.models import (
    NexusConfig, Memory, MemorySource, MemoryStatus,
    ConsolidationDepth, SalienceScore,
)
from nexus.consolidation import ConsolidationEngine


class TestScheduler:
    def test_defer_when_empty(self, episode_buffer, palace, vector_store, mock_llm):
        config = NexusConfig()
        engine = ConsolidationEngine(
            episode_buffer=episode_buffer, palace=palace,
            vector_store=vector_store, llm=mock_llm, config=config,
        )
        assert engine.should_consolidate() == ConsolidationDepth.DEFER

    def test_light_on_buffer_threshold(self, episode_buffer, palace, vector_store, mock_llm, make_episode):
        config = NexusConfig(episode_buffer_trigger=5)
        engine = ConsolidationEngine(
            episode_buffer=episode_buffer, palace=palace,
            vector_store=vector_store, llm=mock_llm, config=config,
        )
        for i in range(5):
            episode_buffer.add(make_episode(f"episode {i}"))
        assert engine.should_consolidate() == ConsolidationDepth.LIGHT

    def test_full_on_large_backlog(self, episode_buffer, palace, vector_store, mock_llm, make_episode):
        config = NexusConfig(episode_buffer_trigger=5)
        engine = ConsolidationEngine(
            episode_buffer=episode_buffer, palace=palace,
            vector_store=vector_store, llm=mock_llm, config=config,
        )
        for i in range(20):  # 4x trigger
            episode_buffer.add(make_episode(f"episode {i}"))
        assert engine.should_consolidate() == ConsolidationDepth.FULL


class TestForgetting:
    def test_user_stated_never_forgotten(self, episode_buffer, palace, vector_store, mock_llm):
        config = NexusConfig()
        engine = ConsolidationEngine(
            episode_buffer=episode_buffer, palace=palace,
            vector_store=vector_store, llm=mock_llm, config=config,
        )

        m = Memory(
            content="user said this",
            source=MemorySource.USER_STATED,
        )
        palace.place_memory(m)

        result = engine._process_forgetting()
        # User-stated memories should never be forgotten
        assert m.status == MemoryStatus.ACTIVE

    def test_pinned_never_forgotten(self, episode_buffer, palace, vector_store, mock_llm):
        config = NexusConfig()
        engine = ConsolidationEngine(
            episode_buffer=episode_buffer, palace=palace,
            vector_store=vector_store, llm=mock_llm, config=config,
        )

        m = Memory(content="pinned memory")
        m.status = MemoryStatus.PINNED
        palace.place_memory(m)

        result = engine._process_forgetting()
        assert m.status == MemoryStatus.PINNED


class TestConflictResolution:
    def test_confidence_capped(self, episode_buffer, palace, vector_store, mock_llm):
        config = NexusConfig()
        engine = ConsolidationEngine(
            episode_buffer=episode_buffer, palace=palace,
            vector_store=vector_store, llm=mock_llm, config=config,
        )

        winner = Memory(content="winner", confidence=0.95)
        loser = Memory(content="loser")
        engine._resolve_conflict(winner, loser, {"strategy": "temporal"})
        assert winner.confidence <= 1.0
        assert loser.status == MemoryStatus.SUPERSEDED


class TestTombstones:
    def test_tombstones_bounded(self, episode_buffer, palace, vector_store, mock_llm):
        config = NexusConfig()
        engine = ConsolidationEngine(
            episode_buffer=episode_buffer, palace=palace,
            vector_store=vector_store, llm=mock_llm, config=config,
        )
        assert isinstance(engine.tombstones, deque)
        assert engine.tombstones.maxlen == 500


class TestConsolidationLog:
    def test_log_bounded(self, episode_buffer, palace, vector_store, mock_llm):
        config = NexusConfig()
        engine = ConsolidationEngine(
            episode_buffer=episode_buffer, palace=palace,
            vector_store=vector_store, llm=mock_llm, config=config,
        )
        assert isinstance(engine.consolidation_log, deque)
        assert engine.consolidation_log.maxlen == 100


class TestStats:
    def test_stats(self, episode_buffer, palace, vector_store, mock_llm):
        config = NexusConfig()
        engine = ConsolidationEngine(
            episode_buffer=episode_buffer, palace=palace,
            vector_store=vector_store, llm=mock_llm, config=config,
        )
        s = engine.stats()
        assert "last_consolidation" in s
        assert "tombstones" in s
        assert "unconsolidated_episodes" in s
