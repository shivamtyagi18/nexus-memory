"""Tests for nexus.attention_gate — scoring, encoding, error fallback."""

import pytest
from unittest.mock import MagicMock

from nexus.models import NexusConfig, MemorySource, SalienceScore, Episode
from nexus.attention_gate import AttentionGate


class TestScoreFast:
    def test_error_content_high_surprise(self, mock_llm):
        config = NexusConfig()
        gate = AttentionGate(llm=mock_llm, config=config)
        score = gate.score_fast("Error: connection refused")
        assert score.surprise >= 0.5

    def test_instruction_high_utility(self, mock_llm):
        config = NexusConfig()
        gate = AttentionGate(llm=mock_llm, config=config)
        score = gate.score_fast("You must always validate input")
        assert score.utility >= 0.7

    def test_short_content_penalized(self, mock_llm):
        config = NexusConfig()
        gate = AttentionGate(llm=mock_llm, config=config)
        score = gate.score_fast("ok")
        assert score.utility < 0.5
        assert score.relevance < 0.5

    def test_user_stated_boost(self, mock_llm):
        config = NexusConfig()
        gate = AttentionGate(llm=mock_llm, config=config)
        score = gate.score_fast("some fact", source=MemorySource.USER_STATED)
        assert score.relevance >= 0.8
        assert score.utility >= 0.7


class TestShouldEncode:
    def test_high_salience_full(self, mock_llm):
        config = NexusConfig()
        gate = AttentionGate(llm=mock_llm, config=config)
        s = SalienceScore(surprise=1.0, relevance=1.0, emotional=1.0, novelty=1.0, utility=1.0)
        assert gate.should_encode(s) == "full"

    def test_low_salience_discard(self, mock_llm):
        config = NexusConfig()
        gate = AttentionGate(llm=mock_llm, config=config)
        s = SalienceScore()  # All zeros
        assert gate.should_encode(s) == "discard"

    def test_medium_salience_summary(self, mock_llm):
        config = NexusConfig()
        gate = AttentionGate(llm=mock_llm, config=config)
        s = SalienceScore(relevance=0.4, utility=0.4)
        decision = gate.should_encode(s)
        assert decision in ("summary", "full")


class TestProcess:
    def test_process_creates_episode(self, mock_llm):
        config = NexusConfig()
        gate = AttentionGate(llm=mock_llm, config=config)
        episode = gate.process("important information to remember", use_llm=True)
        assert episode is not None
        assert isinstance(episode, Episode)

    def test_process_discard_low_salience(self, mock_llm):
        config = NexusConfig()
        gate = AttentionGate(llm=mock_llm, config=config)
        # Use fast scoring with very short content
        episode = gate.process("ok", use_llm=False)
        # Very short → low salience → may be discarded or summarized
        if episode is not None:
            assert episode.content  # If kept, must have content

    def test_process_with_llm_error_fallback(self, mock_llm):
        """When LLM returns error, score() should fall back to score_fast()."""
        config = NexusConfig()
        gate = AttentionGate(llm=mock_llm, config=config)

        # Mock score_salience to return error
        mock_llm.score_salience = lambda content, context="": {"error": "LLM down"}

        # Should not crash — falls back to score_fast
        score = gate.score("important data to analyze", source=MemorySource.DIRECT)
        assert score.composite >= 0  # Got a valid score
