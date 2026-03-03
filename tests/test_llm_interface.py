"""Tests for nexus.llm_interface — error handling, retries, API key checks."""

import pytest
from nexus.llm_interface import LLMInterface, LLMResponse
from nexus.metrics import NexusMetrics


class TestLLMResponse:
    def test_response_fields(self):
        r = LLMResponse(text="hello", model="test")
        assert r.text == "hello"
        assert r.error is None

    def test_response_with_error(self):
        r = LLMResponse(text="", model="test", error="connection failed")
        assert r.error == "connection failed"


class TestAPIKeyMissing:
    def test_openai_missing_key(self):
        llm = LLMInterface(default_model="gpt-4")
        resp = llm._call_openai("test", "gpt-4", None, 0.3, 100)
        assert resp.error is not None
        assert "not configured" in resp.error

    def test_gemini_missing_key(self):
        llm = LLMInterface(default_model="gemini-flash")
        resp = llm._call_gemini("test", "gemini-flash", None, 0.3, 100)
        assert resp.error is not None
        assert "not configured" in resp.error

    def test_anthropic_missing_key(self):
        llm = LLMInterface(default_model="claude-3")
        resp = llm._call_anthropic("test", "claude-3", None, 0.3, 100)
        assert resp.error is not None
        assert "not configured" in resp.error


class TestGenerateJSON:
    def test_generate_json_with_error(self):
        """generate_json should return error dict when LLM fails."""
        llm = LLMInterface(default_model="gpt-4")
        # Will fail because no API key
        result = llm.generate_json("test prompt")
        assert "error" in result

    def test_generate_json_parses_valid(self):
        """generate_json should parse valid JSON from LLM response."""
        llm = LLMInterface(default_model="test")
        # Mock generate to return valid JSON
        llm.generate = lambda prompt, **kwargs: LLMResponse(
            text='{"key": "value"}', model="test"
        )
        result = llm.generate_json("test")
        assert result == {"key": "value"}

    def test_generate_json_extracts_from_markdown(self):
        """generate_json should extract JSON from markdown code blocks."""
        llm = LLMInterface(default_model="test")
        llm.generate = lambda prompt, **kwargs: LLMResponse(
            text='```json\n{"key": "value"}\n```', model="test"
        )
        result = llm.generate_json("test")
        assert result == {"key": "value"}


class TestMetricsIntegration:
    def test_metrics_tracked(self):
        metrics = NexusMetrics()
        llm = LLMInterface(default_model="gpt-4", metrics=metrics)
        # Call will fail (no API key) but should still track metrics
        llm.generate("test")
        assert metrics.llm_call_count.value >= 1
        assert metrics.llm_errors.value >= 1


class TestGenerateReflection:
    def test_reflection_with_error(self):
        """generate_reflection should return fallback on LLM error."""
        llm = LLMInterface(default_model="gpt-4")
        # Will fail because no API key → should return fallback
        result = llm.generate_reflection(["event1", "event2"], level=1)
        assert "2 experiences" in result  # Fallback message
