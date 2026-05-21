"""Tests for SnippetExtractor — sentence-level snippet generation.

Uses `vector_store` and `make_memory` fixtures from tests/conftest.py.
"""
import numpy as np
import pytest


class TestExtractResult:
    def test_dataclass_shape(self):
        from smriti_memcore.snippet import ExtractResult
        r = ExtractResult(used_mode="auto")
        assert r.used_mode == "auto"
        assert r.fallback is False


class TestNoneMode:
    def test_clears_snippet_and_returns(self, vector_store, make_memory):
        from smriti_memcore.snippet import SnippetExtractor
        m = make_memory("anything")
        m.snippet = "leftover from prior recall"
        extractor = SnippetExtractor(vector_store=vector_store)
        result = extractor.extract(m, ["q"], np.zeros(384), mode="none")
        assert m.snippet is None
        assert result.used_mode == "none"


class TestThresholdShortCircuit:
    def test_short_content_not_extracted(self, vector_store, make_memory):
        from smriti_memcore.snippet import SnippetExtractor
        m = make_memory("short.")  # well below 300 chars
        m.snippet = "stale"
        extractor = SnippetExtractor(vector_store=vector_store, min_chars=300)
        extractor.extract(m, ["q"], np.zeros(384), mode="auto")
        # State-leak guard: snippet always cleared, even on short-circuit
        assert m.snippet is None

    def test_exactly_at_min_chars_still_short_circuits(self, vector_store, make_memory):
        """Spec §5.4: `≤ min_chars` short-circuits (inclusive)."""
        from smriti_memcore.snippet import SnippetExtractor
        m = make_memory("x" * 300)
        extractor = SnippetExtractor(vector_store=vector_store, min_chars=300)
        extractor.extract(m, ["x"], np.zeros(384), mode="auto")
        assert m.snippet is None


class TestStateLeakGuard:
    def test_extract_always_clears_snippet_on_entry(self, vector_store, make_memory):
        from smriti_memcore.snippet import SnippetExtractor
        m = make_memory("anything short")
        m.snippet = "stale from prior call"
        extractor = SnippetExtractor(vector_store=vector_store)
        # Even mode='none' must clear (spec §5.2)
        extractor.extract(m, [], np.zeros(384), mode="none")
        assert m.snippet is None
