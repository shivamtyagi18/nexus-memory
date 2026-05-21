"""Tests for smriti.retrieval — multi-hop search, spreading activation."""

import pytest
from smriti_memcore.models import SmritiConfig, Memory, SalienceScore
from smriti_memcore.retrieval import RetrievalEngine


@pytest.fixture
def retrieval_engine(palace, vector_store, working_memory, config, mock_llm):
    from smriti_memcore.retrieval import RetrievalEngine
    return RetrievalEngine(
        palace=palace,
        working_memory=working_memory,
        vector_store=vector_store,
        config=config,
    )


class TestBasicRetrieval:
    def test_retrieve_finds_memories(self, palace, vector_store, working_memory, make_memory):
        config = SmritiConfig()
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
        config = SmritiConfig()
        engine = RetrievalEngine(
            palace=palace, working_memory=working_memory,
            vector_store=vector_store, config=config,
        )
        results = engine.retrieve("nothing here", top_k=5)
        assert results == []


class TestRetrievalStats:
    def test_stats(self, palace, vector_store, working_memory):
        config = SmritiConfig()
        engine = RetrievalEngine(
            palace=palace, working_memory=working_memory,
            vector_store=vector_store, config=config,
        )
        s = engine.stats()
        assert "total_retrievals" in s


class TestMultipleMemories:
    def test_top_k_respected(self, palace, vector_store, working_memory, make_memory):
        config = SmritiConfig()
        engine = RetrievalEngine(
            palace=palace, working_memory=working_memory,
            vector_store=vector_store, config=config,
        )

        for i in range(10):
            palace.place_memory(make_memory(f"memory about topic {i}"))

        results = engine.retrieve("topic", top_k=3)
        assert len(results) <= 3


class TestSmartRecallWiring:
    """Spec §3 — RetrievalEngine wires QueryRewriter + SnippetExtractor."""

    def test_retrieve_accepts_rewrite_and_snippet_params(self, retrieval_engine):
        """New params must be accepted without raising."""
        try:
            retrieval_engine.retrieve("query", rewrite="none", snippet="none")
        except TypeError as e:
            pytest.fail(f"retrieve() should accept rewrite/snippet params: {e}")

    def test_retrieve_uses_config_default_when_param_is_none(self, retrieval_engine):
        """Spec §8.1 — None sentinel falls through to config defaults."""
        # Hard to assert without integration; smoke-test that None is accepted.
        retrieval_engine.retrieve("query", rewrite=None, snippet=None)

    @pytest.fixture
    def smriti_for_leakage(self, tmp_dir, mock_llm):
        """Local copy of the SMRITI fixture from tests/test_core.py — that one isn't
        in conftest.py so we can't import it. Same pattern: real SMRITI with mocked LLM."""
        import os
        from smriti_memcore.models import SmritiConfig
        from smriti_memcore import SMRITI
        config = SmritiConfig(storage_path=os.path.join(tmp_dir, "smriti_db"))
        n = SMRITI(config=config)
        n.llm = mock_llm
        n.attention_gate.llm = mock_llm
        n.consolidation_engine.llm = mock_llm
        yield n
        n.close()

    def test_relevance_score_does_not_leak_across_recalls(self, smriti_for_leakage, monkeypatch):
        """Spec §6 stale-state guard: a memory entering the candidate pool via FTS-only
        (palace.search did not score it) must have its relevance_score cleared BEFORE
        _score_memory consumes it.

        To PROVE the FTS-only path is exercised (not just relying on palace.search
        happening to skip the memory), we monkeypatch palace.search to return an empty
        list. The memory then can only enter the candidate pool via FTS, exercising the
        stale-state guard. Final assertion is the strict `== 0.0`.
        """
        from smriti_memcore.models import MemorySource
        smriti = smriti_for_leakage

        # Seed memory with a rare lexical token + over-threshold content length
        mid = smriti.encode(
            "tokenxyzzy is a placeholder rare lexical marker used by the leakage test " * 8,
            source=MemorySource.USER_STATED,
        )
        if mid is None:
            import pytest
            pytest.skip("attention gate discarded the seeded memory")

        mem = smriti.palace.memories[mid]
        # Stamp the stale value — simulates a leaked lifted score from a prior recall
        mem.relevance_score = 0.95
        assert mem.relevance_score == 0.95  # sanity

        # FORCE the FTS-only path: stub palace.search to return [] so the memory cannot
        # enter via the vector pipeline. It must arrive via FTS, where the guard fires.
        monkeypatch.setattr(smriti.palace, "search", lambda *a, **k: [])

        results = smriti.recall("tokenxyzzy", rewrite="none", snippet="none", top_k=10)

        # The memory must appear in results (FTS found it via the rare token)
        assert any(m.id == mid for m in results), (
            "FTS-only path didn't surface the rare-token memory — test setup is wrong, "
            "OR FTS isn't wired in this SMRITI instance"
        )

        # STRICT assertion: the stale 0.95 must have been cleared to exactly 0.0 by the
        # guard. palace.search returned []; nothing else writes relevance_score in retrieve().
        assert mem.relevance_score == 0.0, (
            f"stale-state guard failed: expected relevance_score == 0.0 after FTS-only "
            f"recall, got {mem.relevance_score}"
        )
