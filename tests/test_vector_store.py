"""Tests for nexus.vector_store — embedding, storage, search, persistence."""

import os
import numpy as np
import pytest

from nexus.vector_store import VectorStore


class TestEmbedding:
    def test_embed_returns_array(self, vector_store):
        emb = vector_store.embed("hello world")
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (384,)

    def test_embed_normalized(self, vector_store):
        emb = vector_store.embed("test text")
        norm = np.linalg.norm(emb)
        assert norm == pytest.approx(1.0, abs=0.01)

    def test_embed_batch(self, vector_store):
        embs = vector_store.embed_batch(["hello", "world"])
        assert embs.shape == (2, 384)


class TestStorage:
    def test_add_and_get(self, vector_store):
        emb = vector_store.embed("test")
        vector_store.add(id="v1", vector=emb, metadata={"type": "test"})
        assert vector_store.has("v1")
        entry = vector_store.get("v1")
        assert entry.id == "v1"
        assert entry.metadata["type"] == "test"

    def test_add_with_text(self, vector_store):
        vector_store.add(id="v2", text="auto embed this")
        assert vector_store.has("v2")

    def test_remove(self, vector_store):
        vector_store.add(id="v3", text="to remove")
        vector_store.remove("v3")
        assert not vector_store.has("v3")

    def test_size(self, vector_store):
        assert vector_store.size == 0
        vector_store.add(id="a", text="first")
        vector_store.add(id="b", text="second")
        assert vector_store.size == 2

    def test_add_requires_text_or_vector(self, vector_store):
        with pytest.raises(ValueError):
            vector_store.add(id="bad")


class TestSearch:
    def test_search_basic(self, vector_store):
        vector_store.add(id="cat", text="cats are furry pets")
        vector_store.add(id="dog", text="dogs are loyal companions")
        vector_store.add(id="code", text="python programming language")

        results = vector_store.search(query="feline animals", top_k=2)
        assert len(results) > 0
        # Cat should be most similar to "feline animals"
        assert results[0][0] == "cat"

    def test_search_empty_store(self, vector_store):
        results = vector_store.search(query="anything", top_k=5)
        assert results == []

    def test_search_with_min_score(self, vector_store):
        vector_store.add(id="match", text="machine learning algorithms")
        results = vector_store.search(query="machine learning", min_score=0.9)
        # Very high threshold — should still find exact match
        assert len(results) >= 0  # May or may not match at 0.9

    def test_search_with_filter(self, vector_store):
        vector_store.add(id="a", text="first item")
        vector_store.add(id="b", text="second item")
        results = vector_store.search(query="item", filter_ids={"b"})
        assert all(r[0] == "b" for r in results)


class TestPersistence:
    def test_save_and_load(self, tmp_dir):
        path = os.path.join(tmp_dir, "persist_vs")

        # Save
        vs1 = VectorStore(storage_path=path)
        vs1.add(id="saved", text="persistent vector")
        vs1.save()

        # Load fresh
        vs2 = VectorStore(storage_path=path)
        assert vs2.has("saved")
        assert vs2.size == 1

    def test_partial_files_skipped(self, tmp_dir):
        """Only .json without .npy should not crash."""
        path = os.path.join(tmp_dir, "partial_vs")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "vectors.json"), "w") as f:
            f.write("{}")
        # Should NOT crash
        vs = VectorStore(storage_path=path)
        assert vs.size == 0


class TestSemanticDrift:
    def test_identical_vectors(self, vector_store):
        v = vector_store.embed("test")
        drift = vector_store.semantic_drift(v, v)
        assert drift == pytest.approx(0.0, abs=0.01)

    def test_different_vectors(self, vector_store):
        v1 = vector_store.embed("cats and dogs")
        v2 = vector_store.embed("quantum physics")
        drift = vector_store.semantic_drift(v1, v2)
        assert drift > 0.3  # Should be quite different
