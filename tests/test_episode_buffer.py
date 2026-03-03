"""Tests for nexus.episode_buffer — add, search, consolidation, persistence."""

import pytest
from nexus.models import Episode, SalienceScore, MemorySource


class TestAdd:
    def test_add_episode(self, episode_buffer, make_episode):
        ep = make_episode("test content")
        ep_id = episode_buffer.add(ep)
        assert ep_id is not None
        assert episode_buffer.count == 1

    def test_add_multiple(self, episode_buffer, make_episode):
        for i in range(5):
            episode_buffer.add(make_episode(f"episode {i}"))
        assert episode_buffer.count == 5
        assert episode_buffer.unconsolidated_count == 5


class TestGet:
    def test_get_existing(self, episode_buffer, make_episode):
        ep = make_episode("find me")
        episode_buffer.add(ep)
        found = episode_buffer.get(ep.id)
        assert found is not None
        assert found.content == "find me"

    def test_get_missing(self, episode_buffer):
        assert episode_buffer.get("nonexistent") is None


class TestRemove:
    def test_remove_episode(self, episode_buffer, make_episode):
        ep = make_episode("remove me")
        episode_buffer.add(ep)
        episode_buffer.remove(ep.id)
        assert episode_buffer.get(ep.id) is None


class TestSearch:
    def test_get_recent(self, episode_buffer, make_episode):
        for i in range(5):
            episode_buffer.add(make_episode(f"episode {i}"))
        recent = episode_buffer.get_recent(n=3)
        assert len(recent) == 3

    def test_search_semantic(self, episode_buffer, make_episode):
        episode_buffer.add(make_episode("cats are furry pets"))
        episode_buffer.add(make_episode("python programming"))
        results = episode_buffer.search_semantic("feline animals", top_k=1)
        assert len(results) >= 1
        assert "cat" in results[0].content.lower()

    def test_get_high_salience(self, episode_buffer):
        low = Episode(content="boring", salience=SalienceScore(relevance=0.1))
        high = Episode(content="critical", salience=SalienceScore(
            surprise=0.9, relevance=0.9, emotional=0.9, novelty=0.9, utility=0.9,
        ))
        episode_buffer.add(low)
        episode_buffer.add(high)
        results = episode_buffer.get_high_salience(min_composite=0.7)
        assert len(results) == 1
        assert results[0].content == "critical"


class TestConsolidation:
    def test_mark_consolidated(self, episode_buffer, make_episode):
        ep = make_episode("to consolidate")
        episode_buffer.add(ep)
        assert episode_buffer.unconsolidated_count == 1

        episode_buffer.mark_consolidated([ep.id])
        assert episode_buffer.unconsolidated_count == 0
        # Consolidated episodes are removed from RAM
        assert ep.id not in episode_buffer._episodes
        # But total count stays the same
        assert episode_buffer.count == 1

    def test_get_unconsolidated(self, episode_buffer, make_episode):
        ep1 = make_episode("one")
        ep2 = make_episode("two")
        episode_buffer.add(ep1)
        episode_buffer.add(ep2)
        episode_buffer.mark_consolidated([ep1.id])

        unconsolidated = episode_buffer.get_unconsolidated()
        assert len(unconsolidated) == 1
        assert unconsolidated[0].id == ep2.id


class TestClose:
    def test_close_idempotent(self, episode_buffer):
        episode_buffer.close()
        episode_buffer.close()  # Should not crash

    def test_save_after_close_safe(self, episode_buffer, make_episode):
        episode_buffer.close()
        ep = make_episode("late")
        episode_buffer._save_episode(ep)  # Should not crash

    def test_mark_consolidated_after_close(self, episode_buffer):
        episode_buffer.close()
        episode_buffer.mark_consolidated(["fake"])  # Should not crash


class TestPersistence:
    def test_episodes_survive_restart(self, tmp_dir, vector_store, make_episode):
        """Episodes should be loadable from SQLite after restart."""
        import os
        from nexus.episode_buffer import EpisodeBuffer

        path = os.path.join(tmp_dir, "persist_eb")

        # Create and add
        eb1 = EpisodeBuffer(storage_path=path, vector_store=vector_store)
        ep = make_episode("persistent episode")
        eb1.add(ep)
        eb1.close()

        # Reload
        eb2 = EpisodeBuffer(storage_path=path, vector_store=vector_store)
        assert eb2.count == 1
        assert eb2.unconsolidated_count == 1
        found = eb2.get(ep.id)
        assert found is not None
        assert found.content == "persistent episode"
        eb2.close()
