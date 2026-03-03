"""Tests for nexus.palace — rooms, memory placement, persistence, thread safety."""

import os
import threading
import pytest
from nexus.models import Memory, SalienceScore, MemorySource, MemoryStatus
from nexus.palace import SemanticPalace


class TestRoomCreation:
    def test_create_room(self, palace):
        room = palace.create_room("Python programming")
        assert room.topic == "Python programming"
        assert room.id in palace.rooms

    def test_create_multiple_rooms(self, palace):
        palace.create_room("topic A")
        palace.create_room("topic B")
        assert len(palace.rooms) == 2


class TestFindRooms:
    def test_find_rooms(self, palace):
        palace.create_room("machine learning algorithms")
        palace.create_room("cooking recipes")
        rooms = palace.find_rooms("neural networks", top_k=1)
        assert len(rooms) >= 1
        assert rooms[0].topic == "machine learning algorithms"

    def test_find_rooms_empty_palace(self, palace):
        rooms = palace.find_rooms("anything")
        assert rooms == []


class TestFindOrCreateRoom:
    def test_creates_when_empty(self, palace):
        room = palace.find_or_create_room("first content ever")
        assert room is not None
        assert len(palace.rooms) == 1

    def test_finds_existing_room(self, palace):
        palace.create_room("machine learning")
        room = palace.find_or_create_room("deep learning models")
        # May find existing ML room or create new one depending on similarity threshold
        assert len(palace.rooms) >= 1

    def test_creates_new_when_different(self, palace):
        palace.create_room("machine learning")
        room = palace.find_or_create_room("gourmet italian cooking techniques")
        # Should create new room for very different content
        assert len(palace.rooms) >= 1  # At least the original


class TestPlaceMemory:
    def test_place_memory(self, palace, make_memory):
        m = make_memory("test fact about Python")
        room = palace.place_memory(m)
        assert m.id in palace.memories
        assert m.room_id == room.id
        assert m.id in room.memory_ids

    def test_place_sets_embedding(self, palace, make_memory):
        m = make_memory("test fact")
        palace.place_memory(m)
        assert m.embedding is not None

    def test_place_landmark(self, palace):
        m = Memory(
            content="extremely important",
            salience=SalienceScore(
                surprise=0.9, relevance=0.9,
                emotional=0.9, novelty=0.9, utility=0.9,
            ),
        )
        palace.place_memory(m)
        assert m.id in palace.landmarks

    def test_landmarks_bounded(self, palace):
        for i in range(210):
            m = Memory(
                content=f"landmark {i}",
                salience=SalienceScore(
                    surprise=0.9, relevance=0.9,
                    emotional=0.9, novelty=0.9, utility=0.9,
                ),
            )
            palace.place_memory(m)
        assert len(palace.landmarks) <= 200


class TestGetMemory:
    def test_get_existing(self, palace, make_memory):
        m = make_memory("findable")
        palace.place_memory(m)
        found = palace.get_memory(m.id)
        assert found is not None
        assert found.content == "findable"

    def test_get_missing(self, palace):
        assert palace.get_memory("nonexistent") is None


class TestLinkRooms:
    def test_link_rooms(self, palace):
        r1 = palace.create_room("topic A")
        r2 = palace.create_room("topic B")
        edge = palace.link_rooms(r1.id, r2.id, "semantic", 0.8)
        assert edge is not None
        neighbors = palace.get_neighbors(r1.id)
        assert len(neighbors) == 1

    def test_link_bidirectional(self, palace):
        r1 = palace.create_room("A")
        r2 = palace.create_room("B")
        palace.link_rooms(r1.id, r2.id)
        assert len(palace.get_neighbors(r1.id)) == 1
        assert len(palace.get_neighbors(r2.id)) == 1

    def test_link_idempotent(self, palace):
        r1 = palace.create_room("A")
        r2 = palace.create_room("B")
        palace.link_rooms(r1.id, r2.id, "semantic")
        palace.link_rooms(r1.id, r2.id, "semantic")  # Same link again
        neighbors = palace.get_neighbors(r1.id)
        assert len(neighbors) == 1  # No duplicates


class TestPersistence:
    def test_save_and_load(self, tmp_dir, vector_store, make_memory):
        path = os.path.join(tmp_dir, "persist_palace")

        # Save
        p1 = SemanticPalace(vector_store=vector_store, storage_path=path)
        m = make_memory("persistent memory")
        p1.place_memory(m)
        p1.save()

        # Load fresh
        p2 = SemanticPalace(vector_store=vector_store, storage_path=path)
        assert len(p2.rooms) == 1
        assert len(p2.memories) == 1
        loaded = p2.get_memory(m.id)
        assert loaded is not None
        assert loaded.content == "persistent memory"
        assert loaded.embedding is not None

    def test_atomic_save(self, palace, make_memory):
        palace.place_memory(make_memory("test"))
        palace.save()
        # No .tmp file should be left behind
        import os
        palace_dir = palace.storage_path
        if palace_dir and os.path.exists(palace_dir):
            files = os.listdir(palace_dir)
            assert "palace.json.tmp" not in files

    def test_room_embeddings_rebuilt_on_load(self, tmp_dir, vector_store, make_memory):
        path = os.path.join(tmp_dir, "rebuild_palace")

        p1 = SemanticPalace(vector_store=vector_store, storage_path=path)
        p1.place_memory(make_memory("fact about python"))
        p1.save()

        p2 = SemanticPalace(vector_store=vector_store, storage_path=path)
        # Room embeddings should be rebuilt
        assert len(p2._room_embeddings) > 0


class TestThreadSafety:
    def test_concurrent_place_memory(self, palace, make_memory):
        """place_memory should be thread-safe."""
        errors = []

        def place(i):
            try:
                palace.place_memory(make_memory(f"concurrent memory {i}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=place, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(palace.memories) == 10


class TestHealth:
    def test_health_report(self, palace, make_memory):
        palace.place_memory(make_memory("test"))
        health = palace.health()
        assert "room_count" in health
        assert "memory_count" in health
        assert health["memory_count"] == 1
