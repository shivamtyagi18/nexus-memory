"""
NEXUS v2 — Meta-Memory.
Self-awareness layer: confidence mapping, knowledge gap tracking,
and ask-vs-recall decision engine. Prevents hallucination by knowing
what the agent knows and doesn't know.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

from nexus.models import ConfidenceLevel, DecisionType, MemoryStatus
from nexus.palace import SemanticPalace

logger = logging.getLogger(__name__)


class MetaMemory:
    """
    The agent's awareness of its own knowledge landscape.
    
    Without this, agents hallucinate — they generate answers about 
    topics they have no real knowledge of, because they can't 
    distinguish "I know this well" from "I have a vaguely similar 
    vector somewhere."
    """

    def __init__(self, palace: SemanticPalace):
        self.palace = palace

        # Knowledge gap registry (bounded to prevent memory leaks)
        self._gap_registry: deque = deque(maxlen=200)
        self._failed_retrievals: deque = deque(maxlen=500)

        # Thresholds
        self.confidence_threshold = 0.3   # Below this → admit gap
        self.stale_threshold = 0.4        # Below this freshness → verify

    def confidence_map(self, topic: str) -> ConfidenceLevel:
        """
        How well does the agent know this topic?
        
        Returns a structured confidence assessment considering:
        - Coverage: how many memories exist on this topic
        - Freshness: how recently the knowledge was accessed
        - Strength: average memory strength (consolidated knowledge is stronger)
        - Depth: highest reflection level (raw episodes vs principles)
        """
        rooms = self.palace.find_rooms(topic, top_k=2)

        if not rooms:
            return ConfidenceLevel()  # Unknown

        # Aggregate across relevant rooms
        all_memories = []
        for room in rooms:
            all_memories.extend(self.palace.get_room_memories(room.id))

        if not all_memories:
            return ConfidenceLevel()

        now = datetime.now()

        # Coverage: ratio of memories to expected coverage
        # (heuristic: we expect ~10 memories for a well-known topic)
        expected = 10
        coverage = min(len(all_memories) / expected, 1.0)

        # Freshness: average recency (exponential decay)
        freshness_scores = []
        for mem in all_memories:
            days = (now - mem.last_accessed).total_seconds() / 86400
            freshness_scores.append(0.95 ** days)
        freshness = sum(freshness_scores) / len(freshness_scores)

        # Strength: average memory strength
        strength = sum(m.strength for m in all_memories) / len(all_memories)
        strength = min(strength / 3.0, 1.0)  # Normalize

        # Depth: max reflection level
        depth = max(m.reflection_level for m in all_memories)

        return ConfidenceLevel(
            coverage=coverage,
            freshness=freshness,
            strength=strength,
            depth=depth,
        )

    def should_recall_or_ask(self, query: str) -> DecisionType:
        """
        Should the agent try to recall, or admit ignorance?
        
        This is the difference between an agent that says "I don't know
        enough about this" and one that hallucinates confidently.
        """
        conf = self.confidence_map(query)

        if conf.is_unknown or conf.coverage < self.confidence_threshold:
            return DecisionType.ADMIT_GAP_AND_ASK
        elif conf.freshness < self.stale_threshold:
            return DecisionType.RECALL_BUT_VERIFY
        else:
            return DecisionType.RECALL_CONFIDENTLY

    def knowledge_gaps(self) -> List[Dict]:
        """What does the agent know it doesn't know?"""
        return list(self._gap_registry)

    def register_gap(self, topic: str, context: str = ""):
        """Record a knowledge gap (from failed retrievals, unresolved questions)."""
        gap = {
            "topic": topic,
            "context": context,
            "discovered_at": datetime.now().isoformat(),
            "resolved": False,
        }
        self._gap_registry.append(gap)
        logger.info(f"Knowledge gap registered: {topic}")

    def register_failed_retrieval(self, query: str, context: str = ""):
        """Record a failed retrieval attempt."""
        self._failed_retrievals.append({
            "query": query,
            "context": context,
            "timestamp": datetime.now().isoformat(),
        })

        # If we've failed to retrieve on similar topics 3+ times, register as a gap
        similar_failures = [
            f for f in self._failed_retrievals
            if _topic_overlap(f["query"], query)
        ]
        if len(similar_failures) >= 3:
            self.register_gap(query)

    def resolve_gap(self, topic: str):
        """Mark a knowledge gap as resolved."""
        for gap in self._gap_registry:
            if gap["topic"] == topic and not gap["resolved"]:
                gap["resolved"] = True
                logger.info(f"Knowledge gap resolved: {topic}")
                break

    def get_confidence_summary(self) -> str:
        """Human-readable summary of the agent's knowledge state."""
        rooms = self.palace.rooms
        if not rooms:
            return "No knowledge stored yet."

        lines = ["Knowledge confidence map:"]
        for room in rooms.values():
            conf = self.confidence_map(room.topic)
            emoji = "🟢" if conf.overall > 0.7 else "🟡" if conf.overall > 0.4 else "🔴"
            lines.append(
                f"  {emoji} {room.topic}: "
                f"coverage={conf.coverage:.0%}, "
                f"freshness={conf.freshness:.0%}, "
                f"depth=L{conf.depth}"
            )

        gaps = [g for g in self._gap_registry if not g["resolved"]]
        if gaps:
            lines.append(f"\nKnown gaps ({len(gaps)}):")
            for gap in gaps[-5:]:
                lines.append(f"  ❓ {gap['topic']}")

        return "\n".join(lines)

    def stats(self) -> dict:
        """Meta-memory statistics."""
        active_gaps = [g for g in self._gap_registry if not g["resolved"]]
        return {
            "total_rooms": len(self.palace.rooms),
            "active_gaps": len(active_gaps),
            "resolved_gaps": len(self._gap_registry) - len(active_gaps),
            "failed_retrievals": len(self._failed_retrievals),
        }


def _topic_overlap(query_a: str, query_b: str) -> bool:
    """Simple heuristic for topic overlap (word intersection)."""
    words_a = set(query_a.lower().split())
    words_b = set(query_b.lower().split())
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
    return overlap > 0.5
