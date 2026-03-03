"""
NEXUS adapter for the benchmark harness.
Wraps the NEXUS core into the BaseMemorySystem interface.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Dict, List, Optional

from baselines.base import BaseMemorySystem, MemoryResponse
from nexus.core import NEXUS
from nexus.models import MemorySource, NexusConfig
from nexus.llm_interface import LLMInterface


class NexusAdapter(BaseMemorySystem):
    """Wraps NEXUS v2 into the BaseMemorySystem benchmark interface."""

    def __init__(self, llm: LLMInterface, config: Optional[NexusConfig] = None):
        super().__init__("NEXUS_v2", llm)
        self.config = config or NexusConfig()
        self.nexus = NEXUS(self.config)

    def ingest(self, message: str, role: str = "user", metadata: Optional[Dict] = None):
        source = MemorySource.USER_STATED if role == "user" else MemorySource.DIRECT
        # Use fast heuristic scoring for benchmark speed
        # Temporarily disable auto-consolidation during batch ingest
        original_trigger = self.nexus.config.episode_buffer_trigger
        self.nexus.config.episode_buffer_trigger = 99999  # Prevent mid-ingest consolidation
        self.nexus.encode(message, source=source, use_llm=False)
        self.nexus.config.episode_buffer_trigger = original_trigger
        self._ingest_count += 1

    def query(self, question: str, context: str = "") -> MemoryResponse:
        def _do_query(q, ctx):
            # Recall memories
            memories = self.nexus.recall(q, context=ctx, top_k=5)

            # Build prompt with retrieved memories and confidence
            confidence = self.nexus.how_well_do_i_know(q)
            memory_texts = [m.content for m in memories]
            memory_str = "\n".join(f"- {t}" for t in memory_texts) if memory_texts else "No relevant memories."

            confidence_note = ""
            if confidence.overall < 0.3:
                confidence_note = "\nNote: I have limited knowledge on this topic."

            prompt = f"""Based on the following memories, answer the question.{confidence_note}

Memories:
{memory_str}

Question: {question}
Answer:"""

            response = self.nexus.llm.generate(prompt)
            return MemoryResponse(
                answer=response.text.strip(),
                memories_used=memory_texts,
                confidence=confidence.overall,
                tokens_used=response.tokens_used,
            )

        return self._timed_query(_do_query, question, context)

    def reset(self):
        import shutil
        self._ingest_count = 0
        self._query_count = 0
        self._total_latency_ms = 0.0
        # Re-initialize NEXUS
        if os.path.exists(self.config.storage_path):
            shutil.rmtree(self.config.storage_path, ignore_errors=True)
        self.nexus = NEXUS(self.config)

    def get_stats(self) -> Dict:
        base = super().get_stats()
        base.update(self.nexus.stats())
        return base
