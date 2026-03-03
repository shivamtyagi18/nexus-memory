"""
Baseline: MemGPT-Style Memory System.
Tiered memory (main context + archival), agent-directed memory management
via function calls, summarization on eviction.
Inspired by MemGPT/Letta's architecture.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Dict, List, Optional

from baselines.base import BaseMemorySystem, MemoryResponse
from nexus.vector_store import VectorStore
from nexus.llm_interface import LLMInterface


class MemGPTStyle(BaseMemorySystem):
    """
    MemGPT-inspired: tiered memory with main context + archival.
    When main context is full, summarize and evict to archival.
    Retrieve from archival when needed.
    """

    def __init__(
        self, llm: LLMInterface, vector_store: VectorStore,
        main_context_size: int = 10, archival_search_k: int = 5,
    ):
        super().__init__("MemGPTStyle", llm, vector_store)
        self.main_context_size = main_context_size
        self.archival_search_k = archival_search_k

        # Main context (like RAM)
        self._main_context: List[Dict] = []
        # Archival storage (like disk)
        self._archival: List[Dict] = []
        # Summaries of evicted context
        self._summaries: List[str] = []

    def ingest(self, message: str, role: str = "user", metadata: Optional[Dict] = None):
        self._main_context.append({"role": role, "content": message})
        self._ingest_count += 1

        # Evict to archival when main context is full
        if len(self._main_context) > self.main_context_size:
            self._evict_to_archival()

    def _evict_to_archival(self):
        """Summarize oldest messages and move to archival storage."""
        # Take the oldest half of main context
        n_evict = len(self._main_context) // 2
        to_evict = self._main_context[:n_evict]
        self._main_context = self._main_context[n_evict:]

        # Summarize the evicted messages
        evicted_text = "\n".join(f"[{m['role']}]: {m['content']}" for m in to_evict)
        summary_result = self.llm.generate(
            f"Summarize this conversation segment in 2-3 sentences:\n{evicted_text}",
            max_tokens=200,
        )
        summary = summary_result.text.strip()
        self._summaries.append(summary)

        # Store each evicted message in archival (vector store)
        for i, msg in enumerate(to_evict):
            arch_id = f"arch_{len(self._archival)}"
            self.vector_store.add(id=arch_id, text=msg["content"])
            self._archival.append({"id": arch_id, **msg})

    def query(self, question: str, context: str = "") -> MemoryResponse:
        def _do_query(q, ctx):
            # Build main context
            main_text = "\n".join(
                f"[{m['role']}]: {m['content']}" for m in self._main_context
            )

            # Search archival for relevant past messages
            archival_text = ""
            archival_retrieved = []
            if self.vector_store.size > 0:
                results = self.vector_store.search(query=q, top_k=self.archival_search_k)
                for vec_id, score in results:
                    for arch in self._archival:
                        if arch["id"] == vec_id:
                            archival_retrieved.append(arch["content"])
                            break

                if archival_retrieved:
                    archival_text = "\n".join(f"- {r}" for r in archival_retrieved)

            # Include summaries of past context
            summary_text = "\n".join(self._summaries[-3:]) if self._summaries else ""

            prompt = f"""You have access to conversation memory at different levels:

CURRENT CONTEXT (recent messages):
{main_text}

ARCHIVAL MEMORY (relevant past messages):
{archival_text if archival_text else 'None retrieved.'}

SUMMARIES OF PAST CONVERSATIONS:
{summary_text if summary_text else 'None.'}

Question: {question}
Answer:"""

            response = self.llm.generate(prompt)
            return MemoryResponse(
                answer=response.text.strip(),
                memories_used=archival_retrieved + self._summaries[-3:],
                tokens_used=response.tokens_used,
            )

        return self._timed_query(_do_query, question, context)

    def reset(self):
        self._main_context.clear()
        self._archival.clear()
        self._summaries.clear()
        self._ingest_count = 0
        self._query_count = 0
        self._total_latency_ms = 0.0
        self.vector_store._vectors.clear()
        self.vector_store._matrix_dirty = True

    def get_stats(self) -> Dict:
        base = super().get_stats()
        base["main_context_size"] = len(self._main_context)
        base["archival_size"] = len(self._archival)
        base["summaries"] = len(self._summaries)
        return base
