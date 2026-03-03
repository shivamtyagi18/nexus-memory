"""
Baseline: Naive RAG (Retrieval-Augmented Generation).
Standard approach: embed all messages → vector search top-k → stuff into context.
No memory management, no consolidation, no forgetting.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Dict, List, Optional

import numpy as np

from baselines.base import BaseMemorySystem, MemoryResponse
from nexus.vector_store import VectorStore
from nexus.llm_interface import LLMInterface


class NaiveRAG(BaseMemorySystem):
    """
    Simplest memory: vector store + top-k retrieval.
    No organization, no forgetting, no consolidation.
    """

    def __init__(self, llm: LLMInterface, vector_store: VectorStore):
        super().__init__("NaiveRAG", llm, vector_store)
        self._messages: List[Dict] = []

    def ingest(self, message: str, role: str = "user", metadata: Optional[Dict] = None):
        msg_id = f"msg_{self._ingest_count}"
        self.vector_store.add(id=msg_id, text=message, metadata={"role": role})
        self._messages.append({"id": msg_id, "content": message, "role": role})
        self._ingest_count += 1

    def query(self, question: str, context: str = "") -> MemoryResponse:
        def _do_query(q, ctx):
            # Retrieve top-k similar messages
            results = self.vector_store.search(query=q, top_k=5)
            retrieved = []
            for vec_id, score in results:
                for msg in self._messages:
                    if msg["id"] == vec_id:
                        retrieved.append(msg["content"])
                        break

            # Build prompt
            memory_text = "\n".join(f"- {r}" for r in retrieved) if retrieved else "No relevant memories."
            prompt = f"""Based on the following memories, answer the question.

Memories:
{memory_text}

Question: {question}
Answer:"""

            response = self.llm.generate(prompt)
            return MemoryResponse(
                answer=response.text.strip(),
                memories_used=retrieved,
                tokens_used=response.tokens_used,
            )

        return self._timed_query(_do_query, question, context)

    def reset(self):
        self._messages.clear()
        self._ingest_count = 0
        self._query_count = 0
        self._total_latency_ms = 0.0
        # Reset vector store
        self.vector_store._vectors.clear()
        self.vector_store._matrix_dirty = True

    def get_stats(self) -> Dict:
        base = super().get_stats()
        base["total_messages"] = len(self._messages)
        base["vector_count"] = self.vector_store.size
        return base
