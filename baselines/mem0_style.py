"""
Baseline: Mem0-Style Memory System.
Auto-extract key facts from conversations, store as structured memory entries,
retrieve by similarity. No consolidation, no reflection, no forgetting.
Inspired by Mem0's architecture.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Dict, List, Optional

from baselines.base import BaseMemorySystem, MemoryResponse
from nexus.vector_store import VectorStore
from nexus.llm_interface import LLMInterface


class Mem0Style(BaseMemorySystem):
    """
    Mem0-inspired: extract key facts → store → retrieve by similarity.
    Better than naive RAG because it extracts structured facts,
    but no consolidation, reflection, or forgetting.
    """

    def __init__(self, llm: LLMInterface, vector_store: VectorStore):
        super().__init__("Mem0Style", llm, vector_store)
        self._memories: List[Dict] = []
        self._conversation: List[Dict] = []

    def ingest(self, message: str, role: str = "user", metadata: Optional[Dict] = None):
        self._conversation.append({"role": role, "content": message})
        self._ingest_count += 1

        # Extract key facts using LLM (Mem0's core mechanism)
        if role == "user" and len(message) > 20:
            facts = self._extract_facts(message)
            for fact in facts:
                self._store_fact(fact)

    def _extract_facts(self, message: str) -> List[str]:
        """Extract key facts from a message using LLM."""
        result = self.llm.generate_json(
            f"""Extract key facts or preferences from this message. 
Return as JSON: {{"facts": ["fact1", "fact2", ...]}}
If no important facts, return {{"facts": []}}

Message: "{message}"
""",
            temperature=0.1,
        )
        return result.get("facts", [])

    def _store_fact(self, fact: str):
        """Store a fact with deduplication check."""
        # Check for duplicates
        if self.vector_store.size > 0:
            similar = self.vector_store.search(query=fact, top_k=1)
            if similar and similar[0][1] > 0.9:
                # Very similar fact exists — update instead of duplicate
                existing_id = similar[0][0]
                for mem in self._memories:
                    if mem["id"] == existing_id:
                        mem["content"] = fact  # Update with newer version
                        self.vector_store.add(id=existing_id, text=fact)
                        return

        mem_id = f"fact_{len(self._memories)}"
        self.vector_store.add(id=mem_id, text=fact)
        self._memories.append({"id": mem_id, "content": fact})

    def query(self, question: str, context: str = "") -> MemoryResponse:
        def _do_query(q, ctx):
            # Retrieve relevant facts
            results = self.vector_store.search(query=q, top_k=5)
            retrieved = []
            for vec_id, score in results:
                for mem in self._memories:
                    if mem["id"] == vec_id:
                        retrieved.append(mem["content"])
                        break

            # Build prompt with retrieved memories
            memory_text = "\n".join(f"- {r}" for r in retrieved) if retrieved else "No relevant memories."

            prompt = f"""You have the following memories about the user and past conversations:

Memories:
{memory_text}

Question: {question}
Answer based only on the memories above:"""

            response = self.llm.generate(prompt)
            return MemoryResponse(
                answer=response.text.strip(),
                memories_used=retrieved,
                tokens_used=response.tokens_used,
            )

        return self._timed_query(_do_query, question, context)

    def reset(self):
        self._memories.clear()
        self._conversation.clear()
        self._ingest_count = 0
        self._query_count = 0
        self._total_latency_ms = 0.0
        self.vector_store._vectors.clear()
        self.vector_store._matrix_dirty = True

    def get_stats(self) -> Dict:
        base = super().get_stats()
        base["extracted_facts"] = len(self._memories)
        base["conversation_length"] = len(self._conversation)
        return base
