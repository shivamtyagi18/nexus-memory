"""
Baseline: Full Context.
Stuff the entire conversation history into the LLM context window.
Represents the "infinite context window" approach — truncate when too long.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Dict, List, Optional

from baselines.base import BaseMemorySystem, MemoryResponse
from nexus.llm_interface import LLMInterface


class FullContext(BaseMemorySystem):
    """
    Simplest possible approach: concatenate everything into context.
    Truncate from the beginning when it gets too long.
    """

    def __init__(self, llm: LLMInterface, max_tokens: int = 4000):
        super().__init__("FullContext", llm)
        self._history: List[Dict] = []
        self.max_tokens = max_tokens

    def ingest(self, message: str, role: str = "user", metadata: Optional[Dict] = None):
        self._history.append({"role": role, "content": message})
        self._ingest_count += 1

    def query(self, question: str, context: str = "") -> MemoryResponse:
        def _do_query(q, ctx):
            # Build full conversation context
            history_text = self._build_context()

            prompt = f"""Based on the following conversation history, answer the question.

Conversation history:
{history_text}

Question: {question}
Answer:"""

            response = self.llm.generate(prompt, max_tokens=512)
            return MemoryResponse(
                answer=response.text.strip(),
                memories_used=[f"{len(self._history)} messages in context"],
                tokens_used=response.tokens_used,
            )

        return self._timed_query(_do_query, question, context)

    def _build_context(self) -> str:
        """Build context from history, truncating from the beginning if needed."""
        lines = []
        for msg in self._history:
            lines.append(f"[{msg['role']}]: {msg['content']}")

        full_text = "\n".join(lines)

        # Simple truncation: estimate ~4 chars per token
        max_chars = self.max_tokens * 4
        if len(full_text) > max_chars:
            # Keep the most recent messages
            full_text = "...[earlier messages truncated]...\n" + full_text[-max_chars:]

        return full_text

    def reset(self):
        self._history.clear()
        self._ingest_count = 0
        self._query_count = 0
        self._total_latency_ms = 0.0

    def get_stats(self) -> Dict:
        base = super().get_stats()
        base["history_length"] = len(self._history)
        total_chars = sum(len(m["content"]) for m in self._history)
        base["total_chars"] = total_chars
        base["estimated_tokens"] = total_chars // 4
        return base
