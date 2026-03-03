"""
Base interface for all memory systems (NEXUS + baselines).
Defines the common API that the benchmark harness talks to.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time


@dataclass
class MemoryResponse:
    """Standardized response from a memory system."""
    answer: str
    memories_used: List[str] = field(default_factory=list)
    confidence: float = 0.0
    latency_ms: float = 0.0
    tokens_used: int = 0


class BaseMemorySystem(ABC):
    """
    Abstract base for all memory systems.
    
    Every system (NEXUS + 4 baselines) implements this interface
    so the benchmark harness can treat them uniformly.
    """

    def __init__(self, name: str, llm_interface=None, vector_store=None):
        self.name = name
        self.llm = llm_interface
        self.vector_store = vector_store
        self._ingest_count = 0
        self._query_count = 0
        self._total_latency_ms = 0.0

    @abstractmethod
    def ingest(self, message: str, role: str = "user", metadata: Optional[Dict] = None):
        """
        Ingest a message into the memory system.
        Called for each message in a conversation.
        """
        pass

    @abstractmethod
    def query(self, question: str, context: str = "") -> MemoryResponse:
        """
        Query the memory system and generate an answer.
        Returns a MemoryResponse with the answer and metadata.
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset the memory system (clear all stored data)."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics for benchmarking."""
        return {
            "name": self.name,
            "messages_ingested": self._ingest_count,
            "queries_answered": self._query_count,
            "avg_latency_ms": (
                self._total_latency_ms / self._query_count
                if self._query_count > 0 else 0
            ),
        }

    def _timed_query(self, fn, question: str, context: str = "") -> MemoryResponse:
        """Utility: time a query function and record latency."""
        start = time.time()
        response = fn(question, context)
        elapsed = (time.time() - start) * 1000
        response.latency_ms = elapsed
        self._query_count += 1
        self._total_latency_ms += elapsed
        return response
