"""
SMRITI v2 — Snippet Extractor.

Trims long memory content down to the sentences most relevant to a query.
Used by RetrievalEngine to reduce per-recall token spend without losing
the underlying memory (memory.content is never mutated; memory.snippet
is a transient field populated in-place).

See docs/superpowers/specs/2026-05-20-smarter-recall-design.md §5.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from smriti_memcore.fts_index import _STOP_WORDS
from smriti_memcore.models import Memory

logger = logging.getLogger(__name__)


@dataclass
class ExtractResult:
    used_mode: str
    fallback: bool = False


class SnippetExtractor:
    """See spec §5."""

    def __init__(
        self,
        vector_store,
        min_chars: int = 300,
        max_sentences: int = 2,
        llm=None,
    ):
        self.vector_store = vector_store
        self.min_chars = min_chars
        self.max_sentences = max_sentences
        self.llm = llm

    def extract(
        self,
        memory: Memory,
        query_variants: List[str],
        raw_query_embedding: np.ndarray,
        mode: str = "auto",
    ) -> ExtractResult:
        # Spec §5.2 — state-leak guard. Always clear before deciding what to populate.
        memory.snippet = None

        if mode == "none":
            return ExtractResult(used_mode="none")

        if len(memory.content) <= self.min_chars:
            # Already atomic; leave snippet as None and let serializer fall back to content
            return ExtractResult(used_mode=mode)

        if mode == "auto":
            return self._extract_auto(memory, query_variants, raw_query_embedding)
        if mode == "llm":
            # Implemented in Task 7
            raise NotImplementedError("llm mode added in Task 7")
        raise ValueError(f"Unknown mode {mode!r}")

    def _extract_auto(self, memory, query_variants, raw_query_embedding) -> ExtractResult:
        # Implemented in Task 5
        raise NotImplementedError("auto sentence-match added in Task 5")
