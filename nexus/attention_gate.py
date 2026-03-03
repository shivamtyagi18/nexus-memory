"""
NEXUS v2 — Attention Gate.
Multi-dimensional salience scoring with conflict detection on ingest.
Inspired by the amygdala's role in emotional tagging of memories.
"""

from __future__ import annotations

import logging
from typing import Optional

from nexus.models import (
    Episode, Memory, MemorySource, NexusConfig, SalienceScore,
)
from nexus.llm_interface import LLMInterface

logger = logging.getLogger(__name__)


class AttentionGate:
    """
    The first filter for incoming information — decides what gets encoded
    with full detail vs. summarized vs. discarded.
    
    Computes a 5-dimensional salience score and routes accordingly:
    - High salience → full encoding to Episode Buffer + Palace
    - Medium salience → summary encoding
    - Low salience → discard
    """

    def __init__(self, llm: LLMInterface, config: NexusConfig):
        self.llm = llm
        self.config = config

        # Thresholds
        self.high_threshold = 0.6
        self.low_threshold = 0.2

    def score(
        self,
        content: str,
        context: str = "",
        source: MemorySource = MemorySource.DIRECT,
    ) -> SalienceScore:
        """
        Score content on 5 salience dimensions using the LLM.
        
        For USER_STATED source, automatically boost relevance and utility
        (user-stated information is inherently important).
        """
        # Get LLM scores
        scores = self.llm.score_salience(content, context)

        # Fall back to fast scoring if LLM returned an error
        if "error" in scores:
            logger.warning(f"LLM scoring failed, using fast heuristic: {scores['error']}")
            return self.score_fast(content, context, source)

        salience = SalienceScore(
            surprise=scores.get("surprise", 0.5),
            relevance=scores.get("relevance", 0.5),
            emotional=scores.get("emotional", 0.3),
            novelty=scores.get("novelty", 0.5),
            utility=scores.get("utility", 0.5),
        )

        # Source-based adjustments
        if source == MemorySource.USER_STATED:
            salience.relevance = max(salience.relevance, 0.8)
            salience.utility = max(salience.utility, 0.7)
        elif source == MemorySource.EXTERNAL:
            # Lower trust for external/shared memories
            salience.relevance *= 0.8
            salience.utility *= 0.8

        return salience

    def score_fast(
        self,
        content: str,
        context: str = "",
        source: MemorySource = MemorySource.DIRECT,
    ) -> SalienceScore:
        """
        Fast heuristic salience scoring (no LLM call).
        Used during high-throughput periods or when LLM is unavailable.
        """
        # Simple heuristics
        content_len = len(content)
        has_question = "?" in content
        has_instruction = any(w in content.lower() for w in [
            "always", "never", "must", "should", "remember", "important",
            "don't forget", "note that", "key point",
        ])
        has_code = "```" in content or "def " in content or "class " in content
        has_error = any(w in content.lower() for w in [
            "error", "bug", "fail", "crash", "exception", "warning",
        ])
        is_short = content_len < 20

        salience = SalienceScore(
            surprise=0.6 if has_error else 0.3,
            relevance=0.7 if has_question or has_instruction else 0.4,
            emotional=0.6 if has_error else 0.2,
            novelty=0.5,
            utility=0.8 if has_instruction or has_code else 0.4,
        )

        # Very short messages are usually low salience
        if is_short:
            salience.utility *= 0.5
            salience.relevance *= 0.5

        # Source adjustments
        if source == MemorySource.USER_STATED:
            salience.relevance = max(salience.relevance, 0.8)
            salience.utility = max(salience.utility, 0.7)

        return salience

    def should_encode(self, salience: SalienceScore) -> str:
        """
        Decide encoding level based on salience.
        Returns: "full", "summary", or "discard"
        """
        composite = salience.composite

        if composite >= self.high_threshold:
            return "full"
        elif composite >= self.low_threshold:
            return "summary"
        else:
            return "discard"

    def process(
        self,
        content: str,
        context: str = "",
        source: MemorySource = MemorySource.DIRECT,
        use_llm: bool = True,
    ) -> Optional[Episode]:
        """
        Full processing pipeline: score → decide → create episode.
        
        Returns an Episode if content should be encoded, None if discarded.
        """
        # Score salience
        if use_llm:
            salience = self.score(content, context, source)
        else:
            salience = self.score_fast(content, context, source)

        # Decide encoding level
        decision = self.should_encode(salience)

        if decision == "discard":
            logger.debug(f"Discarded (salience={salience.composite:.2f}): {content[:60]}...")
            return None

        # Create episode
        if decision == "summary" and len(content) > 200:
            # Summarize for medium-salience content
            content = content[:200] + "..."

        episode = Episode(
            content=content,
            salience=salience,
            source=source,
        )

        logger.debug(
            f"Encoded ({decision}, salience={salience.composite:.2f}): "
            f"{content[:60]}..."
        )
        return episode
