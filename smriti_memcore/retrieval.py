"""
SMRITI v2 — Retrieval Engine.
Multi-hop associative retrieval through the Semantic Palace with
retrieval strengthening (testing effect) and effort-based desirable
difficulty bonus.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Callable, Dict, List, Literal, Optional, Tuple

import numpy as np

from smriti_memcore.fts_index import FTSIndex
from smriti_memcore.query_rewriter import QueryRewriter, ExpandResult
from smriti_memcore.snippet import SnippetExtractor, ExtractResult

from smriti_memcore.models import Memory, SmritiConfig
from smriti_memcore.palace import SemanticPalace
from smriti_memcore.working_memory import WorkingMemory
from smriti_memcore.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    Active retrieval that strengthens memories (the testing effect).
    
    The key innovation: every retrieval is a WRITE operation.
    This implements the most powerful human memory principle — retrieval
    practice — which no existing AI system uses.
    """

    def __init__(
        self,
        palace: SemanticPalace,
        working_memory: WorkingMemory,
        vector_store: VectorStore,
        config: SmritiConfig,
        fts_index: Optional[FTSIndex] = None,
        llm: Optional["LLMInterface"] = None,
    ):
        self.palace = palace
        self.working_memory = working_memory
        self.vector_store = vector_store
        self.config = config
        self.fts_index = fts_index

        self.query_rewriter = QueryRewriter(
            llm=llm,
            cache_size=config.llm_rewrite_cache_size,
            prompt_version=config.llm_rewrite_prompt_version,
        )
        self.snippet_extractor = SnippetExtractor(
            vector_store=vector_store,
            min_chars=config.snippet_min_chars,
            max_sentences=config.snippet_max_sentences,
            llm=llm,
        )

        self.retrieval_log: deque = deque(maxlen=1000)

    def retrieve(
        self,
        query: str,
        context: str = "",
        top_k: Optional[int] = None,
        max_hops: int = 1,
        rewrite: Optional[Literal["auto", "llm", "none"]] = None,
        snippet: Optional[Literal["auto", "llm", "none"]] = None,
    ) -> List[Memory]:
        """
        Full retrieval pipeline:
        1. Query rewriting — embed all variants once
        2. Vector search (palace.search with adjacency lift)
        3. FTS keyword search + RRF merge
        4. Stale-state guard + multi-factor scoring
        5. Retrieval strengthening (testing effect)
        6. Effort-based desirable difficulty bonus
        7. Snippet extraction
        8. Admit top results to Working Memory
        """
        top_k = top_k or self.config.retrieval_top_k
        start_time = time.time()

        # Resolve mode params: caller param > config default
        rewrite_mode = rewrite if rewrite is not None else self.config.rewrite_mode_default
        snippet_mode = snippet if snippet is not None else self.config.snippet_mode_default

        # 1. Query rewriting — embed all variants once
        expand_result = self.query_rewriter.expand(query, mode=rewrite_mode)
        variants = expand_result.variants
        variant_embeddings = [self.vector_store.embed(v) for v in variants]
        raw_query_embedding = variant_embeddings[0]

        # 2. Vector search (palace already widened to top-5 entry rooms internally).
        # palace.search() writes memory.relevance_score on every candidate it scored.
        vector_candidates = self.palace.search(
            variants, variant_embeddings,
            top_k=top_k * 3, max_hops=max_hops,
        )
        # Track which IDs palace scored this call — used below to clear stale
        # relevance_score on FTS-only candidates (spec §6 stale-state guard).
        palace_scored_ids = {m.id for m in vector_candidates}

        # 3. FTS keyword search (joined variants as query string)
        if self.fts_index is not None:
            joined_query = " ".join(variants)
            try:
                fts_results = self.fts_index.search(joined_query, top_k=top_k * 3)
            except Exception:
                logger.warning("FTS search failed — falling back to vector-only retrieval")
                fts_results = []

            merged_ids = self._rrf_merge(vector_candidates, fts_results, pool_size=top_k * 2)
            id_map: Dict[str, Memory] = {m.id: m for m in vector_candidates}
            candidates: List[Memory] = []
            for mid in merged_ids:
                if mid in id_map:
                    candidates.append(id_map[mid])
                else:
                    mem = self.palace.get_memory(mid)
                    if mem is not None:
                        candidates.append(mem)
        else:
            candidates = vector_candidates[: top_k * 2]

        if not candidates:
            logger.debug(f"No memories found for query: {query[:60]}...")
            return []

        # Stale-state guard: clear relevance_score on any candidate palace.search did
        # not score this call (i.e., FTS-only pulls). Memory objects in palace.memories
        # are reused across recalls, so a prior call's lifted value could leak.
        for memory in candidates:
            if memory.id not in palace_scored_ids:
                memory.relevance_score = 0.0

        # 4. Multi-factor scoring — _score_memory reads memory.relevance_score
        # (set by palace.search, or zeroed above for FTS-only candidates).
        now = datetime.now()
        for memory in candidates:
            memory.retrieval_score = self._score_memory(memory, raw_query_embedding, now)
        candidates.sort(key=lambda m: m.retrieval_score, reverse=True)
        selected = candidates[:top_k]

        # 5. Reinforcement (testing effect)
        for memory in selected:
            memory.reinforce(self.config.reinforcement_factor)
            memory.consecutive_successful_reviews += 1
            memory.next_review = self._next_review_interval(memory)

        # 6. Difficulty bonus
        for memory in selected:
            retrieval_effort = self._compute_effort(memory, now)
            if retrieval_effort > self.config.effort_threshold:
                memory.strength *= self.config.difficulty_bonus

        # 7. Snippet extraction
        snippet_fallback_any = False
        for memory in selected:
            result = self.snippet_extractor.extract(memory, variants, raw_query_embedding, mode=snippet_mode)
            if result.fallback:
                snippet_fallback_any = True

        # 8. Working memory admission
        for memory in selected:
            self.working_memory.admit(memory)

        # 9. Log
        elapsed_ms = (time.time() - start_time) * 1000
        self.retrieval_log.append({
            "query": query,
            "results": [m.id for m in selected],
            "scores": [m.retrieval_score for m in selected],
            "latency_ms": elapsed_ms,
            "timestamp": now.isoformat(),
            "rewrite_used_mode": expand_result.used_mode,
            "rewrite_fallback": expand_result.fallback,
            "snippet_fallback": snippet_fallback_any,
        })

        logger.info(
            f"Retrieved {len(selected)} memories for '{query[:40]}...' "
            f"({elapsed_ms:.0f}ms; rewrite={expand_result.used_mode}, snippet={snippet_mode})"
        )
        return selected

    def retrieve_by_id(self, memory_id: str) -> Optional[Memory]:
        """Direct retrieval by ID — still strengthens the memory."""
        memory = self.palace.get_memory(memory_id)
        if memory:
            memory.reinforce(self.config.reinforcement_factor)
        return memory

    # ── Scoring ──────────────────────────────────────────

    def _score_memory(
        self, memory: Memory, query_embedding: np.ndarray, now: datetime
    ) -> float:
        """Multi-factor retrieval scoring. Relevance = palace.search lifted score
        when available; raw cosine fallback for FTS-only candidates."""
        # Spec §3 — palace.search writes the adjacency-lifted relevance to memory.relevance_score.
        # FTS-only candidates skip palace.search and have relevance_score == 0.0 (default).
        if memory.relevance_score > 0:
            relevance = memory.relevance_score
        elif memory.embedding:
            relevance = max(0.0, float(np.dot(query_embedding, np.array(memory.embedding))))
        else:
            relevance = 0.0

        # Recency (exponential decay)
        days_since = (now - memory.last_accessed).total_seconds() / 86400
        recency = self.config.decay_rate ** days_since

        # Strength
        strength = min(memory.strength / 5.0, 1.0)

        # Salience
        salience = memory.salience.composite

        return (
            self.config.relevance_weight * relevance +
            self.config.recency_weight * recency +
            self.config.strength_weight * strength +
            self.config.salience_weight * salience
        )

    def _rrf_merge(
        self,
        vector_candidates: List[Memory],
        fts_results: List[Tuple[str, float]],
        pool_size: int,
        k: int = 60,
    ) -> List[str]:
        scores: Dict[str, float] = defaultdict(float)
        for rank, memory in enumerate(vector_candidates):
            scores[memory.id] += 1.0 / (k + rank + 1)
        for rank, (memory_id, _) in enumerate(fts_results):
            scores[memory_id] += 1.0 / (k + rank + 1)
        return sorted(scores.keys(), key=lambda mid: scores[mid], reverse=True)[:pool_size]

    def _compute_effort(self, memory: Memory, now: datetime) -> float:
        """
        Compute retrieval effort — NOT the same as low relevance.
        
        High effort = multi-hop traversal + long time since access + low strength.
        This is what "desirable difficulty" actually measures.
        """
        hop_effort = memory.hops * 0.5  # Each hop = 0.5 effort
        time_effort = min(
            (now - memory.last_accessed).total_seconds() / (30 * 86400),  # months
            1.0
        )
        weakness_effort = max(0, 1.0 - memory.strength)

        return hop_effort + time_effort + weakness_effort

    def _next_review_interval(self, memory: Memory) -> datetime:
        """Calculate next spaced repetition review time."""
        from datetime import timedelta

        # Base interval expands exponentially with successful reviews
        base_days = 2 ** memory.consecutive_successful_reviews

        # Cap at 180 days
        base_days = min(base_days, 180)

        # Context-shift factor would compress the interval
        # (for now, use strength as a proxy)
        shift_factor = max(0.5, memory.strength / 2.0)

        interval_days = base_days * shift_factor
        return datetime.now() + timedelta(days=interval_days)

    # ── Stats ────────────────────────────────────────────

    def stats(self) -> dict:
        """Retrieval engine statistics."""
        if not self.retrieval_log:
            return {"total_retrievals": 0}

        latencies = [r["latency_ms"] for r in self.retrieval_log]
        return {
            "total_retrievals": len(self.retrieval_log),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            "avg_results_per_query": sum(
                len(r["results"]) for r in self.retrieval_log
            ) / len(self.retrieval_log),
        }
