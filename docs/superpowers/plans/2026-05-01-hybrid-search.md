# Hybrid Search (FTS5 + RRF) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SQLite FTS5 keyword search alongside vector search, merge candidates via Reciprocal Rank Fusion, and validate with a benchmark comparing hybrid vs. vector-only retrieval.

**Architecture:** `FTSIndex` wraps SQLite FTS5 (porter+ascii tokenizer, WAL mode) and is injected into `RetrievalEngine`. On recall, both searches run in parallel; `_rrf_merge` fuses the ranked lists (k=60). Existing `_score_memory` pipeline runs unchanged on the merged pool.

**Tech Stack:** Python 3.9+, sqlite3 (stdlib), sentence-transformers, pytest

**Spec:** `docs/superpowers/specs/2026-05-01-hybrid-search-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `smriti_memcore/fts_index.py` | Create | SQLite FTS5 wrapper — keyword candidate generation |
| `smriti_memcore/retrieval.py` | Modify | Add `fts_index` param, `_rrf_merge()`, update `retrieve()` |
| `smriti_memcore/core.py` | Modify | Instantiate `FTSIndex`, wire into encode/forget/close |
| `tests/test_fts_index.py` | Create | Unit tests: rebuild idempotency, exact-term recall, RRF correctness |
| `benchmarks/__init__.py` | Create | Package marker |
| `benchmarks/bench_hybrid_search.py` | Create | Benchmark: hybrid vs. vector-only, comparison table |

---

## Task 1: FTSIndex — schema, add, remove, rebuild, close

**Files:**
- Create: `smriti_memcore/fts_index.py`
- Create: `tests/test_fts_index.py`

- [ ] **Step 1: Write rebuild idempotency tests**

```python
# tests/test_fts_index.py
import pytest
from smriti_memcore.fts_index import FTSIndex


@pytest.fixture
def fts():
    index = FTSIndex(":memory:")
    yield index
    index.close()


class TestFTSRebuildIdempotency:
    def test_rebuild_twice_same_row_count(self, fts, make_memory):
        memories = [make_memory(f"topic {i}") for i in range(5)]
        fts.rebuild(memories)
        assert fts.needs_rebuild(5) is False
        fts.rebuild(memories)
        assert fts.needs_rebuild(5) is False

    def test_search_identical_after_rebuild(self, fts, make_memory):
        memories = [make_memory(f"topic {i}") for i in range(5)]
        fts.rebuild(memories)
        results1 = [r[0] for r in fts.search("topic")]
        fts.rebuild(memories)
        results2 = [r[0] for r in fts.search("topic")]
        assert results1 == results2
```

- [ ] **Step 2: Run to confirm ImportError**

```bash
cd /Users/shivamtyagi/PycharmProjects/nexus-memory
python3 -m pytest tests/test_fts_index.py -v
```
Expected: `ImportError: cannot import name 'FTSIndex'`

- [ ] **Step 3: Implement FTSIndex**

```python
# smriti_memcore/fts_index.py
from __future__ import annotations

import logging
import os
import sqlite3
from typing import List, Tuple

from smriti_memcore.models import Memory

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories USING fts5(
    memory_id UNINDEXED,
    content,
    tokenize = 'porter ascii'
);
"""


class FTSIndex:
    def __init__(self, storage_path: str):
        if storage_path == ":memory:":
            fts_db_path = ":memory:"
        else:
            fts_db_path = os.path.join(storage_path, "fts.db")
        self._path = fts_db_path
        self._conn = self._open(fts_db_path)

    def _open(self, path: str) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_CREATE_TABLE)
            conn.commit()
            return conn
        except sqlite3.DatabaseError:
            if path != ":memory:":
                logger.warning(f"Corrupt fts.db at {path} — deleting and rebuilding")
                try:
                    os.remove(path)
                except OSError:
                    pass
            conn = sqlite3.connect(path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_CREATE_TABLE)
            conn.commit()
            return conn

    def needs_rebuild(self, active_count: int) -> bool:
        row = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0] != active_count

    def add(self, memory_id: str, content: str) -> None:
        self._conn.execute(
            "INSERT INTO memories(memory_id, content) VALUES (?, ?)",
            (memory_id, content),
        )
        self._conn.commit()

    def remove(self, memory_id: str) -> None:
        self._conn.execute(
            "DELETE FROM memories WHERE memory_id = ?", (memory_id,)
        )
        self._conn.commit()

    def rebuild(self, memories: List[Memory]) -> None:
        self._conn.execute("DELETE FROM memories")
        self._conn.executemany(
            "INSERT INTO memories(memory_id, content) VALUES (?, ?)",
            [(m.id, m.content) for m in memories],
        )
        self._conn.commit()

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        rows = self._conn.execute(
            "SELECT memory_id, bm25(memories) FROM memories "
            "WHERE memories MATCH ? ORDER BY bm25(memories) LIMIT ?",
            (query, top_k),
        ).fetchall()
        return [(row[0], row[1]) for row in rows]

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
```

Note on `:memory:` special-casing: `FTSIndex(":memory:")` skips the `{storage_path}/fts.db` path construction and connects directly. WAL mode is a no-op on in-memory databases (SQLite accepts it silently).

- [ ] **Step 4: Run rebuild tests**

```bash
python3 -m pytest tests/test_fts_index.py::TestFTSRebuildIdempotency -v
```
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add smriti_memcore/fts_index.py tests/test_fts_index.py
git commit -m "feat: add FTSIndex — SQLite FTS5 wrapper with rebuild idempotency"
```

---

## Task 2: FTSIndex — exact-term recall test

**Files:**
- Modify: `tests/test_fts_index.py`

- [ ] **Step 1: Add exact-term recall tests**

```python
# Append to tests/test_fts_index.py

class TestFTSExactTermSearch:
    def test_exact_term_appears_in_results(self, fts, make_memory):
        target = make_memory("ticket YEP-293 auth regression in login flow")
        noise = [make_memory(f"unrelated memory about topic {i}") for i in range(10)]
        fts.rebuild([target] + noise)
        results = fts.search("YEP-293", top_k=5)
        ids = [r[0] for r in results]
        assert target.id in ids

    def test_no_results_when_term_absent(self, fts, make_memory):
        memories = [make_memory(f"topic {i} no special terms") for i in range(10)]
        fts.rebuild(memories)
        results = fts.search("YEP-293", top_k=5)
        assert results == []
```

- [ ] **Step 2: Run tests**

```bash
python3 -m pytest tests/test_fts_index.py::TestFTSExactTermSearch -v
```
Expected: 2 PASSED

- [ ] **Step 3: Commit**

```bash
git add tests/test_fts_index.py
git commit -m "test: add FTS exact-term recall tests"
```

---

## Task 3: RetrievalEngine — `_rrf_merge` tests and implementation

**Files:**
- Modify: `smriti_memcore/retrieval.py`
- Modify: `tests/test_fts_index.py`

- [ ] **Step 1: Write RRF merge tests**

```python
# Append to tests/test_fts_index.py

class TestRRFMerge:
    def _engine(self, palace, vector_store, working_memory):
        from smriti_memcore.models import SmritiConfig
        from smriti_memcore.retrieval import RetrievalEngine
        return RetrievalEngine(
            palace=palace, working_memory=working_memory,
            vector_store=vector_store, config=SmritiConfig(),
        )

    def test_fts_only_memory_in_merged_pool(
        self, palace, vector_store, working_memory, make_memory
    ):
        engine = self._engine(palace, vector_store, working_memory)
        fts_only = make_memory("fts-only memory")
        vector_mem = make_memory("vector memory")
        palace.place_memory(vector_mem)

        merged = engine._rrf_merge(
            vector_candidates=[vector_mem],
            fts_results=[(fts_only.id, -1.0)],
            pool_size=10,
        )
        assert fts_only.id in merged

    def test_both_list_memory_scores_higher(
        self, palace, vector_store, working_memory, make_memory
    ):
        engine = self._engine(palace, vector_store, working_memory)
        both = make_memory("in both lists")
        fts_only = make_memory("in fts only")
        palace.place_memory(both)

        merged = engine._rrf_merge(
            vector_candidates=[both],
            fts_results=[(both.id, -1.0), (fts_only.id, -2.0)],
            pool_size=10,
        )
        assert merged.index(both.id) < merged.index(fts_only.id)
```

- [ ] **Step 2: Run to confirm AttributeError**

```bash
python3 -m pytest tests/test_fts_index.py::TestRRFMerge -v
```
Expected: `AttributeError: 'RetrievalEngine' object has no attribute '_rrf_merge'`

- [ ] **Step 3: Add `fts_index` param and `_rrf_merge` to RetrievalEngine**

In `smriti_memcore/retrieval.py`, update the top-of-file imports:

```python
# Add defaultdict to the existing collections import; add TYPE_CHECKING
from collections import defaultdict, deque
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from smriti_memcore.fts_index import FTSIndex
```

Update `RetrievalEngine.__init__` to accept `fts_index` (existing lines 35–48):

```python
def __init__(
    self,
    palace: SemanticPalace,
    working_memory: WorkingMemory,
    vector_store: VectorStore,
    config: SmritiConfig,
    fts_index: Optional["FTSIndex"] = None,
):
    self.palace = palace
    self.working_memory = working_memory
    self.vector_store = vector_store
    self.config = config
    self.fts_index = fts_index
    self.retrieval_log: deque = deque(maxlen=1000)
```

Add `_rrf_merge` after `_score_memory` (around line 162):

```python
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
    return sorted(scores, key=lambda mid: scores[mid], reverse=True)[:pool_size]
```

- [ ] **Step 4: Run RRF tests**

```bash
python3 -m pytest tests/test_fts_index.py::TestRRFMerge -v
```
Expected: 2 PASSED

- [ ] **Step 5: Run full test suite — no regressions**

```bash
python3 -m pytest tests/ -v --tb=short
```
Expected: all existing tests pass

- [ ] **Step 6: Commit**

```bash
git add smriti_memcore/retrieval.py tests/test_fts_index.py
git commit -m "feat: add _rrf_merge to RetrievalEngine with optional fts_index parameter"
```

---

## Task 4: RetrievalEngine — update `retrieve()` with hybrid step

**Files:**
- Modify: `smriti_memcore/retrieval.py`

The current `retrieve()` step 1 (lines 68–73):
```python
candidates = self.palace.search(query, top_k=top_k * 2, max_hops=max_hops)
if not candidates:
    logger.debug(f"No memories found for query: {query[:60]}...")
    return []
```

- [ ] **Step 1: Replace step 1 with hybrid merge logic**

```python
# Step 1a — vector search (wider pool than before: top_k*3)
vector_candidates = self.palace.search(query, top_k=top_k * 3, max_hops=max_hops)

if self.fts_index is not None:
    # Step 1b — FTS keyword search
    try:
        fts_results = self.fts_index.search(query, top_k=top_k * 3)
    except Exception:
        logger.warning("FTS search failed — falling back to vector-only retrieval")
        fts_results = []

    # Step 1c — RRF merge → ordered list of IDs
    merged_ids = self._rrf_merge(
        vector_candidates, fts_results, pool_size=top_k * 2
    )

    # Step 1d — reconstruct Memory objects, fetching FTS-only IDs from palace
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
```

- [ ] **Step 2: Run retrieval and FTS tests**

```bash
python3 -m pytest tests/test_retrieval.py tests/test_fts_index.py -v
```
Expected: all PASSED

- [ ] **Step 3: Run full test suite**

```bash
python3 -m pytest tests/ -v --tb=short
```
Expected: all existing tests pass

- [ ] **Step 4: Commit**

```bash
git add smriti_memcore/retrieval.py
git commit -m "feat: update RetrievalEngine.retrieve() with FTS hybrid search and RRF merge"
```

---

## Task 5: `core.py` — wire FTSIndex into SMRITI

**Files:**
- Modify: `smriti_memcore/core.py`
- Modify: `tests/test_fts_index.py`

- [ ] **Step 1: Write end-to-end exact-term recall test**

```python
# Append to tests/test_fts_index.py

class TestHybridSearchEndToEnd:
    def test_exact_term_found_in_hybrid_results(self, tmp_dir):
        from smriti_memcore.core import SMRITI
        from smriti_memcore.models import MemorySource, SmritiConfig

        config = SmritiConfig(storage_path=tmp_dir)
        smriti = SMRITI(config=config)

        smriti.encode(
            "ticket YEP-293 causes auth regression in login flow",
            source=MemorySource.USER_STATED,
            use_llm=False,
        )
        for i in range(10):
            smriti.encode(
                f"unrelated topic about thing number {i}",
                source=MemorySource.USER_STATED,
                use_llm=False,
            )

        results = smriti.recall("YEP-293", top_k=5)
        assert any("YEP-293" in m.content for m in results), \
            "Hybrid search must find YEP-293 in top-5"
        smriti.close()
```

- [ ] **Step 2: Run test — expect failure (FTSIndex not yet wired)**

```bash
python3 -m pytest tests/test_fts_index.py::TestHybridSearchEndToEnd -v
```
Expected: AssertionError (YEP-293 memory not found — vector similarity alone is insufficient)

- [ ] **Step 3: Wire FTSIndex into `core.py`**

Add import at the top of `smriti_memcore/core.py` (with other local imports):
```python
from smriti_memcore.fts_index import FTSIndex
```

In `SMRITI.__init__`, after `self.palace = SemanticPalace(...)` (around line 79), add:
```python
# FTS index — expendable derived index, self-heals via rebuild
self.fts_index = FTSIndex(self.config.storage_path)
active_memories = [
    m for m in self.palace.memories.values()
    if m.status == MemoryStatus.ACTIVE
]
if self.fts_index.needs_rebuild(len(active_memories)):
    self.fts_index.rebuild(active_memories)
```

Update `RetrievalEngine` instantiation (around line 92) to pass `fts_index`:
```python
self.retrieval_engine = RetrievalEngine(
    palace=self.palace,
    working_memory=self.working_memory,
    vector_store=self.vector_store,
    config=self.config,
    fts_index=self.fts_index,
)
```

In `encode()`, after `room = self.palace.place_memory(memory)` (around line 163):
```python
try:
    self.fts_index.add(memory.id, content)
except Exception as e:
    logger.warning(f"FTS add failed for {memory.id}: {e}")
```

In `forget()`, after `memory.status = MemoryStatus.ARCHIVED` (around line 277):
```python
self.fts_index.remove(memory_id)
```

In `close()`, before `self._closed = True` (around line 386):
```python
self.fts_index.close()
```

In `_atexit_save()`, extend the existing try/except:
```python
def _atexit_save(self):
    if not self._closed:
        try:
            self.save()
        except Exception:
            pass
        try:
            self.fts_index.close()
        except Exception:
            pass
```

- [ ] **Step 4: Run end-to-end test**

```bash
python3 -m pytest tests/test_fts_index.py::TestHybridSearchEndToEnd -v
```
Expected: PASSED

- [ ] **Step 5: Run full test suite**

```bash
python3 -m pytest tests/ -v --tb=short
```
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add smriti_memcore/core.py tests/test_fts_index.py
git commit -m "feat: wire FTSIndex into SMRITI core — encode/forget/close/startup"
```

---

## Task 6: Benchmark harness — hybrid vs. vector-only

**Files:**
- Create: `benchmarks/__init__.py`
- Create: `benchmarks/bench_hybrid_search.py`

The benchmark encodes 20 controlled memories (5 exact-term, 5 semantic, 10 noise), then runs 10 queries (5 exact-term, 5 semantic) in both vector-only and hybrid modes. Metrics: Hit@5, Hit@1, MRR, avg latency. Results print as a comparison table.

Vector-only mode is achieved by setting `smriti.retrieval_engine.fts_index = None` temporarily on the same instance and palace — no separate data loading needed.

- [ ] **Step 1: Create benchmark files**

```python
# benchmarks/__init__.py
```

```python
# benchmarks/bench_hybrid_search.py
"""
Benchmark: hybrid (FTS5+RRF) vs. vector-only retrieval.

Usage:
    python3 -m benchmarks.bench_hybrid_search

No LLM required — encodes memories with use_llm=False.
"""
from __future__ import annotations

import tempfile
import time
from typing import List, Tuple

from smriti_memcore.core import SMRITI
from smriti_memcore.models import MemorySource, SmritiConfig

# ── Dataset ───────────────────────────────────────────────────────────────────

_MEMORIES = [
    "ticket YEP-293 causes auth regression in login flow",
    "version 2.3.1 introduced breaking API change in payments module",
    "smriti_encode function crashes when content exceeds max_length",
    "PR #456 merges the refactor of the episode buffer subsystem",
    "JIRA-881 database deadlock on concurrent palace writes",
    "Python is preferred for data science and machine learning work",
    "Test-driven development improves long-term code quality and design",
    "Retrieval pipeline scores memories on cosine similarity and recency",
    "Working memory holds 7 plus or minus 2 items at any time",
    "Obsidian vault provides a human-readable mirror of the semantic palace",
] + [f"background noise topic number {i} unrelated to everything" for i in range(10)]

# (query, expected_substring, category)
_QUERIES: List[Tuple[str, str, str]] = [
    ("YEP-293",              "YEP-293",          "exact"),
    ("version 2.3.1",        "2.3.1",            "exact"),
    ("smriti_encode crash",  "smriti_encode",    "exact"),
    ("PR 456 episode buffer","PR #456",          "exact"),
    ("JIRA-881 deadlock",    "JIRA-881",         "exact"),
    ("best language for data science",    "data science",     "semantic"),
    ("how to write tests well",           "Test-driven",      "semantic"),
    ("how does retrieval scoring work",   "cosine similarity","semantic"),
    ("how many slots in working memory",  "7 plus",           "semantic"),
    ("palace sync to obsidian",           "Obsidian",         "semantic"),
]

# ── Runner ────────────────────────────────────────────────────────────────────

def _run_queries(smriti: SMRITI) -> dict:
    hits5 = hits1 = 0
    mrr = 0.0
    latencies = []
    by_cat: dict = {"exact": [], "semantic": []}

    for query, expected, cat in _QUERIES:
        t0 = time.perf_counter()
        results = smriti.recall(query, top_k=5)
        latencies.append((time.perf_counter() - t0) * 1000)

        rank = next(
            (i + 1 for i, m in enumerate(results)
             if expected.lower() in m.content.lower()),
            None,
        )
        hits5 += rank is not None
        hits1 += rank == 1
        mrr += (1.0 / rank) if rank else 0.0
        if cat in by_cat:
            by_cat[cat].append(rank is not None)

    n = len(_QUERIES)
    return {
        "hit@5":       hits5 / n,
        "hit@1":       hits1 / n,
        "mrr":         mrr / n,
        "avg_ms":      sum(latencies) / n,
        "exact_hit@5": sum(by_cat["exact"]) / len(by_cat["exact"]),
        "sem_hit@5":   sum(by_cat["semantic"]) / len(by_cat["semantic"]),
    }


def _print_table(vo: dict, hy: dict):
    rows = [
        ("Hit@5 — all queries",   "hit@5",       ".0%"),
        ("Hit@1 — all queries",   "hit@1",       ".0%"),
        ("MRR  — all queries",    "mrr",         ".3f"),
        ("Hit@5 — exact-term",    "exact_hit@5", ".0%"),
        ("Hit@5 — semantic",      "sem_hit@5",   ".0%"),
        ("Avg latency (ms)",      "avg_ms",      ".1f"),
    ]
    W = 62
    print("\n" + "=" * W)
    print(f"  {'Metric':<32} {'Vector-only':>12} {'Hybrid FTS':>12}")
    print("=" * W)
    for label, key, fmt in rows:
        print(f"  {label:<32} {vo[key]:>12{fmt}} {hy[key]:>12{fmt}}")
    print("=" * W + "\n")


def main():
    with tempfile.TemporaryDirectory(prefix="smriti_bench_") as tmp:
        config = SmritiConfig(storage_path=tmp)
        smriti = SMRITI(config=config)

        print(f"Encoding {len(_MEMORIES)} memories (no LLM)...")
        encoded = sum(
            1 for c in _MEMORIES
            if smriti.encode(c, source=MemorySource.USER_STATED, use_llm=False)
        )
        print(f"  {encoded}/{len(_MEMORIES)} encoded\n")

        # Vector-only: disable FTS on the shared palace
        saved_fts = smriti.retrieval_engine.fts_index
        smriti.retrieval_engine.fts_index = None
        print("Running vector-only queries...")
        vo = _run_queries(smriti)

        # Hybrid: restore FTS
        smriti.retrieval_engine.fts_index = saved_fts
        print("Running hybrid queries...")
        hy = _run_queries(smriti)

        smriti.close()

    _print_table(vo, hy)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run benchmark**

```bash
cd /Users/shivamtyagi/PycharmProjects/nexus-memory
python3 -m benchmarks.bench_hybrid_search
```

Expected output shape (exact numbers vary):
```
Encoding 20 memories (no LLM)...
  20/20 encoded

Running vector-only queries...
Running hybrid queries...

==============================================================
  Metric                           Vector-only   Hybrid FTS
==============================================================
  Hit@5 — all queries                     50%          80%
  Hit@1 — all queries                     30%          60%
  MRR  — all queries                    0.350        0.620
  Hit@5 — exact-term                      20%          80%
  Hit@5 — semantic                        80%          80%
  Avg latency (ms)                        X.X          X.X
==============================================================
```

Hybrid should score higher on exact-term queries; semantic scores should be similar.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/__init__.py benchmarks/bench_hybrid_search.py
git commit -m "feat: add hybrid search benchmark — vector-only vs FTS+RRF comparison"
```

---

## Verification

After all tasks are complete:

```bash
# Full test suite
python3 -m pytest tests/ -v

# Benchmark
python3 -m benchmarks.bench_hybrid_search
```

The benchmark is the primary acceptance gate: hybrid Hit@5 on exact-term queries must exceed vector-only. If it doesn't, check that `fts_index.rebuild()` was called after all `encode()` calls (it is — `encode()` calls `fts_index.add()` incrementally).
