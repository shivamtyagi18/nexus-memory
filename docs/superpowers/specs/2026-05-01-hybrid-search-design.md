# Hybrid Search: SQLite FTS5 + Reciprocal Rank Fusion

**Date:** 2026-05-01
**Status:** Approved for implementation
**Scope:** `smriti_memcore` — retrieval layer only

---

## Problem

Smriti's retrieval scores memories on semantic similarity (cosine) plus recency, strength, and salience. This works well for conceptual queries but fails on exact-term queries — ticket IDs, version numbers, function names, proper nouns — where the embedding similarity to the query may be low even though the memory contains the exact term.

The fix is hybrid search: run keyword (BM25) and vector searches in parallel, merge their candidate pools before scoring, then apply the existing multi-factor scoring pipeline unchanged.

---

## Approach: Parallel Pools + RRF

Vector search and FTS5 each independently produce a ranked candidate list. Reciprocal Rank Fusion (RRF) merges them into a single ordered pool. The existing `_score_memory()` pipeline runs on this merged pool.

RRF formula: `score(d) = Σ 1 / (k + rank(d))` across all lists, `k = 60` (standard).

RRF is parameter-free — no α/β weights to tune. A memory appearing only in the FTS list still contributes `1/(60+1)` to its RRF score and enters the merged pool, guaranteeing exact-match memories reach the scoring stage.

---

## Components

### New: `smriti_memcore/fts_index.py` — `FTSIndex`

Self-contained SQLite FTS5 wrapper. Single responsibility: keyword candidate generation.

**Public interface:**

```python
class FTSIndex:
    def __init__(self, storage_path: str): ...
    def needs_rebuild(self, active_count: int) -> bool: ...
    def add(self, memory_id: str, content: str) -> None: ...
    def remove(self, memory_id: str) -> None: ...
    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]: ...
    def rebuild(self, memories: List[Memory]) -> None: ...
    def close(self) -> None: ...
```

**Storage:** `{storage_path}/fts.db` — SQLite file, same directory as `palace.json`.

**Schema:**
```sql
CREATE VIRTUAL TABLE IF NOT EXISTS memories USING fts5(
    memory_id UNINDEXED,
    content,
    tokenize = 'porter ascii'
);
```

`porter ascii` applies Porter stemming and lowercasing — "consolidating" matches "consolidate". Note: the tokenizer splits on non-alphanumeric characters, so "YEP-293" is stored as two tokens (`yep`, `293`). Queries for "YEP-293" go through the same tokenization and match correctly, but querying "YEP" alone will also match. This is acceptable behaviour for the use case.

**Startup (two-phase, orchestrated by `core.py`):**

`FTSIndex.__init__` handles only connection setup:
1. Check `os.path.exists(fts_db_path)` — if the file is absent, note it (SQLite will create an empty file on connect; the table will be missing, `needs_rebuild` returns True).
2. Open SQLite connection with WAL mode (`PRAGMA journal_mode=WAL`).
3. Execute `CREATE TABLE IF NOT EXISTS` — if this raises `sqlite3.DatabaseError` (corrupt file), delete `fts.db`, reconnect, and re-execute the CREATE.
4. `__init__` does NOT call `rebuild()` — it has no access to the memory list.

`FTSIndex.needs_rebuild(active_count)` returns `True` if the row count in the FTS table differs from `active_count`.

`core.py` is responsible for calling `rebuild()` after construction (see below).

**`rebuild(memories)` contract:** `memories` must be a list of ACTIVE memories only (`status == MemoryStatus.ACTIVE`). Executes DELETE-all + bulk INSERT. Fast at current scale (<10ms for 74 memories).

**WAL mode:** `PRAGMA journal_mode=WAL` — reads do not block writes.

**`fts.db` is expendable:** derived index, not source of truth. A corrupt or deleted `fts.db` is always self-healing via rebuild.

**Query sanitization:** `FTSIndex.search()` does not validate or sanitize the query string. Malformed FTS5 expressions raise `sqlite3.OperationalError`, caught by the fallback in `RetrievalEngine.retrieve()` and treated as an empty FTS result.

**`forget()` + startup interaction:** `fts_index.remove()` is called immediately on forget. If the process exits before `palace.json` persists the status change, the next startup will rebuild from the palace (which still has the memory as ACTIVE), re-adding the FTS entry. This is self-correcting — on the restart after persistence, the row count will again match and no rebuild is needed.

---

### Modified: `smriti_memcore/retrieval.py` — `RetrievalEngine`

`FTSIndex` injected at construction via `__init__`. `retrieve()` gains a pre-scoring merge step between existing steps 1 and 2.

**New method: `_rrf_merge`**

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
    return sorted(scores, key=lambda id: scores[id], reverse=True)[:pool_size]
```

Called as `_rrf_merge(vector_candidates, fts_results, pool_size=top_k * 2)` — `top_k * 2` gives the scoring pipeline a pool twice the final target size, matching the previous vector-only behaviour.

**Updated `retrieve()` flow:**

```
Step 1a  palace.search(query, top_k=top_k*3)       → vector_candidates: List[Memory]
Step 1b  fts_index.search(query, top_k=top_k*3)    → fts_results: List[(id, score)]
            [on exception: fts_results = [], log warning, continue]
Step 1c  _rrf_merge(vector_candidates, fts_results,
                    pool_size=top_k*2)              → merged_ids: List[str]
Step 1d  build {id: Memory} map from vector_candidates;
         for each id in merged_ids not present in map, call palace.get_memory(id)
           [None returns silently dropped — memory archived between search and fetch]
           [exceptions from palace.get_memory() propagate — not caught here]
         reconstruct combined_pool in merged_ids order
Steps 2–5 unchanged: _score_memory(), testing effect, effort bonus, WorkingMemory
```

Each source uses `top_k * 3` candidates so the union pool is deep enough before RRF narrows it to `top_k * 2` for scoring.

---

### Modified: `smriti_memcore/core.py` — `SMRITI`

- `__init__()`:
  ```python
  self.fts_index = FTSIndex(config.storage_path)
  active_memories = [m for m in self.palace.memories.values()
                     if m.status == MemoryStatus.ACTIVE]
  if self.fts_index.needs_rebuild(len(active_memories)):
      self.fts_index.rebuild(active_memories)
  ```
  Pass `fts_index` to `RetrievalEngine`.

- `encode()`: call `fts_index.add(memory.id, memory.content)` after palace write; wrap in try/except — log on failure, never raise.

- `forget()`: sets `memory.status = MemoryStatus.ARCHIVED` on the in-memory object (existing behaviour); then call `fts_index.remove(memory_id)` after the status mutation.

- `close()`: call `fts_index.close()`.

- `_atexit_save()`: call `fts_index.close()` inside the existing best-effort try/except — failure is silently swallowed, same as the existing atexit pattern.

---

## Data Flow

### Encode

```
smriti.encode(text)
  → AttentionGate check
  → EpisodeBuffer.add()
  → palace.add_memory(memory)          [existing]
  → vector_store.add(memory.id, text)  [existing]
  → fts_index.add(memory.id, text)     [new, ~0.1ms, non-fatal on failure]
```

### Recall

```
smriti.recall(query)
  → RetrievalEngine.retrieve(query)
      1a. vector_candidates = palace.search(query, top_k*3)
      1b. fts_results       = fts_index.search(query, top_k*3)
            [on exception: fts_results = [], log warning, continue]
      1c. merged_ids        = _rrf_merge(vector_candidates, fts_results, pool_size=top_k*2)
      1d. build id→Memory map from vector_candidates;
          fetch missing via palace.get_memory() — drop None, let exceptions propagate;
          reconstruct combined_pool in merged_ids order
      2.  score each: composite = cosine + decay + strength + salience  [unchanged]
      3.  sort, select top_k
      4.  testing effect reinforcement                                   [unchanged]
      5.  effort bonus                                                   [unchanged]
      6.  admit to WorkingMemory                                         [unchanged]
```

### Forget

```
smriti.forget(memory_id)
  → memory.status = MemoryStatus.ARCHIVED   [existing — in-memory status mutation]
  → fts_index.remove(memory_id)             [new]
```

---

## Error Handling

FTS failures are non-fatal. If `fts_index.search()` raises any exception, `retrieve()` logs a warning and falls back to vector-only candidates — the existing pipeline runs unchanged.

```python
try:
    fts_results = self.fts_index.search(query, top_k=pool_size)
except Exception:
    logger.warning("FTS search failed — falling back to vector-only retrieval")
    fts_results = []
```

Same pattern for `fts_index.add()` in `encode()`: log and continue, never propagate.

`palace.get_memory()` returning `None` in Step 1d: silently dropped (normal TOCTOU — memory archived between FTS search and fetch). Exceptions from `palace.get_memory()` propagate normally; they indicate a broken palace state, not an FTS issue.

Corrupt `fts.db` on startup: `__init__` catches `sqlite3.DatabaseError` on table creation, deletes the file, reconnects, re-creates the table. `core.py` then calls `needs_rebuild()` → `rebuild()`. Self-healing.

---

## Testing

**1. Exact-term recall**
Encode a memory containing "YEP-293". Query "YEP-293". Assert the memory appears in results. Confirm meaningfulness by asserting it does not appear with FTS disabled (vector-only mode).

**2. RRF merge correctness** (unit test `_rrf_merge` directly, no I/O)
- A memory present only in the FTS list appears in the merged pool.
- A memory present in both lists scores higher than one present in only one.

**3. FTS rebuild idempotency** (uses SQLite `:memory:` path)
Call `rebuild()` twice with the same ACTIVE memory list. Assert row count equals input length both times. Assert `search()` returns identical results after each rebuild.

Note: the startup corrupt-file recovery path (`sqlite3.DatabaseError` → delete → rebuild) requires a real temporary file path to exercise and is left as a manual integration test.

All automated tests use SQLite `":memory:"` path — no disk I/O, no cleanup needed.

---

## Out of Scope

- Indexing memory metadata, room IDs, or salience fields — `content` only
- Exposing FTS5 operators (`NEAR`, `*`, `""`) to callers — plain token matching only; malformed queries degrade gracefully via the error-handling fallback
- Async FTS writes — synchronous is sufficient at this scale
- Any changes to `palace.py`, `vector_store.py`, `models.py`, `consolidation.py`, or `mcp_server.py`

---

## Files Changed

| File | Change |
|---|---|
| `smriti_memcore/fts_index.py` | New |
| `smriti_memcore/retrieval.py` | Add `fts_index` param, `_rrf_merge()`, update `retrieve()` |
| `smriti_memcore/core.py` | Instantiate `FTSIndex`, call `needs_rebuild`/`rebuild`, wire into encode/forget/close |
| `tests/test_fts_index.py` | New — 3 automated tests above |
