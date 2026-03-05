# NEXUS Memory: Developer Design Document

This document outlines the core internal process flows of the NEXUS memory system, specifically tracing the path of data from user input through ingestion, retrieval, and background consolidation. It is intended to help new engineers navigate the codebase and understand the Dual-Process architecture.

---

## 1. System 1: Real-Time Ingestion Flow (`memory.encode()`)

When a user sends a message to the agent, the agent must quickly store the message before generating a response. This process is strictly heuristic and takes milliseconds.

### Trace Path
1. **Agent Wrapper:** The integrating agent calls `nexus.encode(text="User prefers Python.")`.
2. **`core.py` (NEXUS.encode):** The orchestrator receives the raw text.
3. **`attention_gate.py` (Salience Filter):** The text is passed through the `AttentionGate`. A fast heuristic rule-engine checks for keywords, error markers, or explicit commands to score the text across 5 dimensions (surprise, relevance, emotional, novelty, utility). If the composite score falls below `discard_threshold`, the message is silently dropped to prevent DB bloat.
4. **`vector_store.py` (Embedding):** The text is embedded into a 384-dimensional vector using the local `SentenceTransformer` backend.
5. **`episode_buffer.py` (Storage):** The text, vector, and salience score are wrapped in an `Episode` dataclass (from `models.py`) and appended to the SQLite `episodes` table.
6. **Auto-Trigger Check:** `core.py` checks if the number of unconsolidated episodes exceeds the threshold. If yes, it launches a background thread for consolidation (System 2); otherwise, it returns control immediately to the agent.

---

## 2. Real-Time Retrieval Flow (`memory.recall()`)

When the agent needs context to answer a user's question, it queries NEXUS. This flow prioritizes algorithmic speed over LLM reasoning.

### Trace Path
1. **Agent Wrapper:** The integrating agent calls `nexus.recall(query="What language does the user prefer?", top_k=5)`.
2. **`core.py` (NEXUS.recall):** The orchestrator passes the query string to the `RetrievalEngine`.
3. **`vector_store.py` (Flat Search):** The query is embedded. The `VectorStore` (FAISS or NumPy) performs a pure cosine-similarity search against *both* unconsolidated `Episodes` and fully consolidated `Memories`. It returns a widened pool of top candidates (e.g., $K \times 3$).
4. **`retrieval.py` (Heuristic Scoring):** The `RetrievalEngine` loops over every candidate and computes a modified score:
   $$Q(v) = \beta_1(Cosine) + \beta_2(Temporal\_Decay) + \beta_3(Retrieval\_Strength) + \beta_4(Salience)$$
5. **`palace.py` (Contextual Priming):** The engine checks the current `WorkingMemory`. If any concepts currently in Working Memory share a `Room` with a candidate memory, that candidate receives a score boost (simulating cognitive priming).
6. **The Testing Effect:** The candidate list is sorted by the final adjusted score $Q(v)$ and truncated to `top_k`. For the winning `Memories`, the engine calls `memory.reinforce()`, which increments their `access_count` and bumps their base strength so they are easier to retrieve next time.
7. **Return:** The `Memory` strings are returned to `core.py`, which hands them back to the integrating agent to inject into the LLM system prompt.

---

## 3. System 2: Asynchronous Consolidation Flow (`memory.consolidate()`)

This is the "slow", analytical, LLM-driven heart of NEXUS. It runs in an isolated background thread and is responsible for maintaining the `SemanticPalace`.

### Trace Path
1. **Trigger:** Called manually or automatically triggered by `encode()`.
2. **`core.py` (NEXUS.consolidate):** Spawns a thread calling the `ConsolidationEngine`.
3. **`consolidation.py` (The Engine):** Executes up to 8 distinct processes sequentially, depending on the requested depth (`LIGHT` vs `FULL`).

#### The 8 Consolidation Processes:
*   **Process 1: Chunking (`_chunk_episodes`)**
    Extracts raw `Episodes` from the database and prompts the LLM (`llm_interface.py`) to convert colloquial dialogue into atomic, context-independent factual strings.
*   **Process 2: Conflict Resolution (`_detect_conflicts`)**
    Searches the `SemanticPalace` for existing memories that contradict the new chunks (e.g., "User likes Java" vs "User now likes Python"). Uses the LLM to write a superseding memory, marking the old one as `SUPERSEDED`.
*   **Process 3: Integration/Placement (`_integrate_memories`)**
    Takes the resolved facts, embeds them via `VectorStore`, and searches the `SemanticPalace` for the semantically closest `Room`. If no room is close enough, it creates a new one. It instantiates the final `Memory` dataclass and links it to the room.
*   **Process 4: Reflection (`_generate_reflections`)**
    Looks at groups of related active memories and asks the LLM to deduce higher-level insights (e.g., *Fact 1 + Fact 2 → Insight A*).
*   **Process 5: Cross-Referencing (`_cross_reference`)**
    Computes pairwise vector similarities between all `Room` centroids in the palace. If two disparate rooms are semantically close, it writes a `TypedEdge` connecting them, enabling multi-hop associative retrieval later.
*   **Process 6: Skill Extraction (`_extract_skills`)**
    *Experimental.* Scans for repeated procedural actions or coding patterns and saves them to the Skill Vault as reusable tools.
*   **Process 7: Spaced Repetition (`_spaced_repetition_review`)**
    Finds memories that are due for review based on the Ebbinghaus forgetting curve. Uses the LLM to test the agent on these facts; success reinforces the memory, failure accelerates decay.
*   **Process 8: Defragmentation & Forgetting (`_defragment_and_forget`)**
    Garbage collection. Merges small, highly similar `Rooms`. Scans for `Memories` whose strength has decayed below the `strength_hard_threshold` and replaces them with a highly compressed `MemoryTombstone` to save space.

### Error Handling
Each of the 8 processes is individually wrapped in a `try/except` block. If the local LLM times out during Process 4 (Reflection), Process 5 (Cross-Referencing) will still execute perfectly over the geometry of the graph.

---

## 4. Key Data Models (`nexus/models.py`)

*   `Episode`: Short-term, timestamped vector. Has a `salience` score.
*   `Memory`: Long-term fact. Has `strength`, `confidence`, and belongs to a `room_id`.
*   `Room`: A categorical cluster inside the `SemanticPalace`. Has a pure mathematical `centroid_embedding`.
*   `ConfidenceLevel`: A meta-memory output score (0.0 - 1.0) indicating how well the system thinks it knows a given topic.
