# ZVec & MIRA: Research Notes

> Two systems that approach "memory" from very different angles — one for machines searching vectors, the other for RL agents learning from experience.

---

## ZVec — The SQLite of Vector Databases

**What it is:** An open-source, lightweight, **in-process** vector database developed by Alibaba's Tongyi Lab. It runs entirely inside your application — no servers, no daemons, no infrastructure.

**Website:** [zvec.org](https://zvec.org)

### Core Concept

ZVec stores and searches **vector embeddings** — high-dimensional numerical representations of data (text, images, code) that capture semantic meaning. Items with similar meaning end up close together in vector space, enabling **semantic search**: finding things by *meaning* rather than keyword matching.

### Key Features

| Feature | Description |
|---------|-------------|
| **In-Process** | Runs directly in your app (Python). No external services, no network overhead. |
| **Blazing Fast** | Sub-millisecond search latency, scales to **billions** of vectors |
| **Dense & Sparse Vectors** | Supports both types plus multi-vector queries |
| **Filtered Search** | Combine semantic similarity with structured metadata filters |
| **Grouped Search** | Vector search with GROUP BY-style clauses |
| **Simple API** | `pip install zvec` — three lines of code to start |

### Built on Proxima

ZVec is powered by **Proxima**, Alibaba's production-grade vector search engine that runs:
- Taobao search (China's largest e-commerce platform)
- Alipay face payment systems
- Other systems handling billions of queries daily

### Quick Example

```python
import zvec

# 1. Create a collection
schema = zvec.CollectionSchema(
    name="example",
    vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, 4),
)
collection = zvec.create_and_open(path="./zvec_example", schema=schema)

# 2. Insert data
collection.insert(
    zvec.Doc(id="1", vectors={"embedding": [0.1, 0.2, 0.3, 0.4]})
)

# 3. Search by similarity
results = collection.query(
    vectors=zvec.VectorQuery("embedding", vector=[0.4, 0.3, 0.3, 0.1]),
    topk=10,
)
```

### Use Cases

- **RAG (Retrieval-Augmented Generation)** — Enhance LLM responses by retrieving relevant context from a knowledge base
- **Image Search** — Find visually or semantically similar images at scale
- **Code Search** — Find code snippets by describing what you want in natural language
- **AI Memory Systems** — Serve as the semantic retrieval layer for agent memory architectures
- **Edge/Mobile AI** — Resource-constrained environments where a full server-based DB is impractical

### How It Compares

| Aspect | ZVec | Traditional Vector DBs (Milvus, Pinecone, Weaviate) |
|--------|------|------------------------------------------------------|
| Deployment | In-process (embedded) | Client-server (requires infrastructure) |
| Setup | `pip install zvec` | Docker, config, cluster management |
| Latency | Sub-millisecond (no network) | Network-dependent |
| Best for | Prototyping, edge, embedded apps, notebooks | Large-scale production with multi-tenant needs |
| Analogy | SQLite | PostgreSQL |

---

## MIRA — Memory-Integrated Reinforcement Learning Agent

**What it is:** A reinforcement learning framework that uses a **persistent memory graph** to amortize LLM guidance, so an RL agent can learn efficiently without needing constant LLM supervision.

**Paper:** Published at **ICLR 2026** (conference paper) and **AAAI 2026** (student abstract)
**Authors:** Narjes Nourzad & Carlee Joe-Wong (USC & Carnegie Mellon University)
**Website:** [narjesno.github.io/MIRA](https://narjesno.github.io/MIRA/)

### The Problem MIRA Solves

RL agents face a fundamental tension:

| Challenge | Description |
|-----------|-------------|
| **Sample complexity** | RL agents need *millions* of interactions to learn, especially with sparse/delayed rewards |
| **LLM guidance helps but doesn't scale** | LLMs can provide subgoals and trajectory suggestions, but per-step LLM calls are expensive (500+ queries/run), slow, and create dependency on potentially hallucinated outputs |
| **LLMs can mislead** | Hallucinations, inconsistencies, or lack of physical grounding can generate bad advice |
| **Over-reliance** | Heavy LLM dependence overrides environment feedback, limiting genuine learning |

**MIRA's key insight:** *Build the memory once, reuse it forever.* Instead of querying an LLM at every step, amortize the LLM's knowledge into a **structured memory graph** that the agent can consult autonomously.

### How MIRA Works

```
┌─────────────────────────────────────────────────┐
│                   MIRA Pipeline                  │
├─────────────────────────────────────────────────┤
│                                                  │
│  1. MEMORY GRAPH CONSTRUCTION                   │
│     ├── Store trajectory segments                │
│     ├── Store subgoal decompositions             │
│     ├── Merge agent experience + LLM outputs     │
│     └── Graph evolves as agent improves          │
│                                                  │
│  2. UTILITY SIGNAL DERIVATION                    │
│     ├── Compare agent behavior to stored paths   │
│     ├── Weight by goal alignment                 │
│     └── Compute confidence scores                │
│                                                  │
│  3. ADVANTAGE SHAPING                            │
│     │   Ã_t = η_t · A_t + ξ_t · U_t            │
│     ├── A_t = standard policy advantage          │
│     ├── U_t = utility from memory graph          │
│     └── η_t, ξ_t = control weights              │
│                                                  │
│  4. ADAPTIVE DECAY                               │
│     ├── ξ_t decays as policy improves            │
│     ├── Agent gradually relies on own experience │
│     └── Converges to true reward function        │
│                                                  │
└─────────────────────────────────────────────────┘
```

### Key Contributions

1. **Memory-Integrated Framework** — Co-constructs a memory graph from agent experience + offline/infrequent LLM outputs
2. **Utility-Based Shaping** — Novel advantage estimation that injects graph-derived utility, compatible with any advantage-based policy gradient method (e.g., PPO)
3. **Convergence Guarantees** — Mathematical proof that decaying shaping preserves long-horizon convergence while correcting LLM inaccuracies
4. **Empirical Validation** — Achieves comparable performance to continuous LLM supervision using **~95% fewer LLM queries**

### Key Results

| Metric | Result |
|--------|--------|
| **Sample Efficiency** | Faster early learning than PPO across all tested environments |
| **Query Efficiency** | Matches LLM4Teach performance with ~95% fewer LLM queries |
| **Robustness** | Stable even when late-stage LLM outputs are degraded or unreliable |
| **Generalization** | Works across navigation, irreversible dynamics, sequential dependencies, and distractor-rich environments |

---

## ZVec vs. MIRA — Conceptual Comparison

These are very different systems that share the theme of **memory in AI**:

| Dimension | ZVec | MIRA |
|-----------|------|------|
| **Domain** | Data infrastructure / vector storage | Reinforcement learning / agent architecture |
| **Type of "memory"** | Semantic vector storage & retrieval | Experiential memory graph for learning |
| **Purpose** | Find similar items by meaning | Guide an agent's policy using past experience |
| **LLM relationship** | Powers RAG to *feed* data to LLMs | *Amortizes* LLM guidance so agents need fewer calls |
| **Users** | Developers building search, RAG, recommendation | RL researchers training agents in complex environments |
| **Key innovation** | In-process vector DB (no server needed) | Memory graph that replaces continuous LLM supervision |

### Potential Synergy

ZVec could serve as the **vector storage backend** for MIRA's memory graph — storing trajectory embeddings and enabling fast semantic retrieval of relevant past experiences. This would combine ZVec's sub-millisecond search with MIRA's structured memory-guided learning.

---

*Sources: [zvec.org](https://zvec.org), [MIRA project page (ICLR 2026)](https://narjesno.github.io/MIRA/), Alibaba Tongyi Lab documentation, web research.*
