# 🧬 NEXUS v2: Neuro-Inspired EXperience-Unified System
## A Novel Memory Architecture for AI Agents — Inspired by Human Cognition

> *"The best bio-inspired AI doesn't copy biology — it understands why a biological mechanism works and implements the computational principle, adapting it to the medium. Wings don't flap; airplanes generate lift differently."*

---

## The Thesis

Current AI memory systems (MemGPT, Generative Agents, Mem0, etc.) borrow heavily from **computer science metaphors** — RAM vs. disk, databases, key-value stores. They work, but they miss the deeper principles that make human memory extraordinary.

NEXUS transplants the *computational principles* behind human memory champion techniques into AI agent memory design. Not surface-level analogies — not simulating spatial navigation or circadian rhythms — but the **underlying mechanisms** that make those techniques work, adapted for the computational medium.

---

## What Makes Human Memory Champions Different

From our [human memory research](file:///Users/shivtatva/HomeProjects/Memory/sharp_memory_techniques.md), seven principles stand out — and each maps to a gap in current AI memory systems:

| # | Human Memory Principle | Underlying Mechanism | What AI Systems Lack |
|---|----------------------|---------------------|---------------------|
| 1 | **Memory Palace** | Contextual priming + associative traversal | No structural scaffolding for multi-hop retrieval |
| 2 | **Spaced Repetition** | Utility-driven reinforcement at optimal intervals | No decay management; memories are equally permanent |
| 3 | **Retrieval Practice** | The act of recall strengthens the pathway | Retrieval is passive lookup; it doesn't modify memory |
| 4 | **Reflection → Abstraction** | Episodes distill into transferable principles | Most systems skip hierarchical abstraction |
| 5 | **Emotional Weighting** | Multi-dimensional salience scoring with learned weights | Importance scoring is shallow (1-10 scale) |
| 6 | **Chunking** | Grouping fragments into meaningful units | Memories stored as isolated individual fragments |
| 7 | **Sleep Consolidation** | Background reorganization, pruning, and strengthening | No offline memory maintenance process |

---

## Architecture Overview

```
╔══════════════════════════════════════════════════════════════════════════╗
║                     NEXUS v2 ARCHITECTURE                               ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  ┌──────────────────────────────────────────────────────────────────┐    ║
║  │                      ATTENTION GATE                              │    ║
║  │  Salience scoring (5D) + Conflict detection + Routing            │    ║
║  └─────────────────────────────┬────────────────────────────────────┘    ║
║                                │                                         ║
║  ┌─────────────────────────────▼────────────────────────────────────┐    ║
║  │              WORKING MEMORY (Capacity: 7 slots / 4 chunks)       │    ║
║  │  Priority queue + eviction logging + LLM context management      │    ║
║  │  [Ambient Monitor: proactive surfacing of relevant memories]     │    ║
║  └───────┬────────────────────────────────────────────┬─────────────┘    ║
║          │                                            │                  ║
║  ┌───────▼────────────┐    ┌──────────────────────────▼─────────────┐   ║
║  │  SEMANTIC PALACE   │    │       KNOWLEDGE GRAPH                  │   ║
║  │  (Associative      │    │       (Semantic Core)                  │   ║
║  │   Index)           │    │                                        │   ║
║  │                    │    │ Entities, relations, concepts           │   ║
║  │  Rooms = semantic  │    │ + Contradiction detection              │   ║
║  │  clusters with     │    │ + Confidence scores per fact           │   ║
║  │  contextual        │    │ [ZVec-backed vector retrieval]         │   ║
║  │  priming           │    │                                        │   ║
║  │                    │    └────────────────────────────────────────┘   ║
║  │  Hallways = typed  │                                                 ║
║  │  associative       │    ┌────────────────────────────────────────┐   ║
║  │  bridges           │    │       SKILL VAULT                     │   ║
║  │  (multi-hop        │    │       (Procedural Memory)             │   ║
║  │   reasoning)       │    │  + Preconditions/postconditions       │   ║
║  │                    │    │  + Usage statistics per skill          │   ║
║  │  Landmarks =       │    │  [Voyager-style skill library]        │   ║
║  │  retrieval entry   │    └────────────────────────────────────────┘   ║
║  │  points            │                                                 ║
║  └─────┬──────────────┘    ┌────────────────────────────────────────┐   ║
║        │                   │       META-MEMORY                     │   ║
║        │                   │       (Self-Awareness Layer)          │   ║
║        │                   │  Confidence map + knowledge gaps      │   ║
║        │                   │  Ask-vs-recall decision engine        │   ║
║        │                   └────────────────────────────────────────┘   ║
║        │                                                                 ║
║  ┌─────▼───────────────────────────────────────────────────────────────┐║
║  │                      EPISODE BUFFER                                 │║
║  │  Raw experience log + salience scores + trajectory segments         │║
║  │  + reflection annotations + source attribution                      │║
║  └─────────────────────────────────┬───────────────────────────────────┘║
║                                    │                                     ║
║  ┌─────────────────────────────────▼───────────────────────────────────┐║
║  │              CONSOLIDATION ENGINE (Event-Driven)                    │║
║  │                                                                      │║
║  │  1. Spaced Repetition     — utility-based, not time-based           │║
║  │  2. Chunking Processor    — group related fragments                 │║
║  │  3. Reflection Synthesizer — hierarchical abstraction               │║
║  │  4. Forgetting Manager     — utility + context-shift decay          │║
║  │  5. Cross-Reference Linker — discover hidden connections            │║
║  │  6. Skill Extractor        — patterns → procedures                  │║
║  │  7. Conflict Resolver      — detect + resolve contradictions        │║
║  │  8. Palace Defragmenter    — merge/prune rooms, ensure connectivity │║
║  │                                                                      │║
║  │  Triggers: buffer-full, idle-time, backlog, event-count             │║
║  │  Depth: adaptive to system load                                     │║
║  └──────────────────────────────────────────────────────────────────────┘║
║                                                                          ║
║  ┌──────────────────────────────────────────────────────────────────────┐║
║  │  SHARED MEMORY BUS — Publish/subscribe with trust + discounting     │║
║  └──────────────────────────────────────────────────────────────────────┘║
║                                                                          ║
║  ┌──────────────────────────────────────────────────────────────────────┐║
║  │  SALIENCE WEIGHT LEARNER — Usage tracking → regression → tuning     │║
║  └──────────────────────────────────────────────────────────────────────┘║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## The Nine Core Components

### 1. 🚪 Attention Gate — *"The Amygdala"*
**Inspired by:** Emotional weighting, importance scoring
**Mechanism captured:** Multi-dimensional salience scoring with learned weights

**What current systems lack:** Most systems treat all incoming information equally. Generative Agents score importance, but only on a 1-10 scale with no context.

**NEXUS approach — Multi-dimensional salience scoring:**

```python
class SalienceScore:
    surprise: float     # How unexpected? (deviation from predictions)
    relevance: float    # How related to current goals?
    emotional: float    # Positive/negative outcome intensity
    novelty: float      # How different from existing knowledge?
    utility: float      # How practically useful?
    
    # Composite score with LEARNED weights
    composite = w1*surprise + w2*relevance + w3*emotional + w4*novelty + w5*utility
```

**How it works:**
- Every incoming piece of information passes through the Attention Gate
- The Gate computes a salience score across 5 dimensions
- High-salience events are immediately written to the Episode Buffer with full detail
- Low-salience events are summarized or discarded
- The weights (w1–w5) are learned via the **Salience Weight Learner** (see Component 9)

**Additionally:** The Attention Gate runs incoming data through the **Conflict Resolver** — if a new memory contradicts an existing one, the conflict is flagged immediately rather than discovered later.

---

### 2. 🏛️ Semantic Palace — *"Contextual Priming + Multi-Hop Retrieval"*
**Inspired by:** Method of Loci (Memory Palace)
**Mechanism captured:** NOT spatial navigation (AI has no spatial cognition hardware) — but rather **contextual priming** and **associative multi-hop traversal**

> [!IMPORTANT]
> The reason the Memory Palace works for humans is that we have evolved hippocampal place cells and grid cells — specialized spatial navigation hardware. AI agents have no such hardware. A graph labeled with "rooms" is just a graph. The real value of the Memory Palace for AI is **contextual priming** (activating one topic pre-loads related concepts) and **multi-hop associative retrieval** (following typed edges to surface memories that pure vector search would miss).

**NEXUS approach — Hierarchical semantic clustering with typed associations:**

```
The Semantic Palace is a navigable graph where:
  - ROOMS = semantic clusters that provide contextual priming
            (activating a room pre-loads related concepts)
  - HALLWAYS = typed associative bridges between rooms
               (causal, temporal, analogical, compositional)
               enabling multi-hop reasoning during retrieval
  - OBJECTS = individual memories anchored to their room context
  - LANDMARKS = high-salience retrieval entry points (like a table of contents)
```

**Implementation:**

```python
class SemanticPalace:
    rooms: Dict[str, Room]           # Semantic clusters
    hallways: List[TypedEdge]        # Associative bridges with relationship types
    landmarks: List[Memory]          # High-salience retrieval entry points

class Room:
    topic: str                       # "Authentication System"
    objects: List[Memory]            # Memories belonging to this cluster
    centroid_embedding: Vector       # Room's semantic center (ZVec-stored)
    visit_count: int                 # How often this cluster is accessed
    last_visited: datetime           # Recency tracking
    health: RoomHealth               # Object count, avg strength, connectivity
    
class TypedEdge:
    source_room: Room
    target_room: Room
    relationship: str                # "causal", "temporal", "analogical", "compositional"
    strength: float                  # Edge weight (reinforced by traversal)
    
class Memory:
    content: str                     # The actual memory
    embedding: Vector                # Semantic vector (ZVec-stored)
    modality: Modality               # TEXT, CODE, IMAGE, STRUCTURED_DATA
    room: Room                       # Which cluster it belongs to
    associations: List[Memory]       # Cross-room linked memories
    strength: float                  # Retrieval strength (decays, reinforced on access)
    confidence: float                # How trustworthy is this memory?
    source: MemorySource             # DIRECT (self-observed), TOLD (user-stated), 
                                     # INFERRED (reflected), EXTERNAL (shared by another agent)
    creation_time: datetime
    last_accessed: datetime
    access_count: int
    salience: SalienceScore
    status: str                      # "active", "superseded", "decaying"
    superseded_by: Optional[str]     # If contradicted, link to replacement
```

**Why this works for AI:**
- **Contextual priming** — when a query matches Room A, memories in rooms *connected to A* are also pre-loaded as candidates. Pure vector search misses these contextual neighbors
- **Multi-hop reasoning** — following typed hallways (causal → temporal → analogical) surfaces memories that are indirectly relevant. "Optimize API speed" → follows causal hallway to "Compression reduces latency" → follows compositional hallway to "gzip middleware pattern"
- **Cluster coherence** — memories within a room mutually reinforce retrieval (just as humans find it easier to recall related items in a group)

---

### 3. 🧠 Working Memory — *"The Stage"*
**Inspired by:** Human working memory limitations (Miller's 7±2, Cowan's 4 chunks)
**Mechanism captured:** Capacity constraints force prioritization, which makes the rest of the system meaningful

> [!IMPORTANT]
> Without a capacity limit, working memory is just "the LLM context window" — which is exactly what MemGPT already does. The constraint creates the **retrieval pressure** that makes salience scoring, consolidation, and forgetting genuinely necessary.

```python
class WorkingMemory:
    MAX_SLOTS = 7            # Inspired by Miller's Law
    ACTIVE_CHUNKS = 4        # Active reasoning limit (Cowan)
    
    slots: PriorityQueue     # Ranked by relevance to current goal
    eviction_log: List       # What was pushed out (useful for consolidation)
    
    def admit(self, memory: Memory):
        if len(self.slots) >= self.MAX_SLOTS:
            evicted = self.slots.pop_lowest_priority()
            self.eviction_log.append(evicted)  # Track what was pushed out
            evicted.log_eviction_context()      # Record WHY it was evicted
        self.slots.push(memory)
    
    def get_llm_context(self) -> str:
        """What the LLM actually sees — only the top active chunks"""
        return format_for_llm(self.slots.top_k(self.ACTIVE_CHUNKS))
    
    def get_peripheral_context(self) -> str:
        """Available but not primary — slots 5–7"""
        return format_as_background(self.slots.items[self.ACTIVE_CHUNKS:])
```

**The Ambient Monitor** runs inside working memory, proactively surfacing relevant memories during context shifts:

```python
class AmbientMonitor:
    """Proactively surface memories — 'this reminds me of...'"""
    
    def on_context_change(self, new_context: Context):
        """Triggered when conversation topic shifts"""
        proactive_candidates = palace.search_all_rooms(
            new_context.embedding, 
            min_strength=HIGH_STRENGTH_THRESHOLD
        )
        for memory in proactive_candidates:
            if memory.proactive_relevance(new_context) > PROACTIVE_THRESHOLD:
                working_memory.surface_suggestion(memory)
    
    def on_pattern_match(self, current_trajectory: List[Action]):
        """Triggered when current actions resemble a past trajectory"""
        similar = episode_buffer.search_trajectories(current_trajectory)
        for trajectory in similar:
            if trajectory.outcome == 'failure':
                working_memory.surface_warning(
                    f"Similar approach failed before: {trajectory.reflection}"
                )
```

---

### 4. 💤 Consolidation Engine — *"The Sleep Cycle"*
**Inspired by:** Sleep-based memory consolidation, spaced repetition, forgetting curves
**Mechanism captured:** Background reorganization, strengthening, pruning, and abstraction

**What current systems lack:** No existing AI memory system has a background consolidation process. Memories are stored and retrieved — there's no mechanism for reorganizing, strengthening, or pruning them. This is the biggest gap.

**NEXUS uses 8 background processes, triggered adaptively (not by clock time):**

**Scheduling — Event-driven, not time-based:**

```python
class AdaptiveConsolidationScheduler:
    """AI agents don't have circadian rhythms. Trigger by cognitive load."""
    
    # Event-driven triggers
    MICRO_TRIGGER = "episode_buffer.count > 50"              # Buffer getting full
    STANDARD_TRIGGER = "new_memories_since_last > 200"       # Significant new input
    DEEP_TRIGGER = "total_unconsolidated_memories > 1000"    # Backlog threshold
    IDLE_TRIGGER = "no_user_interaction_for > 5_minutes"     # Consolidate when idle
    
    def compute_depth(self, system_load: float) -> ConsolidationDepth:
        if system_load < 0.3:
            return ConsolidationDepth.FULL     # All 8 processes
        elif system_load < 0.7:
            return ConsolidationDepth.LIGHT    # Chunking + conflict detection only
        else:
            return ConsolidationDepth.DEFER    # Too busy, schedule for later
```

#### Process 1: Spaced Repetition Scheduler (Utility-Based)

> [!NOTE]
> Human spaced repetition uses temporal intervals (1 day, 3 days, 7 days) because human forgetting is time-based. AI memory doesn't degrade temporally — it loses **relevance** as context shifts and **utility** as access patterns change. NEXUS uses utility-based decay instead.

```python
class UtilityBasedRepetition:
    def compute_strength_decay(self, memory: Memory) -> float:
        # Utility component (primary) — is it being used?
        utility_factor = memory.access_count / max(1, expected_usage_rate(memory))
        
        # Context-shift component — has the world changed around this memory?
        context_staleness = semantic_drift(
            memory.embedding, current_domain_embedding
        )
        
        # Temporal component (minor) — mild recency bias
        temporal_factor = 0.99 ** days_since_last_access(memory)
        
        return (
            0.5 * utility_factor +      # Primary: is it being used?
            0.3 * context_staleness +    # Is it still relevant?
            0.2 * temporal_factor        # Mild time-based decay
        )
    
    def schedule_review(self, memory: Memory):
        """Review intervals scale with demonstrated utility, not fixed time"""
        if memory.access_count == 0:
            memory.next_review = now() + interval(days=1)   # New memory: short leash
        else:
            # Interval expands based on strength, compressed by context shift
            base_interval = 2 ** memory.consecutive_successful_reviews
            shift_factor = 1.0 - semantic_drift(memory.embedding, current_domain)
            memory.next_review = now() + interval(days=base_interval * shift_factor)
```

#### Process 2: Chunking Processor
```
Scan recent episodes for patterns:
  - Group co-occurring memories into meaningful clusters
  - Example: 5 separate "user asked about API rate limits" episodes 
    → single chunk: "User's recurring API concern pattern"
  - Reduce N memory tokens to 1 chunk that captures the essence
  - Store chunk in the Knowledge Graph as a consolidated fact
  - Original episodes can be pruned after chunking (with back-link preserved)
```

#### Process 3: Reflection Synthesizer
```
Periodically review Episode Buffer:
  - Identify clusters of related experiences
  - Generate tiered abstractions:
    - Level 0: Raw episode ("User got error 403 on /api/data endpoint")
    - Level 1: Observation ("Authentication issues occur with this endpoint")
    - Level 2: Insight ("The API key rotation schedule causes periodic failures")
    - Level 3: Principle ("Always verify token expiry before API calls")
  - Higher-level abstractions → Knowledge Graph (long-lived)
  - Lower levels → prunable after abstraction (short-lived)
  
This captures the expert vs. novice distinction:
  Novices remember episodes; experts remember principles.
```

#### Process 4: Forgetting Manager

```python
class ForgettingManager:
    """Managed forgetting based on utility, not just time"""
    
    def evaluate_for_decay(self, memory: Memory) -> Decision:
        # Never forget
        if memory.status == 'pinned':       # User-marked as important
            return Decision.KEEP
        if memory.source == 'user_stated':  # Direct user instruction
            return Decision.KEEP
        
        # Compute composite decay score
        decay = self.compute_strength_decay(memory)
        
        if decay < HARD_THRESHOLD:
            return Decision.REMOVE          # Gracefully delete
        elif decay < SOFT_THRESHOLD:
            return Decision.ARCHIVE         # Move to cold storage (retrievable but not indexed)
        else:
            return Decision.KEEP
    
    def graceful_remove(self, memory: Memory):
        """Don't just delete — leave a tombstone"""
        tombstone = MemoryTombstone(
            summary=one_line_summary(memory),
            room=memory.room,
            was_about=memory.embedding,     # Can still be found if specifically sought
            removed_at=now(),
            reason=memory.decay_reason,
        )
        self.tombstones.append(tombstone)
```

#### Process 5: Cross-Reference Linker
```
Discover hidden connections between memories:
  - Periodically run semantic similarity scans across rooms
  - When memories in different rooms are unexpectedly similar:
    - Create a hallway (typed edge) between those rooms
    - Annotate: causal, analogical, temporal, compositional
  - This produces the "incubation effect" — recognizing patterns
    across disparate domains that weren't explicitly connected
```

#### Process 6: Skill Extractor
```
Detect repeated procedural patterns:
  - Monitor action sequences in the Episode Buffer
  - When the agent performs the same sequence 3+ times:
    - Extract as a named skill with preconditions and postconditions
    - Store as executable code in the Skill Vault (Voyager-style)
    - Track usage statistics per skill
  - Skills compose: complex skills reference simpler skills
  - Unused skills decay; frequently-used skills get priority retrieval
```

#### Process 7: Conflict Resolver *(NEW in v2)*

```python
class ConflictResolver:
    """Detect and resolve contradicting memories"""
    
    def detect_conflicts(self, new_memory: Memory):
        similar = knowledge_graph.search(new_memory.embedding, topk=20)
        for existing in similar:
            contradiction_score = compute_contradiction(new_memory, existing)
            if contradiction_score > CONFLICT_THRESHOLD:
                self.resolve(new_memory, existing)
    
    def resolve(self, newer: Memory, older: Memory):
        strategy = self.select_strategy(newer, older)
        
        if strategy == 'temporal':
            winner, loser = newer, older          # Latest info supersedes
        elif strategy == 'authority':
            winner = max(newer, older, key=lambda m: m.source_authority)
            loser = min(newer, older, key=lambda m: m.source_authority)
        elif strategy == 'frequency':
            winner = max(newer, older, key=lambda m: m.corroboration_count)
            loser = min(newer, older, key=lambda m: m.corroboration_count)
        elif strategy == 'explicit':
            self.flag_for_user_disambiguation(newer, older)
            return
        
        # Don't delete — archive with "superseded_by" link
        loser.status = 'superseded'
        loser.superseded_by = winner.id
        winner.confidence += CONFLICT_RESOLUTION_BONUS
    
    def select_strategy(self, newer, older) -> str:
        if newer.source == MemorySource.USER_STATED:
            return 'authority'          # User always wins
        if older.access_count > 10 and newer.access_count == 0:
            return 'explicit'           # Well-established vs. brand new? Ask user
        return 'temporal'               # Default: newer wins
```

> [!NOTE]
> This is where NEXUS deliberately **improves on** human cognition rather than imitating it. Humans are notoriously bad at updating old beliefs — we suffer from anchoring bias and belief perseverance. AI memory should do better by actively detecting and resolving contradictions.

#### Process 8: Palace Defragmenter *(NEW in v2)*

```python
class PalaceDefragmenter:
    """Prevent graph fragmentation as the palace grows"""
    
    def defragment(self):
        # 1. Merge semantically overlapping rooms
        for room_a, room_b in self.find_high_overlap_pairs():
            if cosine_similarity(room_a.centroid, room_b.centroid) > MERGE_THRESHOLD:
                self.merge_rooms(room_a, room_b)
        
        # 2. Remove empty or abandoned rooms
        for room in palace.rooms:
            if len(room.objects) == 0 or room.last_visited < STALE_CUTOFF:
                self.archive_room(room)
        
        # 3. Ensure connectivity — no isolated room clusters
        components = palace.find_connected_components()
        if len(components) > 1:
            self.bridge_isolated_components(components)
        
        # 4. Report health metrics
        self.palace_health = PalaceHealth(
            room_count=len(palace.rooms),
            avg_room_size=mean([len(r.objects) for r in palace.rooms]),
            connectivity=palace.clustering_coefficient(),
            avg_path_length=palace.average_shortest_path(),
        )
```

---

### 5. 🔍 Retrieval Engine — *"Active Recall"*
**Inspired by:** Retrieval practice / the testing effect
**Mechanism captured:** The act of recall strengthens the memory + "desirable difficulty"

**What current systems lack:** In every existing system, retrieval is a passive read. NEXUS makes retrieval a **write operation** — every recall simultaneously strengthens the retrieved memory.

```python
def retrieve(query: str, context: Context) -> List[Memory]:
    # 1. Navigate the Semantic Palace (contextual priming + multi-hop)
    entry_rooms = palace.find_rooms(query)
    candidates = []
    
    for room in entry_rooms:
        # Direct matches within the room
        candidates.extend(room.search_objects(query))
        
        # Multi-hop: follow hallways to connected rooms (contextual priming)
        for hallway in room.hallways:
            neighbor = hallway.target_room
            neighbor_matches = neighbor.search_objects(query, threshold=0.5)
            for match in neighbor_matches:
                match.hops = 1  # Track traversal depth
            candidates.extend(neighbor_matches)
    
    # 2. Multi-factor scoring
    for memory in candidates:
        memory.retrieval_score = (
            recency_weight * recency(memory) +
            relevance_weight * cosine_similarity(query, memory.embedding) +
            strength_weight * memory.strength +
            salience_weight * memory.salience.composite
        )
    
    # 3. THE KEY INNOVATION: Retrieval strengthens the memory
    selected = top_k(candidates, k=10)
    for memory in selected:
        memory.strength *= REINFORCEMENT_FACTOR
        memory.last_accessed = now()
        memory.access_count += 1
        memory.next_review = recalculate_interval(memory)
    
    # 4. Desirable difficulty: EFFORT-based bonus (not low-score bonus)
    for memory in selected:
        retrieval_effort = (
            getattr(memory, 'hops', 0) +           # Multi-hop = more effort
            days_since(memory.last_accessed) / 30 + # Longer gap = harder recall
            (1.0 - memory.strength)                 # Weak memories need more effort
        )
        if retrieval_effort > EFFORT_THRESHOLD:
            # Hard to find AND useful = extra reinforcement
            if memory_was_useful_in_context(memory, context):
                memory.strength *= DIFFICULTY_BONUS
    
    # 5. Log for salience weight learning
    salience_learner.record_retrieval(query, selected, context)
    
    return selected
```

> [!IMPORTANT]
> The desirable difficulty bonus is based on retrieval **effort** (hops traversed, time since last access, memory weakness), NOT on low retrieval scores. A low similarity score means the memory is poorly matched to the query — that's noise, not desirable difficulty. The key requirement: the memory must also be **useful in context**.

---

### 6. 📝 Episode Buffer — *"The Diary"*
**Inspired by:** Episodic memory, memory streams, Reflexion's self-reflection buffer

| Feature | Generative Agents | Reflexion | Mem0 | NEXUS v2 |
|---------|------------------|-----------|------|----------|
| Raw event logging | ✅ | ❌ | ✅ | ✅ |
| Importance scoring | ✅ (1-10) | ❌ | ❌ | ✅ (5D + learned) |
| Self-reflection | Limited | ✅ | ❌ | ✅ + hierarchical |
| Trajectory segments | ❌ | ❌ | ❌ | ✅ (MIRA-style) |
| Automatic chunking | ❌ | ❌ | ❌ | ✅ |
| Forgetting | ❌ | ❌ | ❌ | ✅ (utility decay) |
| Strengthening on recall | ❌ | ❌ | ❌ | ✅ |
| Conflict detection | ❌ | ❌ | ❌ | ✅ |
| Source attribution | ❌ | ❌ | Partial | ✅ |
| Multi-modal | ❌ | ❌ | ❌ | ✅ |

---

### 7. 🏗️ Knowledge Graph + Skill Vault — *"Long-Term Memory"*
**Inspired by:** Semantic + procedural memory, Voyager's skill library, CoALA taxonomy

The split long-term store:
- **Knowledge Graph:** Facts, concepts, relationships, insights, chunked summaries — all semantically indexed by ZVec. Each fact carries a confidence score and source attribution. Contradictions are tracked, not silently overwritten.
- **Skill Vault:** Proven workflows, tool chains, code recipes — executable procedural knowledge indexed by intent description. Each skill has preconditions, postconditions, and usage statistics.

Both are populated primarily by the **Consolidation Engine**, not direct writes — mirroring how human long-term memory is built through consolidation, not conscious effort.

---

### 8. 🪞 Meta-Memory — *"Knowing What You Know"* *(NEW in v2)*
**Inspired by:** Human metacognition — awareness of one's own knowledge state

**What every existing system lacks:** No AI agent can currently distinguish "I know this well" from "I have a vaguely similar vector somewhere." Without meta-memory, agents hallucinate — they generate answers about topics they have no real knowledge of.

```python
class MetaMemory:
    """The agent's awareness of its own knowledge landscape"""
    
    def confidence_map(self, topic: str) -> ConfidenceLevel:
        """How well does the agent know this topic?"""
        room = palace.find_room(topic)
        if not room:
            return ConfidenceLevel.UNKNOWN     # "I have no knowledge here"
        
        return ConfidenceLevel(
            coverage=room.object_count / expected_coverage_for_topic(topic),
            freshness=mean_recency(room.objects),
            strength=mean_strength(room.objects),
            depth=max_reflection_level(room.objects),  # L0 episodes vs L3 principles
        )
    
    def knowledge_gaps(self) -> List[str]:
        """What does the agent know it DOESN'T know?"""
        # Sourced from: failed retrievals, unresolved questions, low-confidence rooms
        return self.gap_registry
    
    def should_recall_or_ask(self, query: str) -> Decision:
        """Should the agent try to recall, or admit ignorance?"""
        conf = self.confidence_map(query)
        if conf == ConfidenceLevel.UNKNOWN or conf.coverage < 0.3:
            return Decision.ADMIT_GAP_AND_ASK
        elif conf.freshness < STALE_THRESHOLD:
            return Decision.RECALL_BUT_VERIFY   # "I think X, but let me check"
        else:
            return Decision.RECALL_CONFIDENTLY
```

**Why this matters:** Meta-memory is the difference between an agent that says *"I don't know enough about this to give you a good answer"* and one that hallucinate confidently. It enables:
- **Honest uncertainty** — "I have limited knowledge on this topic"
- **Targeted learning** — "I need to learn more about X" (drives the Automatic Curriculum)
- **Calibrated confidence** — responses include confidence levels based on actual knowledge depth

---

### 9. 📡 Supporting Systems

#### Shared Memory Bus *(Multi-Agent)*

```python
class SharedMemoryBus:
    """Cross-agent memory exchange"""
    
    def publish(self, memory: Memory, scope: Scope):
        """scope: PRIVATE | TEAM | PUBLIC"""
        if scope >= Scope.TEAM:
            self.bus.publish(memory.serialize(), scope=scope)
    
    def subscribe(self, topics: List[str]):
        for topic in topics:
            self.bus.on(topic, self._ingest_external)
    
    def _ingest_external(self, external: Memory):
        external.source = MemorySource.EXTERNAL
        external.confidence *= EXTERNAL_DISCOUNT  # Lower trust for secondhand knowledge
        attention_gate.process(external)           # Normal salience scoring applies
```

#### Salience Weight Learner

```python
class SalienceWeightLearner:
    """Learn optimal salience weights from retrieval outcomes"""
    
    def record_retrieval(self, query, selected_memories, context):
        for memory in selected_memories:
            self.log.append({
                'salience_at_encoding': memory.salience,
                'was_used': memory in context.actually_used_memories,
                'outcome_quality': context.task_success_score,
            })
    
    def record_miss(self, query, context):
        """When retrieval fails to find what was needed"""
        self.miss_log.append({
            'query': query,
            'what_was_eventually_needed': context.actual_answer_source,
        })
    
    def retrain_weights(self):
        """Adjust w1–w5 based on what predicted actual usefulness"""
        X = [[s.surprise, s.relevance, s.emotional, s.novelty, s.utility] 
             for s in self.log]
        y = [entry['outcome_quality'] for entry in self.log]
        self.weights = linear_regression(X, y)
```

---

## Data Flow Example

A complete lifecycle of a memory through NEXUS v2:

```
     User says: "Always use gzip compression for API responses over 1KB"
                                    │
              ┌─────────────────────▼─────────────────────────┐
              │ ATTENTION GATE                                 │
              │ Surprise: 0.3  Relevance: 0.9  Utility: 0.95 │
              │ Composite salience: HIGH                       │
              │ Conflict check: no contradictions found        │
              │ Meta-memory: coverage in "API" room is 0.6    │
              └─────────────────────┬─────────────────────────┘
                                    │
              ┌─────────────────────▼─────────────────────────┐
              │ WORKING MEMORY (slot 3 of 7)                   │
              │ Admitted to priority queue                     │
              │ Ambient Monitor: no proactive suggestions      │
              └─────────────────────┬─────────────────────────┘
                                    │
              ┌─────────────────────▼─────────────────────────┐
              │ EPISODE BUFFER                                 │
              │ Stored as raw episode with salience annotation│
              │ Placed in Semantic Palace room: "API Practices"│
              │ Source: USER_STATED (high authority)           │
              │ Initial strength: 1.0                         │
              └─────────────────────┬─────────────────────────┘
                                    │
                    [Episode buffer hits 50 items — consolidation triggered]
                                    │
              ┌─────────────────────▼─────────────────────────┐
              │ CONSOLIDATION ENGINE (MICRO depth)             │
              │                                                │
              │ 1. Chunking: Grouped with other API best      │
              │    practices into "API Performance Guidelines" │
              │                                                │
              │ 2. Reflection: Combined with 3 past slow-     │
              │    response episodes → insight produced:       │
              │    "Compression reduces latency ~60% for >1KB" │
              │                                                │
              │ 3. Cross-linking: Hallway created to room      │
              │    "Performance Optimization" (type: causal)   │
              │                                                │
              │ 4. Skill extraction: Pattern matched →         │
              │    add_gzip_middleware() added to Skill Vault   │
              │                                                │
              │ 5. Conflict check: No contradictions           │
              │                                                │
              │ 6. Meta-memory updated: "API" room             │
              │    coverage now 0.7, depth increased to L2     │
              └─────────────────────┬─────────────────────────┘
                                    │
              ┌─────────────────────▼─────────────────────────┐
              │ LONG-TERM MEMORY                               │
              │                                                │
              │ Knowledge Graph:                               │
              │   "API responses > 1KB should use gzip"        │
              │   confidence: 0.95 | source: USER_STATED       │
              │   → linked to "Performance Optimization"       │
              │   → linked to "API Best Practices"             │
              │                                                │
              │ Skill Vault:                                   │
              │   add_gzip_middleware()                         │
              │   precondition: response_size > 1KB             │
              │   usage_count: 0                                │
              └───────────────────────────────────────────────┘
                                    │
                    [Later — Agent encounters API task]
                                    │
              ┌─────────────────────▼─────────────────────────┐
              │ META-MEMORY CHECK                              │
              │ Query: "optimize API response time"            │
              │ Confidence: coverage=0.7, depth=L2             │
              │ Decision: RECALL_CONFIDENTLY                   │
              └─────────────────────┬─────────────────────────┘
                                    │
              ┌─────────────────────▼─────────────────────────┐
              │ RETRIEVAL ENGINE                               │
              │ → Enters "Performance" room (entry point)      │
              │ → Follows causal hallway to "API Practices"    │
              │ → Finds gzip memory + related insights (1 hop) │
              │ → Memory strength reinforced: 1.0 → 1.15      │
              │ → Effort bonus: 1 hop + moderate age = applied │
              │ → Next review interval expanded                │
              │ → Also retrieves add_gzip_middleware() skill   │
              │ → Logged for salience weight learning           │
              └───────────────────────────────────────────────┘
```

---

## Technical Implementation Blueprint

### Storage Layer

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Semantic Palace graph | NetworkX (embedded) or Neo4j-lite | Topological traversal, multi-hop queries, connected components |
| Vector embeddings | **ZVec** (in-process) | Sub-ms search, no infrastructure, scales to billions |
| Episode Buffer | SQLite + ZVec | Time-ordered events with semantic search overlay |
| Knowledge Graph facts | ZVec + JSON metadata | Semantic retrieval with confidence + source attributes |
| Skill Vault | File system + ZVec index | Executable code indexed by intent embedding |
| Meta-memory state | In-memory + periodic snapshot | Fast access for real-time confidence checks |
| Salience weight model | Lightweight regression model | Retrained periodically from usage logs |
| Shared Memory Bus | Redis Pub/Sub or ZeroMQ | Cross-agent messaging with topic-based routing |

### API Surface

```python
nexus = NEXUS(config)

# ── Core Operations ──
nexus.encode(content, context, modality='text')  # Salience scoring + palace placement
memories = nexus.recall(query, context)           # Navigate + retrieve + strengthen
confidence = nexus.how_well_do_i_know(topic)      # Meta-memory confidence check

# ── Consolidation ──
nexus.consolidate(depth='auto')     # Trigger with adaptive depth
nexus.reflect()                     # Force reflection cycle
nexus.defragment()                  # Palace maintenance

# ── Memory Management ──
nexus.pin(memory_id)                # Mark as never-forget
nexus.forget(memory_id)             # Explicit removal
nexus.resolve_conflict(mem_a, mem_b, strategy='temporal')

# ── Palace Organization ──
nexus.palace.create_room(topic)
nexus.palace.link_rooms(a, b, relationship='causal')
nexus.palace.visualize()            # Render graph topology
nexus.palace.health()               # Connectivity, fragmentation metrics

# ── Multi-Agent ──
nexus.publish(memory, scope='team')
nexus.subscribe(topics=['backend', 'security'])

# ── Inspection ──
nexus.stats()                       # Memory count, health, decay rates
nexus.knowledge_gaps()              # What the agent knows it doesn't know
nexus.eviction_history()            # What was pushed out of working memory
```

---

## How NEXUS v2 Differs From Everything Else

| Existing Systems | NEXUS v2 |
|-----------------|----------|
| Memory as **storage** | Memory as a **living system** that evolves |
| Flat vector retrieval | Multi-hop associative retrieval via typed hallways |
| No forgetting | **Utility-based** managed forgetting with tombstones |
| Retrieval is read-only | Retrieval **strengthens** memories (testing effect) |
| No background processing | 8-process **Consolidation Engine** (event-driven) |
| Static importance | 5D salience with **learned weights** (feedback loop) |
| No self-awareness | **Meta-memory**: confidence maps, knowledge gaps, ask-vs-recall |
| No contradiction handling | **Conflict Resolver** with multi-strategy resolution |
| Reactive only | **Proactive** ambient surfacing ("this reminds me of...") |
| Single agent | **Shared Memory Bus** with trust discounting |
| Text only | **Multi-modal** (text, code, images, structured data) |
| Episodes and semantics separated | Episodes **distill into** semantics through hierarchical reflection |
| No capacity constraints | **Working memory limit** (7 slots) forces meaningful prioritization |

---

## Summary

| Human Technique | Underlying Mechanism | NEXUS v2 Component |
|----------------|---------------------|-------------------|
| Memory Palace | Contextual priming + associative traversal | Semantic Palace with typed hallways |
| Spaced Repetition | Utility-driven reinforcement | Consolidation Engine (utility-based, not temporal) |
| Retrieval Practice (Testing Effect) | Recall effort strengthens pathways | Retrieval that writes (effort-based difficulty bonus) |
| Reflection | Hierarchical abstraction (episode → principle) | Reflection Synthesizer (4 levels) |
| Emotional Weighting | Multi-dimensional salience with learned weights | Attention Gate (5D) + Salience Weight Learner |
| Chunking | Fragment grouping → expanded capacity | Chunking Processor (Consolidation Engine) |
| Sleep Consolidation | Background reorganization when idle | Event-driven Consolidation Engine (8 processes) |
| Forgetting Curve | Utility-based decay (not temporal) | Forgetting Manager + graceful tombstones |
| Active Elaboration | Cross-domain connection building | Cross-Reference Linker |
| Metacognition | Knowledge of own knowledge state | Meta-Memory (confidence, gaps, ask-vs-recall) |

### The Design Principles

1. **Capture mechanisms, not metaphors** — Don't simulate spatial navigation; implement contextual priming. Don't copy temporal forgetting curves; model utility-based decay.
2. **Constraints create intelligence** — Working memory limits force prioritization. Forgetting forces consolidation. Capacity pressure makes the whole system meaningful.
3. **Improve on biology where appropriate** — Contradiction resolution should be *better* than human memory, not a copy of our anchoring biases.
4. **Every retrieval is a write** — The testing effect is the single most powerful memory principle, and no existing AI system implements it.
5. **Memory is a process, not a store** — Consolidation, reflection, chunking, and forgetting are continuous background processes that transform raw experiences into durable knowledge.

The result: an AI agent that doesn't just *store* memories — it **cultivates** them.

---

*Derived from synthesis of: [Human Memory Techniques](file:///Users/shivtatva/HomeProjects/Memory/sharp_memory_techniques.md) • [ZVec & MIRA Research](file:///Users/shivtatva/HomeProjects/Memory/zvec_and_mira_research.md) • [AI Agent Memory Landscape](file:///Users/shivtatva/HomeProjects/Memory/ai_agent_memory_systems.md) • [Expert Review](file:///Users/shivtatva/HomeProjects/Memory/nexus_v2_expert_review.md)*
