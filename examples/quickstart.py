"""
NEXUS Memory — Quickstart Example

Prerequisites:
    1. pip install nexus-memory
    2. Ollama running locally:  ollama serve
    3. Pull a model:           ollama pull mistral
"""

from nexus import NEXUS, NexusConfig

# ── 1. Initialize with custom config (all fields are optional) ──────────
config = NexusConfig(
    storage_path="./my_agent_memory",   # where memories persist on disk
    llm_model="mistral",                # any Ollama model name
    working_memory_slots=7,             # Miller's Law: 7 ± 2
)
agent_memory = NEXUS(config=config)

# ── 2. Encode memories ──────────────────────────────────────────────────
agent_memory.encode("User's name is Alice and she is a software engineer.")
agent_memory.encode("Alice prefers Python for backend and Rust for systems work.")
agent_memory.encode("Alice is mass-allergic to shellfish — critical health info.",
                    context="medical")
agent_memory.encode("Had a casual chat about the weather today.")  # low-salience → may be filtered

# ── 3. Recall by natural-language query ─────────────────────────────────
results = agent_memory.recall("What programming languages does Alice use?")
for mem in results:
    print(f"  [{mem.strength:.2f}] {mem.content}")

# ── 4. Check meta-memory confidence ────────────────────────────────────
confidence = agent_memory.how_well_do_i_know("Alice's dietary restrictions")
print(f"\nConfidence on dietary restrictions: {confidence.overall:.0%}")
print(f"  Coverage: {confidence.coverage:.0%}  |  Freshness: {confidence.freshness:.0%}")

# ── 5. Run background consolidation (spaced repetition, defrag, etc.) ──
agent_memory.consolidate()

# ── 6. Persist everything to disk ──────────────────────────────────────
agent_memory.save()
print(f"\nSystem stats: {agent_memory.stats()}")
print("Memory saved to ./my_agent_memory/")
