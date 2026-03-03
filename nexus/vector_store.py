"""
NEXUS v2 — Vector Store.
Embedding generation + similarity search abstraction.
Uses sentence-transformers for embeddings, NumPy for search (FAISS-optional upgrade).
"""

from __future__ import annotations

import logging
import os
import json
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class VectorEntry:
    """A stored vector with its ID and metadata."""
    id: str
    vector: np.ndarray
    metadata: Dict = field(default_factory=dict)


class VectorStore:
    """
    In-process vector storage and similarity search.
    
    Uses sentence-transformers for embedding generation and NumPy for 
    cosine similarity search. Designed as the "SQLite of vector databases" 
    — no server needed, runs entirely in-process.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        dimension: int = 384,
        storage_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.dimension = dimension
        self.storage_path = storage_path
        self._model = None
        
        # In-memory storage
        self._vectors: Dict[str, VectorEntry] = {}
        self._matrix: Optional[np.ndarray] = None  # Cache for batch search
        self._matrix_ids: List[str] = []
        self._matrix_dirty = True
        self._lock = threading.Lock()

        # Load persisted data if available
        if storage_path:
            json_path = os.path.join(storage_path, "vectors.json")
            npy_path = os.path.join(storage_path, "vectors.npy")
            if os.path.exists(json_path) and os.path.exists(npy_path):
                self._load(storage_path)
            elif os.path.exists(json_path) or os.path.exists(npy_path):
                logger.warning(f"Partial vector store files in {storage_path}, skipping load")

    @property
    def model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
                raise
        return self._model

    # ── Embedding Generation ─────────────────────────────

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32)

    # ── Storage Operations ───────────────────────────────

    def add(self, id: str, text: Optional[str] = None, vector: Optional[np.ndarray] = None,
            metadata: Optional[Dict] = None):
        """Add a vector to the store. Provide either text (auto-embed) or vector."""
        if vector is None and text is not None:
            vector = self.embed(text)
        elif vector is None:
            raise ValueError("Must provide either text or vector")

        entry = VectorEntry(
            id=id,
            vector=np.array(vector, dtype=np.float32),
            metadata=metadata or {},
        )
        with self._lock:
            self._vectors[id] = entry
            self._matrix_dirty = True

    def remove(self, id: str):
        """Remove a vector from the store."""
        with self._lock:
            if id in self._vectors:
                del self._vectors[id]
                self._matrix_dirty = True

    def get(self, id: str) -> Optional[VectorEntry]:
        """Get a specific vector entry by ID."""
        return self._vectors.get(id)

    def has(self, id: str) -> bool:
        """Check if an ID exists in the store."""
        return id in self._vectors

    @property
    def size(self) -> int:
        """Number of vectors stored."""
        return len(self._vectors)

    # ── Search ───────────────────────────────────────────

    def search(
        self,
        query: Optional[str] = None,
        query_vector: Optional[np.ndarray] = None,
        top_k: int = 10,
        min_score: float = 0.0,
        filter_ids: Optional[set] = None,
    ) -> List[Tuple[str, float]]:
        """
        Search for the most similar vectors.
        Returns list of (id, similarity_score) tuples, sorted by descending score.
        """
        if query_vector is None and query is not None:
            query_vector = self.embed(query)
        elif query_vector is None:
            raise ValueError("Must provide either query text or query_vector")

        if len(self._vectors) == 0:
            return []

        # Rebuild matrix cache if needed
        with self._lock:
            if self._matrix_dirty:
                self._rebuild_matrix()
            matrix = self._matrix
            matrix_ids = list(self._matrix_ids)

        query_vector = np.array(query_vector, dtype=np.float32)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Cosine similarity (vectors are already normalized)
        similarities = np.dot(matrix, query_vector.T).flatten()

        # Apply filters
        results = []
        for idx, score in enumerate(similarities):
            id = matrix_ids[idx]
            if score >= min_score:
                if filter_ids is None or id in filter_ids:
                    results.append((id, float(score)))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def semantic_drift(self, vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """Compute how much two vectors have drifted apart (0=identical, 1=orthogonal)."""
        a = np.array(vector_a, dtype=np.float32)
        b = np.array(vector_b, dtype=np.float32)
        similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        return 1.0 - max(0.0, float(similarity))

    # ── Persistence ──────────────────────────────────────

    def save(self, path: Optional[str] = None):
        """Save vectors to disk."""
        save_path = path or self.storage_path
        if not save_path:
            return

        os.makedirs(save_path, exist_ok=True)

        # Save vectors as numpy
        if self._vectors:
            ids = list(self._vectors.keys())
            vectors = np.array([self._vectors[id].vector for id in ids])
            metadata = {id: self._vectors[id].metadata for id in ids}

            np.save(os.path.join(save_path, "vectors.npy"), vectors)
            with open(os.path.join(save_path, "vectors.json"), "w") as f:
                json.dump({"ids": ids, "metadata": metadata}, f)

        logger.info(f"Saved {len(self._vectors)} vectors to {save_path}")

    def _load(self, path: str):
        """Load vectors from disk."""
        try:
            with open(os.path.join(path, "vectors.json"), "r") as f:
                data = json.load(f)
            vectors = np.load(os.path.join(path, "vectors.npy"))

            ids = data["ids"]
            metadata = data.get("metadata", {})

            for i, id in enumerate(ids):
                self._vectors[id] = VectorEntry(
                    id=id,
                    vector=vectors[i],
                    metadata=metadata.get(id, {}),
                )
            self._matrix_dirty = True
            logger.info(f"Loaded {len(self._vectors)} vectors from {path}")
        except Exception as e:
            logger.error(f"Failed to load vectors from {path}: {e}")

    # ── Internal ─────────────────────────────────────────

    def _rebuild_matrix(self):
        """Rebuild the search matrix from stored vectors."""
        if len(self._vectors) == 0:
            self._matrix = np.empty((0, self.dimension), dtype=np.float32)
            self._matrix_ids = []
        else:
            self._matrix_ids = list(self._vectors.keys())
            self._matrix = np.array(
                [self._vectors[id].vector for id in self._matrix_ids],
                dtype=np.float32,
            )
        self._matrix_dirty = False
