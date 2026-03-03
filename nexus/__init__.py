"""
NEXUS v2: Neuro-Inspired EXperience-Unified System
A novel memory architecture for AI agents inspired by human cognition.
"""

from nexus.core import NEXUS
from nexus.models import Memory, SalienceScore, MemorySource, Modality, NexusConfig
from nexus.metrics import NexusMetrics

__version__ = "0.1.0"
__all__ = ["NEXUS", "NexusConfig", "NexusMetrics", "Memory", "SalienceScore", "MemorySource", "Modality"]

