"""
Benchmark Metrics — Evaluation scoring for memory system comparison.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from nexus.llm_interface import LLMInterface


def f1_score(prediction: str, reference: str) -> float:
    """Token-level F1 score between prediction and reference."""
    pred_tokens = set(_normalize(prediction).split())
    ref_tokens = set(_normalize(reference).split())

    if not ref_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    common = pred_tokens & ref_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    """Exact match after normalization."""
    return 1.0 if _normalize(prediction) == _normalize(reference) else 0.0


def llm_judge_score(
    question: str,
    reference: str,
    prediction: str,
    llm: LLMInterface,
    judge_model: str = "gemini-flash",
) -> Dict[str, float]:
    """
    LLM-as-judge evaluation.
    Uses a strong model (Gemini Flash / GPT-4o-mini) to score answer quality.
    Returns: correctness, completeness, relevance, overall (all 0-1).
    """
    result = llm.judge_answer(question, reference, prediction)

    # Ensure all scores are valid floats
    return {
        "correctness": _safe_float(result.get("correctness", 0)),
        "completeness": _safe_float(result.get("completeness", 0)),
        "relevance": _safe_float(result.get("relevance", 0)),
        "overall": _safe_float(result.get("overall", 0)),
    }


def compute_all_metrics(
    question: str,
    reference: str,
    prediction: str,
    latency_ms: float,
    tokens_used: int,
    llm: Optional[LLMInterface] = None,
    use_llm_judge: bool = False,
) -> Dict[str, Any]:
    """Compute all metrics for a single QA pair."""
    metrics = {
        "f1": f1_score(prediction, reference),
        "exact_match": exact_match(prediction, reference),
        "latency_ms": latency_ms,
        "tokens_used": tokens_used,
        "answer_length": len(prediction),
    }

    if use_llm_judge and llm:
        judge = llm_judge_score(question, reference, prediction, llm)
        metrics["judge_correctness"] = judge["correctness"]
        metrics["judge_completeness"] = judge["completeness"]
        metrics["judge_relevance"] = judge["relevance"]
        metrics["judge_overall"] = judge["overall"]

    return metrics


def aggregate_metrics(all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics across all questions."""
    if not all_metrics:
        return {}

    keys = all_metrics[0].keys()
    aggregated = {}

    for key in keys:
        values = [m[key] for m in all_metrics if isinstance(m.get(key), (int, float))]
        if values:
            aggregated[f"{key}_mean"] = sum(values) / len(values)
            aggregated[f"{key}_min"] = min(values)
            aggregated[f"{key}_max"] = max(values)
            if key == "latency_ms" and len(values) >= 20:
                sorted_vals = sorted(values)
                aggregated["latency_p95"] = sorted_vals[int(len(sorted_vals) * 0.95)]

    aggregated["total_questions"] = len(all_metrics)
    return aggregated


def _normalize(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def _safe_float(v, default: float = 0.0) -> float:
    """Safely convert to float."""
    try:
        return float(v)
    except (ValueError, TypeError):
        return default
