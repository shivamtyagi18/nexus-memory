"""
Benchmark Harness — Main runner for comparing memory systems.
Loads dataset, feeds conversations to each system, evaluates answers.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from benchmarks.data_loaders import BenchmarkDataset, load_locomo, load_longmemeval
from benchmarks.metrics import compute_all_metrics, aggregate_metrics
from baselines.base import BaseMemorySystem

logger = logging.getLogger(__name__)


class BenchmarkHarness:
    """
    Runs benchmark evaluations across multiple memory systems.
    
    Pipeline for each system:
    1. Ingest all conversation messages in order
    2. Ask each question
    3. Evaluate answers
    4. Aggregate and compare
    """

    def __init__(
        self,
        systems: List[BaseMemorySystem],
        dataset: BenchmarkDataset,
        llm=None,
        use_llm_judge: bool = False,
        output_dir: str = "results",
    ):
        self.systems = systems
        self.dataset = dataset
        self.llm = llm
        self.use_llm_judge = use_llm_judge
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        """Run the full benchmark across all systems."""
        logger.info(f"Starting benchmark: {self.dataset.name}")
        logger.info(f"Systems: {[s.name for s in self.systems]}")
        logger.info(f"Sessions: {len(self.dataset.sessions)}, Questions: {len(self.dataset.questions)}")

        all_results = {}
        start = time.time()

        for system in self.systems:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {system.name}")
            logger.info(f"{'='*60}")

            result = self._evaluate_system(system)
            all_results[system.name] = result

            logger.info(f"{system.name} complete: {result['aggregate']}")

        total_time = time.time() - start

        # Build comparison report
        report = {
            "dataset": self.dataset.name,
            "timestamp": datetime.now().isoformat(),
            "total_time_seconds": total_time,
            "systems": all_results,
            "comparison": self._build_comparison(all_results),
        }

        # Save results
        output_file = os.path.join(
            self.output_dir,
            f"benchmark_{self.dataset.name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"\nResults saved to {output_file}")
        return report

    def _evaluate_system(self, system: BaseMemorySystem) -> Dict[str, Any]:
        """Evaluate a single memory system."""
        # Reset system
        system.reset()

        # Phase 1: Ingest all conversations
        ingest_start = time.time()
        total_messages = 0

        for session in self.dataset.sessions:
            for msg in session.messages:
                system.ingest(
                    message=msg["content"],
                    role=msg["role"],
                    metadata={"session_id": session.session_id},
                )
                total_messages += 1

        ingest_time = time.time() - ingest_start
        logger.info(f"  Ingested {total_messages} messages in {ingest_time:.1f}s")

        # Phase 2: Answer questions
        question_results = []
        for i, question in enumerate(self.dataset.questions):
            logger.info(f"  Question {i+1}/{len(self.dataset.questions)}: {question.question[:50]}...")

            try:
                response = system.query(question.question)

                metrics = compute_all_metrics(
                    question=question.question,
                    reference=question.reference_answer,
                    prediction=response.answer,
                    latency_ms=response.latency_ms,
                    tokens_used=response.tokens_used,
                    llm=self.llm,
                    use_llm_judge=self.use_llm_judge,
                )

                result = {
                    "question_id": question.question_id,
                    "question": question.question,
                    "reference": question.reference_answer,
                    "prediction": response.answer,
                    "category": question.category,
                    "confidence": response.confidence,
                    "metrics": metrics,
                }
                question_results.append(result)

                logger.info(
                    f"    F1={metrics['f1']:.2f}, "
                    f"latency={metrics['latency_ms']:.0f}ms"
                )
            except Exception as e:
                logger.error(f"  Error on question {question.question_id}: {e}")
                question_results.append({
                    "question_id": question.question_id,
                    "error": str(e),
                    "metrics": {"f1": 0, "latency_ms": 0},
                })

        # Aggregate metrics
        all_metrics = [r["metrics"] for r in question_results if "error" not in r]
        aggregate = aggregate_metrics(all_metrics)

        # Category breakdown
        category_metrics = {}
        for result in question_results:
            if "error" in result:
                continue
            cat = result["category"]
            if cat not in category_metrics:
                category_metrics[cat] = []
            category_metrics[cat].append(result["metrics"])

        category_aggregate = {
            cat: aggregate_metrics(metrics)
            for cat, metrics in category_metrics.items()
        }

        return {
            "ingest_time_seconds": ingest_time,
            "messages_ingested": total_messages,
            "questions": question_results,
            "aggregate": aggregate,
            "by_category": category_aggregate,
            "system_stats": system.get_stats(),
        }

    def _build_comparison(self, all_results: Dict) -> Dict:
        """Build a comparison table across all systems."""
        comparison = {}

        for name, result in all_results.items():
            agg = result.get("aggregate", {})
            comparison[name] = {
                "f1_mean": agg.get("f1_mean", 0),
                "latency_mean_ms": agg.get("latency_ms_mean", 0),
                "tokens_mean": agg.get("tokens_used_mean", 0),
            }

            if "judge_overall_mean" in agg:
                comparison[name]["judge_score"] = agg["judge_overall_mean"]

        return comparison


def print_comparison_table(report: Dict):
    """Pretty-print comparison results."""
    comparison = report.get("comparison", {})
    if not comparison:
        print("No comparison data.")
        return

    print(f"\n{'='*70}")
    print(f"BENCHMARK: {report.get('dataset', 'Unknown')}")
    print(f"{'='*70}")
    print(f"{'System':<20} {'F1':>8} {'Latency':>10} {'Tokens':>8}", end="")

    has_judge = any("judge_score" in v for v in comparison.values())
    if has_judge:
        print(f" {'Judge':>8}", end="")
    print()
    print("-" * 70)

    # Sort by F1 score
    sorted_systems = sorted(comparison.items(), key=lambda x: x[1].get("f1_mean", 0), reverse=True)

    for name, metrics in sorted_systems:
        f1 = metrics.get("f1_mean", 0)
        latency = metrics.get("latency_mean_ms", 0)
        tokens = metrics.get("tokens_mean", 0)
        print(f"{name:<20} {f1:>8.3f} {latency:>8.0f}ms {tokens:>8.0f}", end="")
        if has_judge:
            judge = metrics.get("judge_score", 0)
            print(f" {judge:>8.3f}", end="")
        print()

    print(f"{'='*70}")
