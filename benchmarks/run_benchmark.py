#!/usr/bin/env python3
"""
NEXUS v2 Benchmark Runner
Run: python3.11 benchmarks/run_benchmark.py --dataset locomo --output results/
"""

import argparse
import json
import logging
import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexus.models import NexusConfig
from nexus.llm_interface import LLMInterface
from nexus.vector_store import VectorStore

from baselines.naive_rag import NaiveRAG
from baselines.full_context import FullContext
from baselines.mem0_style import Mem0Style
from baselines.memgpt_style import MemGPTStyle
from baselines.nexus_adapter import NexusAdapter

from benchmarks.data_loaders import load_locomo, load_longmemeval
from benchmarks.harness import BenchmarkHarness, print_comparison_table


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def create_systems(llm: LLMInterface, selected: list) -> list:
    """Create the requested memory systems."""
    systems = []

    for name in selected:
        if name == "nexus":
            config = NexusConfig(storage_path="/tmp/nexus_bench")
            systems.append(NexusAdapter(llm, config))

        elif name == "naive_rag":
            vs = VectorStore()
            systems.append(NaiveRAG(llm, vs))

        elif name == "full_context":
            systems.append(FullContext(llm))

        elif name == "mem0":
            vs = VectorStore()
            systems.append(Mem0Style(llm, vs))

        elif name == "memgpt":
            vs = VectorStore()
            systems.append(MemGPTStyle(llm, vs))

    return systems


def main():
    parser = argparse.ArgumentParser(description="NEXUS v2 Benchmark Runner")
    parser.add_argument(
        "--systems", type=str,
        default="nexus,naive_rag,full_context,mem0,memgpt",
        help="Comma-separated list of systems to benchmark",
    )
    parser.add_argument(
        "--dataset", type=str, default="locomo",
        choices=["locomo", "longmemeval"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--max-conversations", type=int, default=1,
        help="Max conversations to load (LoCoMo)",
    )
    parser.add_argument(
        "--max-questions", type=int, default=20,
        help="Max questions per conversation/total",
    )
    parser.add_argument(
        "--output", type=str, default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--llm-judge", action="store_true",
        help="Use LLM-as-judge for evaluation (requires API key)",
    )
    parser.add_argument(
        "--ollama-url", type=str, default="http://localhost:11434",
        help="Ollama API URL",
    )
    parser.add_argument(
        "--model", type=str, default="mistral",
        help="Ollama model to use for system backbone",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger("benchmark")

    # Initialize LLM
    llm = LLMInterface(
        ollama_base_url=args.ollama_url,
        default_model=args.model,
    )

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    if args.dataset == "locomo":
        dataset = load_locomo(
            max_conversations=args.max_conversations,
            max_questions_per_conv=args.max_questions,
        )
    else:
        dataset = load_longmemeval(
            max_questions=args.max_questions,
        )
    logger.info(f"Loaded {len(dataset.sessions)} sessions, {len(dataset.questions)} questions")

    # Create systems
    selected = [s.strip() for s in args.systems.split(",")]
    systems = create_systems(llm, selected)
    logger.info(f"Systems: {[s.name for s in systems]}")

    # Run benchmark
    harness = BenchmarkHarness(
        systems=systems,
        dataset=dataset,
        llm=llm,
        use_llm_judge=args.llm_judge,
        output_dir=args.output,
    )

    report = harness.run()

    # Print results
    print_comparison_table(report)

    # Cleanup NEXUS temp data
    if os.path.exists("/tmp/nexus_bench"):
        shutil.rmtree("/tmp/nexus_bench", ignore_errors=True)

    print(f"\nDetailed results saved to: {args.output}/")


if __name__ == "__main__":
    main()
