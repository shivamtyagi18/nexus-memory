"""
Benchmark Dataset Loaders — LoCoMo and LongMemEval.
Parses real datasets into standardized format for the benchmark harness.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# LoCoMo QA category mapping
LOCOMO_CATEGORIES = {
    1: "single_hop",
    2: "temporal",
    3: "multi_hop",
    4: "open_domain",
    5: "adversarial",
}


@dataclass
class ConversationSession:
    """A single conversation session."""
    session_id: str
    messages: List[Dict[str, str]]
    date_time: str = ""


@dataclass
class BenchmarkQuestion:
    """A question to evaluate against a memory system."""
    question_id: str
    question: str
    reference_answer: str
    category: str = ""
    evidence_sessions: List[str] = field(default_factory=list)


@dataclass
class BenchmarkDataset:
    """Full benchmark dataset."""
    name: str
    sessions: List[ConversationSession]
    questions: List[BenchmarkQuestion]
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_locomo(
    data_path: str = "data/locomo",
    max_conversations: int = 3,
    max_questions_per_conv: int = 30,
) -> BenchmarkDataset:
    """
    Load the LoCoMo dataset from the snap-research repo.

    Args:
        data_path: Path to the locomo data directory containing locomo10.json
        max_conversations: Number of conversations to load (1-10)
        max_questions_per_conv: Max QA pairs per conversation
    """
    # Try repo clone location first, then local data dir
    candidates = [
        os.path.join(data_path, "locomo10.json"),
        "/tmp/locomo_repo/data/locomo10.json",
    ]
    dataset_file = None
    for path in candidates:
        if os.path.exists(path):
            dataset_file = path
            break

    if dataset_file is None:
        logger.warning("LoCoMo dataset not found. Run: git clone https://github.com/snap-research/locomo.git /tmp/locomo_repo")
        logger.info("Falling back to synthetic dataset")
        return _generate_synthetic_locomo(data_path)

    with open(dataset_file, "r") as f:
        raw_data = json.load(f)

    all_sessions = []
    all_questions = []
    total_turns = 0

    for conv_idx, conv in enumerate(raw_data[:max_conversations]):
        sample_id = conv.get("sample_id", f"conv_{conv_idx}")
        conversation = conv.get("conversation", {})
        speaker_a = conversation.get("speaker_a", "User")
        speaker_b = conversation.get("speaker_b", "Assistant")

        # Parse sessions
        session_idx = 1
        while True:
            session_key = f"session_{session_idx}"
            date_key = f"session_{session_idx}_date_time"

            if session_key not in conversation:
                break

            session_data = conversation[session_key]
            if not session_data:
                session_idx += 1
                continue

            session_date = conversation.get(date_key, "")
            messages = []

            for turn in session_data:
                speaker = turn.get("speaker", "")
                text = turn.get("text", "")
                if not text:
                    continue

                # Map speakers to roles
                role = "user" if speaker == speaker_a else "assistant"
                messages.append({"role": role, "content": text})

            if messages:
                sess_id = f"{sample_id}_s{session_idx}"
                all_sessions.append(ConversationSession(
                    session_id=sess_id,
                    messages=messages,
                    date_time=session_date,
                ))
                total_turns += len(messages)

            session_idx += 1

        # Parse QA pairs
        qa_pairs = conv.get("qa", [])
        for qi, qa in enumerate(qa_pairs[:max_questions_per_conv]):
            category_num = qa.get("category", 0)
            category = LOCOMO_CATEGORIES.get(category_num, f"cat_{category_num}")

            # Handle answer being a string or int
            answer = qa.get("answer", "")
            if not isinstance(answer, str):
                answer = str(answer)

            evidence = qa.get("evidence", [])

            all_questions.append(BenchmarkQuestion(
                question_id=f"{sample_id}_q{qi}",
                question=qa.get("question", ""),
                reference_answer=answer,
                category=category,
                evidence_sessions=evidence,
            ))

    logger.info(
        f"LoCoMo loaded: {len(all_sessions)} sessions, "
        f"{total_turns} total turns, {len(all_questions)} questions"
    )

    return BenchmarkDataset(
        name="LoCoMo",
        sessions=all_sessions,
        questions=all_questions,
        metadata={
            "source": "locomo",
            "conversations_loaded": min(max_conversations, len(raw_data)),
            "total_turns": total_turns,
        },
    )


def load_longmemeval(
    data_path: str = "data/longmemeval",
    variant: str = "oracle",
    max_questions: int = 50,
) -> BenchmarkDataset:
    """
    Load the LongMemEval dataset.

    Args:
        data_path: Path to longmemeval data directory
        variant: 'oracle' (small, just relevant sessions) or 's' (full haystack)
        max_questions: Maximum number of questions to load
    """
    if variant == "oracle":
        filename = "longmemeval_oracle.json"
    else:
        filename = "longmemeval_s_cleaned.json"

    dataset_file = os.path.join(data_path, filename)
    if not os.path.exists(dataset_file):
        logger.warning(f"LongMemEval not found at {dataset_file}")
        return _generate_synthetic_longmemeval(data_path)

    with open(dataset_file, "r") as f:
        raw_data = json.load(f)

    all_sessions = []
    all_questions = []
    seen_session_ids = set()
    total_turns = 0

    for qi, item in enumerate(raw_data[:max_questions]):
        q_id = item.get("question_id", f"lme_{qi}")
        q_type = item.get("question_type", "unknown")

        # Parse haystack sessions (the conversation history)
        haystack_sessions = item.get("haystack_sessions", [])
        haystack_session_ids = item.get("haystack_session_ids", [])
        haystack_dates = item.get("haystack_dates", [])

        for si, session in enumerate(haystack_sessions):
            sess_id = haystack_session_ids[si] if si < len(haystack_session_ids) else f"{q_id}_s{si}"

            if sess_id in seen_session_ids:
                continue
            seen_session_ids.add(sess_id)

            messages = []
            if isinstance(session, list):
                for msg in session:
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", ""),
                    })
            elif isinstance(session, dict):
                for msg in session.get("messages", []):
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", ""),
                    })

            if messages:
                sess_date = haystack_dates[si] if si < len(haystack_dates) else ""
                all_sessions.append(ConversationSession(
                    session_id=sess_id,
                    messages=messages,
                    date_time=sess_date,
                ))
                total_turns += len(messages)

        # Parse question
        answer = item.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer)

        all_questions.append(BenchmarkQuestion(
            question_id=q_id,
            question=item.get("question", ""),
            reference_answer=answer,
            category=q_type,
            evidence_sessions=item.get("answer_session_ids", []),
        ))

    logger.info(
        f"LongMemEval loaded ({variant}): {len(all_sessions)} sessions, "
        f"{total_turns} total turns, {len(all_questions)} questions"
    )

    return BenchmarkDataset(
        name=f"LongMemEval ({variant})",
        sessions=all_sessions,
        questions=all_questions,
        metadata={
            "source": "longmemeval",
            "variant": variant,
            "total_turns": total_turns,
        },
    )


# ── Synthetic Fallbacks ──────────────────────────────────

def _generate_synthetic_locomo(data_path: str) -> BenchmarkDataset:
    """Fallback synthetic dataset if real LoCoMo isn't available."""
    os.makedirs(data_path, exist_ok=True)

    sessions = [
        ConversationSession(session_id="s1", messages=[
            {"role": "user", "content": "Hi! I'm Alex. I work as a software engineer at TechCorp."},
            {"role": "assistant", "content": "Nice to meet you, Alex! Software engineering at TechCorp sounds great."},
            {"role": "user", "content": "I mainly work with Python and React. We're building a new API platform."},
            {"role": "assistant", "content": "Python and React is a solid stack! What kind of API platform?"},
            {"role": "user", "content": "It's a REST API for our enterprise clients. We use PostgreSQL for the database."},
        ]),
        ConversationSession(session_id="s2", messages=[
            {"role": "user", "content": "We just launched our API platform last week! Got 50 enterprise signups."},
            {"role": "assistant", "content": "Congratulations on the launch! 50 enterprise signups is impressive."},
            {"role": "user", "content": "Yeah, I'm also training for a marathon. Running my first one in Boston in April."},
        ]),
    ]

    questions = [
        BenchmarkQuestion("q1", "What company does Alex work at?", "TechCorp", "single_hop"),
        BenchmarkQuestion("q2", "What database do they use?", "PostgreSQL", "single_hop"),
    ]

    return BenchmarkDataset(
        name="LoCoMo (synthetic fallback)",
        sessions=sessions,
        questions=questions,
        metadata={"source": "synthetic"},
    )


def _generate_synthetic_longmemeval(data_path: str) -> BenchmarkDataset:
    """Fallback synthetic dataset if real LongMemEval isn't available."""
    os.makedirs(data_path, exist_ok=True)

    sessions = [
        ConversationSession(session_id="lme_s1", messages=[
            {"role": "user", "content": "I love hiking in the Rocky Mountains every summer."},
            {"role": "assistant", "content": "The Rocky Mountains are beautiful in summer!"},
        ]),
    ]

    questions = [
        BenchmarkQuestion("lme_q1", "What does the user like doing in summer?", "Hiking in the Rocky Mountains", "information_extraction"),
    ]

    return BenchmarkDataset(
        name="LongMemEval (synthetic fallback)",
        sessions=sessions,
        questions=questions,
        metadata={"source": "synthetic"},
    )
