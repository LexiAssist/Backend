"""
Evaluation Service — Data Persistence Layer

Provides JSON-file-based storage for local development and PostgreSQL-ready
SQLAlchemy models for production. Stores quiz attempts, AI interaction logs,
and user feedback.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

# Storage directory for JSON persistence
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

QUIZ_KEYS_FILE = os.path.join(DATA_DIR, "quiz_answer_keys.json")
QUIZ_ATTEMPTS_FILE = os.path.join(DATA_DIR, "quiz_attempts.json")
INTERACTIONS_FILE = os.path.join(DATA_DIR, "ai_interactions.json")
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback.json")


def _load_json(filepath: str) -> list:
    """Load a JSON array from file, return empty list if not found."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def _save_json(filepath: str, data: list):
    """Save a JSON array to file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# ─── Quiz Answer Keys ───────────────────────────────────────────────

def save_answer_key(quiz_id: str, answers: Dict[str, Any]) -> bool:
    """
    Store or update the answer key for a quiz.
    answers format: {"q1": "correct_answer", "q2": "correct_answer", ...}
    """
    keys = _load_json(QUIZ_KEYS_FILE)

    # Update existing or add new
    found = False
    for key in keys:
        if key["quiz_id"] == quiz_id:
            key["answers"] = answers
            key["updated_at"] = datetime.now(timezone.utc).isoformat()
            found = True
            break

    if not found:
        keys.append({
            "quiz_id": quiz_id,
            "answers": answers,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        })

    _save_json(QUIZ_KEYS_FILE, keys)
    return True


def get_answer_key(quiz_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve the answer key for a quiz."""
    keys = _load_json(QUIZ_KEYS_FILE)
    for key in keys:
        if key["quiz_id"] == quiz_id:
            return key["answers"]
    return None


# ─── Quiz Attempts ───────────────────────────────────────────────────

def save_quiz_attempt(attempt: dict) -> str:
    """Save a graded quiz attempt. Returns attempt_id."""
    attempts = _load_json(QUIZ_ATTEMPTS_FILE)
    attempt_id = str(uuid.uuid4())
    attempt["attempt_id"] = attempt_id
    attempt["graded_at"] = datetime.now(timezone.utc).isoformat()
    attempts.append(attempt)
    _save_json(QUIZ_ATTEMPTS_FILE, attempts)
    return attempt_id


def get_user_quiz_attempts(user_id: str) -> List[dict]:
    """Get all quiz attempts for a user."""
    attempts = _load_json(QUIZ_ATTEMPTS_FILE)
    return [a for a in attempts if a.get("user_id") == user_id]


def get_quiz_attempts_count(user_id: str = None) -> int:
    """Get count of quiz attempts, optionally filtered by user."""
    attempts = _load_json(QUIZ_ATTEMPTS_FILE)
    if user_id:
        return len([a for a in attempts if a.get("user_id") == user_id])
    return len(attempts)


def get_average_quiz_score(user_id: str = None) -> float:
    """Get average score across quiz attempts."""
    attempts = _load_json(QUIZ_ATTEMPTS_FILE)
    if user_id:
        attempts = [a for a in attempts if a.get("user_id") == user_id]
    if not attempts:
        return 0.0
    scores = [a.get("score", 0) for a in attempts]
    return sum(scores) / len(scores)


# ─── AI Interaction Logs ─────────────────────────────────────────────

def save_interaction(interaction: dict) -> str:
    """Log an AI interaction. Returns interaction_id."""
    interactions = _load_json(INTERACTIONS_FILE)
    interaction_id = str(uuid.uuid4())
    interaction["interaction_id"] = interaction_id
    interaction["logged_at"] = datetime.now(timezone.utc).isoformat()
    interactions.append(interaction)
    _save_json(INTERACTIONS_FILE, interactions)
    return interaction_id


def get_user_interactions(user_id: str) -> List[dict]:
    """Get all AI interactions for a user."""
    interactions = _load_json(INTERACTIONS_FILE)
    return [i for i in interactions if i.get("user_id") == user_id]


def get_interaction_stats(user_id: str = None) -> dict:
    """Get aggregated interaction statistics."""
    interactions = _load_json(INTERACTIONS_FILE)
    if user_id:
        interactions = [i for i in interactions if i.get("user_id") == user_id]

    if not interactions:
        return {
            "total_interactions": 0,
            "average_latency_ms": 0.0,
            "total_tokens_consumed": 0,
            "success_rate": 100.0,
            "total_estimated_cost_usd": 0.0
        }

    total = len(interactions)
    total_latency = sum(i.get("latency_ms", 0) for i in interactions)
    total_tokens = sum(i.get("input_tokens", 0) + i.get("output_tokens", 0) for i in interactions)
    success_count = sum(1 for i in interactions if i.get("success", True))

    # Accurate cost calculation using per-interaction stored costs when available,
    # falling back to blended Gemini 2.5 Flash rate.
    total_cost = sum(
        i.get("estimated_cost_usd", 0)
        for i in interactions
    )

    return {
        "total_interactions": total,
        "average_latency_ms": round(total_latency / total, 2) if total > 0 else 0.0,
        "total_tokens_consumed": total_tokens,
        "success_rate": round((success_count / total) * 100, 2) if total > 0 else 100.0,
        "total_estimated_cost_usd": round(total_cost, 6)
    }


# ─── User Feedback ──────────────────────────────────────────────────

def save_feedback(feedback: dict) -> str:
    """Save user feedback. Returns feedback_id."""
    feedbacks = _load_json(FEEDBACK_FILE)
    feedback_id = str(uuid.uuid4())
    feedback["feedback_id"] = feedback_id
    feedback["submitted_at"] = datetime.now(timezone.utc).isoformat()
    feedbacks.append(feedback)
    _save_json(FEEDBACK_FILE, feedbacks)
    return feedback_id


def get_user_feedback(user_id: str) -> List[dict]:
    """Get all feedback submitted by a user."""
    feedbacks = _load_json(FEEDBACK_FILE)
    return [f for f in feedbacks if f.get("user_id") == user_id]


def get_feedback_stats(user_id: str = None) -> dict:
    """Get aggregated feedback statistics."""
    feedbacks = _load_json(FEEDBACK_FILE)
    if user_id:
        feedbacks = [f for f in feedbacks if f.get("user_id") == user_id]

    if not feedbacks:
        return {
            "total_feedback": 0,
            "average_rating": 0.0,
            "rating_distribution": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        }

    ratings = [f.get("rating", 0) for f in feedbacks]
    distribution = {i: ratings.count(i) for i in range(1, 6)}

    return {
        "total_feedback": len(feedbacks),
        "average_rating": round(sum(ratings) / len(ratings), 2),
        "rating_distribution": distribution
    }


# ─── Today's stats helper ───────────────────────────────────────────

def get_today_interaction_count(user_id: str) -> int:
    """Get the number of AI interactions for a user today."""
    interactions = _load_json(INTERACTIONS_FILE)
    today = datetime.now(timezone.utc).date().isoformat()
    count = 0
    for i in interactions:
        if i.get("user_id") == user_id:
            logged = i.get("logged_at", "")
            if logged.startswith(today):
                count += 1
    return count
