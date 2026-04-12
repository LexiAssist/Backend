"""
Evaluation Service — Data Persistence Layer (PostgreSQL)

Uses SQLAlchemy ORM with PostgreSQL for production-grade storage of
quiz attempts, AI interaction logs, and user feedback.
Falls back gracefully if the database is unavailable.
"""

import os
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Boolean,
    Text, DateTime, JSON, text, func,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# ─── Database Connection ────────────────────────────────────────────────

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://lexiassist:lexiassist_secret@localhost:5432/lexiassist"
)

engine = None
SessionLocal = None
DB_CONNECTED = False

try:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    DB_CONNECTED = True
    print("✅ Evaluation Service: PostgreSQL connected")
except Exception as e:
    print(f"⚠️  Evaluation Service: PostgreSQL not available: {e}")
    print("   Endpoints will return errors until the database is accessible.")


# ─── ORM Models ──────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class QuizAnswerKey(Base):
    __tablename__ = "eval_quiz_answer_keys"

    quiz_id = Column(String, primary_key=True)
    answers = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))


class QuizAttempt(Base):
    __tablename__ = "eval_quiz_attempts"

    attempt_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    quiz_id = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    answers = Column(JSON, nullable=True)
    score = Column(Float, nullable=False, default=0.0)
    correct_count = Column(Integer, nullable=False, default=0)
    total_questions = Column(Integer, nullable=False, default=0)
    time_taken_seconds = Column(Integer, default=0)
    feedback = Column(JSON, nullable=True)
    graded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class AIInteraction(Base):
    __tablename__ = "eval_ai_interactions"

    interaction_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    service_type = Column(String, nullable=False)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    latency_ms = Column(Integer, default=0)
    success = Column(Boolean, default=True)
    model_name = Column(String, nullable=True)
    estimated_cost_usd = Column(Float, default=0.0)
    logged_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class UserFeedback(Base):
    __tablename__ = "eval_user_feedback"

    feedback_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    interaction_id = Column(String, nullable=True)
    rating = Column(Integer, nullable=False)
    comment = Column(Text, nullable=True)
    feature_type = Column(String, nullable=False)
    submitted_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ─── Create Tables ───────────────────────────────────────────────────────

if DB_CONNECTED and engine:
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Evaluation tables created/verified")
    except Exception as e:
        print(f"⚠️  Could not create evaluation tables: {e}")


# ─── Helper ──────────────────────────────────────────────────────────────

def _get_session():
    """Get a database session. Raises RuntimeError if DB not connected."""
    if not DB_CONNECTED or not SessionLocal:
        raise RuntimeError("Database not connected")
    return SessionLocal()


# ─── Quiz Answer Keys ───────────────────────────────────────────────────

def save_answer_key(quiz_id: str, answers: Dict[str, Any]) -> bool:
    """Store or update the answer key for a quiz."""
    db = _get_session()
    try:
        existing = db.query(QuizAnswerKey).filter(QuizAnswerKey.quiz_id == quiz_id).first()
        if existing:
            existing.answers = answers
            existing.updated_at = datetime.now(timezone.utc)
        else:
            db.add(QuizAnswerKey(quiz_id=quiz_id, answers=answers))
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"Error saving answer key: {e}")
        raise
    finally:
        db.close()


def get_answer_key(quiz_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve the answer key for a quiz."""
    db = _get_session()
    try:
        key = db.query(QuizAnswerKey).filter(QuizAnswerKey.quiz_id == quiz_id).first()
        return key.answers if key else None
    finally:
        db.close()


# ─── Quiz Attempts ───────────────────────────────────────────────────────

def save_quiz_attempt(attempt: dict) -> str:
    """Save a graded quiz attempt. Returns attempt_id."""
    db = _get_session()
    try:
        attempt_id = str(uuid.uuid4())
        db_attempt = QuizAttempt(
            attempt_id=attempt_id,
            quiz_id=attempt["quiz_id"],
            user_id=attempt["user_id"],
            answers=attempt.get("answers"),
            score=attempt.get("score", 0.0),
            correct_count=attempt.get("correct_count", 0),
            total_questions=attempt.get("total_questions", 0),
            time_taken_seconds=attempt.get("time_taken_seconds", 0),
            feedback=attempt.get("feedback"),
        )
        db.add(db_attempt)
        db.commit()
        return attempt_id
    except Exception as e:
        db.rollback()
        print(f"Error saving quiz attempt: {e}")
        raise
    finally:
        db.close()


def get_user_quiz_attempts(user_id: str) -> List[dict]:
    """Get all quiz attempts for a user."""
    db = _get_session()
    try:
        attempts = db.query(QuizAttempt).filter(
            QuizAttempt.user_id == user_id
        ).order_by(QuizAttempt.graded_at.desc()).all()
        return [
            {
                "attempt_id": a.attempt_id,
                "quiz_id": a.quiz_id,
                "user_id": a.user_id,
                "score": a.score,
                "correct_count": a.correct_count,
                "total_questions": a.total_questions,
                "time_taken_seconds": a.time_taken_seconds,
                "feedback": a.feedback,
                "graded_at": a.graded_at.isoformat() if a.graded_at else None,
            }
            for a in attempts
        ]
    finally:
        db.close()


def get_quiz_attempts_count(user_id: str = None) -> int:
    """Get count of quiz attempts, optionally filtered by user."""
    db = _get_session()
    try:
        query = db.query(func.count(QuizAttempt.attempt_id))
        if user_id:
            query = query.filter(QuizAttempt.user_id == user_id)
        return query.scalar() or 0
    finally:
        db.close()


def get_average_quiz_score(user_id: str = None) -> float:
    """Get average score across quiz attempts."""
    db = _get_session()
    try:
        query = db.query(func.avg(QuizAttempt.score))
        if user_id:
            query = query.filter(QuizAttempt.user_id == user_id)
        result = query.scalar()
        return float(result) if result is not None else 0.0
    finally:
        db.close()


# ─── AI Interaction Logs ─────────────────────────────────────────────────

def save_interaction(interaction: dict) -> str:
    """Log an AI interaction. Returns interaction_id."""
    db = _get_session()
    try:
        interaction_id = str(uuid.uuid4())
        db_interaction = AIInteraction(
            interaction_id=interaction_id,
            user_id=interaction["user_id"],
            service_type=interaction["service_type"],
            input_tokens=interaction.get("input_tokens", 0),
            output_tokens=interaction.get("output_tokens", 0),
            latency_ms=interaction.get("latency_ms", 0),
            success=interaction.get("success", True),
            model_name=interaction.get("model_name"),
            estimated_cost_usd=interaction.get("estimated_cost_usd", 0.0),
        )
        db.add(db_interaction)
        db.commit()
        return interaction_id
    except Exception as e:
        db.rollback()
        print(f"Error saving interaction: {e}")
        raise
    finally:
        db.close()


def get_interaction_stats(user_id: str = None) -> dict:
    """Get aggregated interaction statistics."""
    db = _get_session()
    try:
        query = db.query(AIInteraction)
        if user_id:
            query = query.filter(AIInteraction.user_id == user_id)

        total = query.count()
        if total == 0:
            return {
                "total_interactions": 0,
                "average_latency_ms": 0.0,
                "total_tokens_consumed": 0,
                "success_rate": 100.0,
                "total_estimated_cost_usd": 0.0,
            }

        avg_latency = db.query(func.avg(AIInteraction.latency_ms))
        total_input = db.query(func.sum(AIInteraction.input_tokens))
        total_output = db.query(func.sum(AIInteraction.output_tokens))
        success_count = db.query(func.count(AIInteraction.interaction_id)).filter(
            AIInteraction.success == True
        )
        total_cost = db.query(func.sum(AIInteraction.estimated_cost_usd))

        if user_id:
            avg_latency = avg_latency.filter(AIInteraction.user_id == user_id)
            total_input = total_input.filter(AIInteraction.user_id == user_id)
            total_output = total_output.filter(AIInteraction.user_id == user_id)
            success_count = success_count.filter(AIInteraction.user_id == user_id)
            total_cost = total_cost.filter(AIInteraction.user_id == user_id)

        in_tokens = total_input.scalar() or 0
        out_tokens = total_output.scalar() or 0

        return {
            "total_interactions": total,
            "average_latency_ms": round(float(avg_latency.scalar() or 0), 2),
            "total_tokens_consumed": in_tokens + out_tokens,
            "success_rate": round((float(success_count.scalar() or 0) / total) * 100, 2),
            "total_estimated_cost_usd": round(float(total_cost.scalar() or 0), 6),
        }
    finally:
        db.close()


def get_today_interaction_count(user_id: str) -> int:
    """Get the number of AI interactions for a user today."""
    db = _get_session()
    try:
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        count = db.query(func.count(AIInteraction.interaction_id)).filter(
            AIInteraction.user_id == user_id,
            AIInteraction.logged_at >= today_start,
        ).scalar()
        return count or 0
    finally:
        db.close()


# ─── User Feedback ──────────────────────────────────────────────────────

def save_feedback(feedback: dict) -> str:
    """Save user feedback. Returns feedback_id."""
    db = _get_session()
    try:
        feedback_id = str(uuid.uuid4())
        db_feedback = UserFeedback(
            feedback_id=feedback_id,
            user_id=feedback["user_id"],
            interaction_id=feedback.get("interaction_id"),
            rating=feedback["rating"],
            comment=feedback.get("comment"),
            feature_type=feedback["feature_type"],
        )
        db.add(db_feedback)
        db.commit()
        return feedback_id
    except Exception as e:
        db.rollback()
        print(f"Error saving feedback: {e}")
        raise
    finally:
        db.close()


def get_user_feedback(user_id: str) -> List[dict]:
    """Get all feedback submitted by a user."""
    db = _get_session()
    try:
        feedbacks = db.query(UserFeedback).filter(
            UserFeedback.user_id == user_id
        ).order_by(UserFeedback.submitted_at.desc()).all()
        return [
            {
                "feedback_id": f.feedback_id,
                "user_id": f.user_id,
                "interaction_id": f.interaction_id,
                "rating": f.rating,
                "comment": f.comment,
                "feature_type": f.feature_type,
                "submitted_at": f.submitted_at.isoformat() if f.submitted_at else None,
            }
            for f in feedbacks
        ]
    finally:
        db.close()


def get_feedback_stats(user_id: str = None) -> dict:
    """Get aggregated feedback statistics."""
    db = _get_session()
    try:
        query = db.query(UserFeedback)
        if user_id:
            query = query.filter(UserFeedback.user_id == user_id)

        feedbacks = query.all()
        if not feedbacks:
            return {
                "total_feedback": 0,
                "average_rating": 0.0,
                "rating_distribution": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            }

        ratings = [f.rating for f in feedbacks]
        distribution = {i: ratings.count(i) for i in range(1, 6)}

        return {
            "total_feedback": len(feedbacks),
            "average_rating": round(sum(ratings) / len(ratings), 2),
            "rating_distribution": distribution,
        }
    finally:
        db.close()
