from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import os
import sys
import uvicorn

# Add current directory for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import (
    save_answer_key,
    get_answer_key,
    save_quiz_attempt,
    get_user_quiz_attempts,
    get_quiz_attempts_count,
    get_average_quiz_score,
    save_interaction,
    get_interaction_stats,
    get_today_interaction_count,
    save_feedback,
    get_user_feedback,
    get_feedback_stats,
)

# Initialize FastAPI
app = FastAPI(
    title="LexiAssist Evaluation Service",
    description="Analytics, quiz grading, and feedback collection",
    version="2.0.0"
)

# CORS
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic Models ────────────────────────────────────────────────

class QuizAnswerKey(BaseModel):
    quiz_id: str
    answers: Dict[str, Any]  # {"q1": "correct_answer", "q2": ["option_a", "option_b"], ...}

class QuizSubmission(BaseModel):
    quiz_id: str
    user_id: str
    answers: Dict[str, Any]  # {question_id: user_answer}
    time_taken_seconds: int = 0

class GradeResponse(BaseModel):
    attempt_id: str
    quiz_id: str
    user_id: str
    score: float
    total_questions: int
    correct_count: int
    correct_answers: Dict[str, Any]
    feedback: Dict[str, str]

class AIInteractionLog(BaseModel):
    user_id: str
    service_type: str = Field(..., description="e.g., 'chat', 'summary', 'quiz_gen'")
    input_tokens: int
    output_tokens: int
    latency_ms: int
    success: bool
    model_name: Optional[str] = "gemini-2.0-flash"

class FeedbackSubmission(BaseModel):
    interaction_id: Optional[str] = None
    user_id: str
    rating: int = Field(..., ge=1, le=5, description="1-5 star rating")
    comment: Optional[str] = None
    feature_type: str = Field(..., description="e.g., 'chat_response', 'quiz_hint'")

class AnalyticsResponse(BaseModel):
    total_interactions: int
    average_latency_ms: float
    total_tokens_consumed: int
    success_rate: float
    total_estimated_cost_usd: float = 0.0


# ─── Health Checks ───────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "evaluation",
        "port": 5006,
        "version": "2.0.0",
        "features": ["grading", "analytics", "feedback"]
    }

@app.get("/health")
async def health():
    from database import DB_CONNECTED
    return {
        "status": "ok",
        "storage": "postgresql" if DB_CONNECTED else "unavailable",
        "services": ["grading", "analytics", "feedback"]
    }


# ─── Quiz Answer Key Management ─────────────────────────────────────

@app.post("/answer-key")
async def set_answer_key(key: QuizAnswerKey):
    """
    Store the answer key for a quiz.
    Must be called before grading submissions for that quiz.
    
    Example body:
    {
        "quiz_id": "quiz-001",
        "answers": {"q1": "B", "q2": "True", "q3": "photosynthesis"}
    }
    """
    save_answer_key(key.quiz_id, key.answers)
    return {
        "status": "saved",
        "quiz_id": key.quiz_id,
        "questions_count": len(key.answers)
    }


@app.get("/answer-key/{quiz_id}")
async def get_quiz_answer_key(quiz_id: str):
    """Retrieve the answer key for a quiz (admin endpoint)."""
    answers = get_answer_key(quiz_id)
    if answers is None:
        raise HTTPException(status_code=404, detail=f"No answer key found for quiz '{quiz_id}'")
    return {"quiz_id": quiz_id, "answers": answers}


# ─── Quiz Grading ────────────────────────────────────────────────────

def normalize_answer(answer: Any) -> str:
    """Normalize an answer for comparison (case-insensitive, trimmed)."""
    if isinstance(answer, str):
        return answer.strip().lower()
    if isinstance(answer, (list, tuple)):
        return str(sorted([str(a).strip().lower() for a in answer]))
    return str(answer).strip().lower()


@app.post("/grade-quiz", response_model=GradeResponse)
async def grade_quiz(submission: QuizSubmission):
    """
    Auto-grades quiz submissions by comparing with stored answer keys.
    
    Flow:
    1. Look up answer key for the quiz_id
    2. Compare each submitted answer with the correct answer
    3. Calculate score and per-question feedback
    4. Persist the attempt
    5. Return grade
    """
    # Fetch answer key
    correct_answers = get_answer_key(submission.quiz_id)

    if correct_answers is None:
        raise HTTPException(
            status_code=404,
            detail=f"No answer key found for quiz '{submission.quiz_id}'. "
                   f"POST to /answer-key first to register the correct answers."
        )

    # Grade each question
    correct_count = 0
    total_questions = len(correct_answers)
    feedback = {}

    for q_id, correct_answer in correct_answers.items():
        user_answer = submission.answers.get(q_id)

        if user_answer is None:
            feedback[q_id] = "❌ Not answered"
            continue

        if normalize_answer(user_answer) == normalize_answer(correct_answer):
            correct_count += 1
            feedback[q_id] = "✅ Correct"
        else:
            feedback[q_id] = f"❌ Incorrect — your answer: {user_answer}, correct: {correct_answer}"

    # Handle questions submitted but not in answer key
    for q_id in submission.answers:
        if q_id not in correct_answers:
            feedback[q_id] = "⚠️ Question not in answer key — skipped"

    score = (correct_count / total_questions * 100) if total_questions > 0 else 0

    # Persist the attempt
    attempt_data = {
        "quiz_id": submission.quiz_id,
        "user_id": submission.user_id,
        "answers": submission.answers,
        "score": round(score, 2),
        "correct_count": correct_count,
        "total_questions": total_questions,
        "time_taken_seconds": submission.time_taken_seconds,
        "feedback": feedback
    }
    attempt_id = save_quiz_attempt(attempt_data)

    return GradeResponse(
        attempt_id=attempt_id,
        quiz_id=submission.quiz_id,
        user_id=submission.user_id,
        score=round(score, 2),
        total_questions=total_questions,
        correct_count=correct_count,
        correct_answers=correct_answers,
        feedback=feedback
    )


# ─── Quiz Attempt History ───────────────────────────────────────────

@app.get("/quiz-history/{user_id}")
async def get_quiz_history(user_id: str):
    """Get all quiz attempts for a user."""
    attempts = get_user_quiz_attempts(user_id)
    return {
        "user_id": user_id,
        "total_attempts": len(attempts),
        "attempts": attempts
    }


# ─── AI Interaction Logging ─────────────────────────────────────────

@app.post("/log-interaction")
async def log_ai_interaction(log: AIInteractionLog):
    """
    Logs AI usage metrics (tokens, latency, costs).
    Called by Orchestrator Service after each Gemini API call.
    """
    total_tokens = log.input_tokens + log.output_tokens
    cost_per_1k_tokens = 0.0005
    estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens

    interaction_data = {
        "user_id": log.user_id,
        "service_type": log.service_type,
        "input_tokens": log.input_tokens,
        "output_tokens": log.output_tokens,
        "latency_ms": log.latency_ms,
        "success": log.success,
        "model_name": log.model_name,
        "estimated_cost_usd": round(estimated_cost, 6)
    }

    interaction_id = save_interaction(interaction_data)

    print(f"📊 Logged interaction {interaction_id} for user {log.user_id}: "
          f"{total_tokens} tokens, {log.latency_ms}ms, ${estimated_cost:.6f}")

    return {
        "status": "logged",
        "interaction_id": interaction_id,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(estimated_cost, 6)
    }


# ─── Feedback Collection ────────────────────────────────────────────

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackSubmission):
    """
    Collects user ratings (1-5) and comments on AI responses.
    Called by Frontend after user rates a chat response.
    """
    feedback_data = {
        "user_id": feedback.user_id,
        "interaction_id": feedback.interaction_id,
        "rating": feedback.rating,
        "comment": feedback.comment,
        "feature_type": feedback.feature_type
    }

    feedback_id = save_feedback(feedback_data)

    print(f"⭐ Feedback {feedback_id}: {feedback.rating}/5 from user {feedback.user_id} "
          f"on {feedback.feature_type}")

    return {
        "status": "saved",
        "feedback_id": feedback_id,
        "thank_you_message": "Thanks for your feedback! It helps us improve."
    }


@app.get("/feedback/{user_id}")
async def get_user_feedback_history(user_id: str):
    """Get all feedback submitted by a user."""
    feedbacks = get_user_feedback(user_id)
    stats = get_feedback_stats(user_id)
    return {
        "user_id": user_id,
        "stats": stats,
        "feedback": feedbacks
    }


# ─── Analytics Dashboard ────────────────────────────────────────────
# IMPORTANT: Static routes MUST come before dynamic /{user_id} routes
# to prevent FastAPI from matching "system" as a user_id.

@app.get("/analytics/system/summary", response_model=AnalyticsResponse)
async def get_system_analytics():
    """
    Admin dashboard: System-wide analytics across all users.
    """
    stats = get_interaction_stats()
    return AnalyticsResponse(**stats)


@app.get("/analytics/system/feedback")
async def get_system_feedback():
    """Admin dashboard: System-wide feedback summary."""
    return get_feedback_stats()


@app.get("/analytics/{user_id}")
async def get_user_analytics(user_id: str):
    """
    Returns study analytics for a specific user.
    Aggregates quiz performance, AI usage, and feedback.
    """
    interaction_stats = get_interaction_stats(user_id)
    feedback_stats = get_feedback_stats(user_id)
    quiz_count = get_quiz_attempts_count(user_id)
    avg_score = get_average_quiz_score(user_id)
    today_count = get_today_interaction_count(user_id)

    return {
        "user_id": user_id,
        "quizzes_completed": quiz_count,
        "average_quiz_score": round(avg_score, 2),
        "ai_interactions_today": today_count,
        "ai_usage": interaction_stats,
        "feedback_summary": feedback_stats
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5006"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
