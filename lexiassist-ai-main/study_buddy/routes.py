# study_tools/routes.py
import io
import uuid
import logging
from typing import Literal, Optional

logger = logging.getLogger(__name__)

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import UserSession, SessionType, get_db
from study_buddy.flashcards import flashcard_graph
from study_buddy.quizzes import quiz_graph
from job_queue import enqueue_job
from shared.ai_cache import ai_cache


# ─── Cached graph invocations (Redis dedup, 24h TTL) ────────────────────────

@ai_cache("flashcards", ttl=86400)
def _generate_flashcards_cached(document_text: str, num_cards: int):
    """Deterministic wrapper so cache key is stable (no random IDs in state)."""
    return flashcard_graph.invoke({
        "document_text": document_text,
        "num_cards": num_cards,
        "flashcards": [],
    })


@ai_cache("quizzes", ttl=86400)
def _generate_quiz_cached(document_text: str, quiz_type: str, num_questions: int):
    """Deterministic wrapper so cache key is stable (no random IDs in state)."""
    return quiz_graph.invoke({
        "document_text": document_text,
        "quiz_type": quiz_type,
        "num_questions": num_questions,
        "questions": [],
    })

router = APIRouter(prefix="/study", tags=["Study Tools"])


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper
# ─────────────────────────────────────────────────────────────────────────────

async def extract_text(file: UploadFile) -> str:
    content = await file.read()
    filename = file.filename or ""

    if filename.endswith(".txt"):
        return content.decode("utf-8", errors="ignore")

    if filename.endswith(".pdf"):
        import pdfplumber
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)

    if filename.endswith(".docx"):
        from docx import Document
        doc = Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs)

    raise HTTPException(
        status_code=415,
        detail=f"Unsupported file type '{filename}'. Accepted: .pdf, .txt, .docx",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Flashcard schemas
# ─────────────────────────────────────────────────────────────────────────────

class Flashcard(BaseModel):
    front: str
    back: str
    topic: str


class FlashcardResponse(BaseModel):
    session_id: str = Field(..., description="Store this to retrieve this session later")
    user_id: str
    filename: Optional[str]
    num_requested: int
    num_generated: int
    flashcards: list[Flashcard]


class FlashcardSessionDetail(BaseModel):
    session_id: str
    user_id: str
    filename: Optional[str]
    created_at: str
    num_cards: int
    flashcards: list[Flashcard]


# ─────────────────────────────────────────────────────────────────────────────
# Quiz schemas
# ─────────────────────────────────────────────────────────────────────────────

class MCQOptions(BaseModel):
    A: str
    B: str
    C: str
    D: str


class MultipleChoiceQuestion(BaseModel):
    question: str
    options: MCQOptions
    correct_answer: Literal["A", "B", "C", "D"]
    explanation: str
    topic: str


class TheoryQuestion(BaseModel):
    question: str
    model_answer: str
    marking_guide: list[str]
    marks: int
    topic: str


class QuizResponse(BaseModel):
    session_id: str = Field(..., description="Store this to retrieve this session later")
    user_id: str
    filename: Optional[str]
    quiz_type: str
    num_requested: int
    num_generated: int
    questions: list[MultipleChoiceQuestion] | list[TheoryQuestion]


class QuizSessionDetail(BaseModel):
    session_id: str
    user_id: str
    filename: Optional[str]
    created_at: str
    quiz_type: str
    num_questions: int
    questions: list[MultipleChoiceQuestion] | list[TheoryQuestion]


class SessionSummary(BaseModel):
    session_id: str
    session_type: str
    filename: Optional[str]
    created_at: str
    quiz_type: Optional[str]
    num_cards: Optional[int]
    num_questions: Optional[int]


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Flashcards
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/flashcards",
    response_model=FlashcardResponse,
    summary="Generate flashcards from uploaded notes — creates a new session",
)
async def generate_flashcards(
    file: UploadFile = File(..., description="Notes or textbook (.pdf, .txt, .docx)"),
    user_id: str = Form(..., description="ID of the authenticated user"),
    num_cards: int = Form(10, ge=1, le=50, description="Number of flashcards to generate (1–50)"),
    db: Session = Depends(get_db),
):
    """
    Generates flashcards strictly from the uploaded document and persists
    them under a new `session_id`.

    Store the returned `session_id` on the frontend. The user can retrieve
    this session later via `GET /study/flashcards/session/{session_id}`.
    """
    document_text = await extract_text(file)
    logger.info(f"Flashcards request - filename: {file.filename}, length: {len(document_text)}, preview: {repr(document_text[:200])}")
    if not document_text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from the file.")

    session_id = str(uuid.uuid4())

    result = _generate_flashcards_cached(document_text, num_cards)

    cards_raw = result.get("flashcards", [])
    cards = [Flashcard(**c) for c in cards_raw]

    db.add(UserSession(
        session_id   = session_id,
        user_id      = user_id,
        session_type = SessionType.flashcard,
        filename     = file.filename,
        flashcards   = [c.dict() for c in cards],
        num_cards    = len(cards),
    ))
    db.commit()

    return FlashcardResponse(
        session_id    = session_id,
        user_id       = user_id,
        filename      = file.filename,
        num_requested = num_cards,
        num_generated = len(cards),
        flashcards    = cards,
    )


@router.get(
    "/flashcards/session/{session_id}",
    response_model=FlashcardSessionDetail,
    summary="Retrieve a past flashcard session",
)
def get_flashcard_session(
    session_id: str,
    user_id: str,
    db: Session = Depends(get_db),
):
    """
    Returns the flashcards from a previous session.
    `user_id` must match the session owner.
    """
    row = db.query(UserSession).filter(
        UserSession.session_id   == session_id,
        UserSession.user_id      == user_id,
        UserSession.session_type == SessionType.flashcard,
    ).first()

    if not row:
        raise HTTPException(status_code=404, detail="Flashcard session not found.")

    return FlashcardSessionDetail(
        session_id = row.session_id,
        user_id    = row.user_id,
        filename   = row.filename,
        created_at = row.created_at.isoformat(),
        num_cards  = row.num_cards or 0,
        flashcards = [Flashcard(**c) for c in (row.flashcards or [])],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Quiz
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/quiz",
    response_model=QuizResponse,
    summary="Generate a quiz from uploaded notes — creates a new session",
)
async def generate_quiz(
    file: UploadFile = File(..., description="Notes or textbook (.pdf, .txt, .docx)"),
    user_id: str = Form(..., description="ID of the authenticated user"),
    quiz_type: Literal["multiple_choice", "theory"] = Form(...),
    num_questions: int = Form(5, ge=1, le=30),
    db: Session = Depends(get_db),
):
    """
    Generates a quiz strictly from the uploaded document and persists it
    under a new `session_id`.

    Store the returned `session_id` on the frontend. The user can retrieve
    this session later via `GET /study/quiz/session/{session_id}`.
    """
    document_text = await extract_text(file)
    logger.info(f"Quiz request - filename: {file.filename}, type: {quiz_type}, length: {len(document_text)}, preview: {repr(document_text[:200])}")
    if not document_text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from the file.")

    session_id = str(uuid.uuid4())

    result = _generate_quiz_cached(document_text, quiz_type, num_questions)

    questions_raw = result.get("questions", [])

    if quiz_type == "multiple_choice":
        questions = [
            MultipleChoiceQuestion(**{**q, "options": MCQOptions(**q["options"])})
            for q in questions_raw
        ]
    else:
        questions = [TheoryQuestion(**q) for q in questions_raw]

    db.add(UserSession(
        session_id    = session_id,
        user_id       = user_id,
        session_type  = SessionType.quiz,
        filename      = file.filename,
        quiz_type     = quiz_type,
        questions     = [q.dict() for q in questions],
        num_questions = len(questions),
    ))
    db.commit()

    return QuizResponse(
        session_id    = session_id,
        user_id       = user_id,
        filename      = file.filename,
        quiz_type     = quiz_type,
        num_requested = num_questions,
        num_generated = len(questions),
        questions     = questions,
    )


@router.get(
    "/quiz/session/{session_id}",
    response_model=QuizSessionDetail,
    summary="Retrieve a past quiz session",
)
def get_quiz_session(
    session_id: str,
    user_id: str,
    db: Session = Depends(get_db),
):
    """
    Returns the questions from a previous quiz session.
    `user_id` must match the session owner.
    """
    row = db.query(UserSession).filter(
        UserSession.session_id   == session_id,
        UserSession.user_id      == user_id,
        UserSession.session_type == SessionType.quiz,
    ).first()

    if not row:
        raise HTTPException(status_code=404, detail="Quiz session not found.")

    questions_raw = row.questions or []
    if row.quiz_type == "multiple_choice":
        questions = [
            MultipleChoiceQuestion(**{**q, "options": MCQOptions(**q["options"])})
            for q in questions_raw
        ]
    else:
        questions = [TheoryQuestion(**q) for q in questions_raw]

    return QuizSessionDetail(
        session_id    = row.session_id,
        user_id       = row.user_id,
        filename      = row.filename,
        created_at    = row.created_at.isoformat(),
        quiz_type     = row.quiz_type,
        num_questions = row.num_questions or 0,
        questions     = questions,
    )


# ─────────────────────────────────────────────────────────────────────────────
# History
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/history",
    response_model=list[SessionSummary],
    summary="Get all flashcard and quiz sessions for a user",
)
def get_study_history(
    user_id: str,
    db: Session = Depends(get_db),
):
    """
    Returns metadata for all past flashcard and quiz sessions, newest first.
    Use this to populate the history/dashboard page.

    Each item has only the metadata — call the individual session endpoints
    to get the full cards or questions.
    """
    rows = db.query(UserSession).filter(
        UserSession.user_id.in_([user_id]),
        UserSession.session_type.in_([SessionType.flashcard, SessionType.quiz]),
    ).order_by(UserSession.created_at.desc()).all()

    return [
        SessionSummary(
            session_id    = r.session_id,
            session_type  = r.session_type,
            filename      = r.filename,
            created_at    = r.created_at.isoformat(),
            quiz_type     = r.quiz_type,
            num_cards     = r.num_cards,
            num_questions = r.num_questions,
        )
        for r in rows
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Async Job-based Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/flashcards/async")
async def generate_flashcards_async(
    file: UploadFile = File(..., description="Notes or textbook (.pdf, .txt, .docx)"),
    user_id: str = Form(..., description="ID of the authenticated user"),
    num_cards: int = Form(10, ge=1, le=50, description="Number of flashcards to generate (1–50)"),
):
    """Enqueue flashcard generation and return a job_id immediately."""
    document_text = await extract_text(file)
    if not document_text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from the file.")

    job_id = enqueue_job(
        "study_flashcards",
        {
            "document_text": document_text,
            "num_cards": num_cards,
            "filename": file.filename,
        },
        user_id,
    )
    return {"job_id": job_id, "status": "pending"}


@router.post("/quiz/async")
async def generate_quiz_async(
    file: UploadFile = File(..., description="Notes or textbook (.pdf, .txt, .docx)"),
    user_id: str = Form(..., description="ID of the authenticated user"),
    quiz_type: Literal["multiple_choice", "theory"] = Form(...),
    num_questions: int = Form(5, ge=1, le=30),
):
    """Enqueue quiz generation and return a job_id immediately."""
    document_text = await extract_text(file)
    if not document_text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from the file.")

    job_id = enqueue_job(
        "study_quiz",
        {
            "document_text": document_text,
            "quiz_type": quiz_type,
            "num_questions": num_questions,
            "filename": file.filename,
        },
        user_id,
    )
    return {"job_id": job_id, "status": "pending"}
