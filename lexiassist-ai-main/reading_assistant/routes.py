# reading_assistant/routes.py
import base64
import io
import uuid
from typing import Literal, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import UserSession, SessionType, get_db
from reading_assistant.reading_engine import ReadingState, reading_graph
from reading_assistant.tts_engine import TTSGenerator
from job_queue import enqueue_job, get_job_status

router = APIRouter(prefix="/reading", tags=["Reading Assistant"])
tts_generator = TTSGenerator()

AVAILABLE_VOICES = ["Zephyr", "Puck", "Athena", "Aria", "Nova"]





# ─────────────────────────────────────────────────────────────────────────────
# Helpers
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
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class VocabTerm(BaseModel):
    term: str
    definition: str
    context_snippet: str


class ReadingAnalysisResponse(BaseModel):
    session_id: str = Field(..., description="Store this — used to retrieve this session later")
    user_id: str
    summary_type: str
    summary: str
    vocab_terms: list[VocabTerm]
    tts_audio_b64: str
    audio_mime_type: str
    voice: str


class SessionSummary(BaseModel):
    session_id: str
    session_type: str
    filename: Optional[str]
    created_at: str
    summary_type: Optional[str]
    quiz_type: Optional[str]
    num_cards: Optional[int]
    num_questions: Optional[int]


class ReadingSessionDetail(BaseModel):
    session_id: str
    user_id: str
    filename: Optional[str]
    created_at: str
    summary_type: str
    summary: str
    vocab_terms: list[VocabTerm]
    tts_audio_b64: str


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/analyse",
    response_model=ReadingAnalysisResponse,
    summary="Analyse a document — generates a new session",
)
async def analyse_document(
    file: UploadFile = File(..., description="Document to analyse (.pdf, .txt, .docx)"),
    user_id: str = Form(..., description="ID of the authenticated user"),
    summary_type: Literal["brief", "concise", "detailed"] = Form("concise"),
    voice: str = Form("Zephyr"),
    speaker_label: str = Form("Reader"),
    temperature: float = Form(1.0, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
):
    """
    Uploads a document and runs the full reading pipeline.

    Every call generates a **new session_id** and a **new Weaviate collection**
    scoped to that session — so each upload is fully isolated.

    The `session_id` is returned in the response. The client should store it
    (in localStorage, or your user profile DB) to enable history retrieval.

    Results are persisted in PostgreSQL so the user can retrieve them later
    via `GET /reading/session/{session_id}` without re-running the pipeline.
    """
    if voice not in AVAILABLE_VOICES:
        raise HTTPException(
            status_code=422,
            detail=f"Voice '{voice}' not recognised. Available: {AVAILABLE_VOICES}",
        )

    document_text = await extract_text(file)
    if not document_text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from the file.")

    # New session + new Weaviate collection per upload
    session_id = str(uuid.uuid4())
    collection_name = f"reading_{session_id.replace('-', '_')}"

    result: ReadingState = reading_graph.invoke({
        "document_text":  document_text,
        "summary":        "",
        "vocab_terms":    [],
        "tts_audio_b64":  "",
        "tts_config": {
            "voice":         voice,
            "speaker_label": speaker_label,
            "temperature":   temperature,
        },
        "stored_doc_id":  "",
        "summary_type":   summary_type,
        "audio":          None,
        "collection_name": collection_name,
    })

    audio_result = result.get("audio") or {}
    raw_bytes    = audio_result.get("audio_data", b"")
    mime_type    = audio_result.get("mime_type", "audio/wav")
    tts_b64      = base64.b64encode(raw_bytes).decode() if raw_bytes else result.get("tts_audio_b64", "")

    vocab_raw  = result.get("vocab_terms", [])
    vocab_list = []
    for item in vocab_raw:
        try:
            vocab_list.append(VocabTerm(**item))
        except Exception:
            continue

    # Persist to PostgreSQL
    db_session = UserSession(
        session_id          = session_id,
        user_id             = user_id,
        session_type        = SessionType.reading,
        filename            = file.filename,
        weaviate_collection = collection_name,
        summary             = result.get("summary", ""),
        summary_type        = result.get("summary_type", summary_type),
        tts_audio_b64       = tts_b64,
        vocab_terms         = [v.dict() for v in vocab_list],
    )
    db.add(db_session)
    db.commit()

    return ReadingAnalysisResponse(
        session_id    = session_id,
        user_id       = user_id,
        summary_type  = result.get("summary_type", summary_type),
        summary       = result.get("summary", ""),
        vocab_terms   = vocab_list,
        tts_audio_b64 = tts_b64,
        audio_mime_type = mime_type,
        voice         = voice,
    )


@router.get(
    "/session/{session_id}",
    response_model=ReadingSessionDetail,
    summary="Retrieve a past reading session by session_id",
)
def get_reading_session(
    session_id: str,
    user_id: str,
    db: Session = Depends(get_db),
):
    """
    Returns the full results of a previous reading session.

    The `user_id` must match the one that created the session — prevents
    users from accessing each other's sessions.
    """
    row = db.query(UserSession).filter(
        UserSession.session_id   == session_id,
        UserSession.user_id      == user_id,
        UserSession.session_type == SessionType.reading,
    ).first()

    if not row:
        raise HTTPException(status_code=404, detail="Session not found.")

    return ReadingSessionDetail(
        session_id    = row.session_id,
        user_id       = row.user_id,
        filename      = row.filename,
        created_at    = row.created_at.isoformat(),
        summary_type  = row.summary_type,
        summary       = row.summary,
        vocab_terms   = [VocabTerm(**v) for v in (row.vocab_terms or [])],
        tts_audio_b64 = row.tts_audio_b64 or "",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Async Job-based Routes (for long-running analysis)
# ─────────────────────────────────────────────────────────────────────────────

class StartAnalysisResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    progress_message: str
    session_id: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


@router.post(
    "/analyse/async",
    response_model=StartAnalysisResponse,
    summary="Start async document analysis - returns immediately with job_id",
)
async def start_analyse_document(
    file: UploadFile = File(..., description="Document to analyse (.pdf, .txt, .docx)"),
    user_id: str = Form(..., description="ID of the authenticated user"),
    summary_type: Literal["brief", "concise", "detailed"] = Form("concise"),
    voice: str = Form("Zephyr"),
    speaker_label: str = Form("Reader"),
    temperature: float = Form(1.0, ge=0.0, le=1.0),
):
    """
    Uploads a document and starts analysis in the background.

    Returns immediately with a job_id. Use GET /reading/analyse/status/{job_id}
    to poll for completion, then GET /reading/session/{session_id} to get results.
    """
    if voice not in AVAILABLE_VOICES:
        raise HTTPException(
            status_code=422,
            detail=f"Voice '{voice}' not recognised. Available: {AVAILABLE_VOICES}",
        )

    document_text = await extract_text(file)
    if not document_text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from the file.")

    job_id = enqueue_job(
        task_type="reading_analyse",
        payload={
            "document_text": document_text,
            "user_id": user_id,
            "filename": file.filename or "unknown",
            "summary_type": summary_type,
            "voice": voice,
            "speaker_label": speaker_label,
            "temperature": temperature,
        },
        user_id=user_id,
    )

    return StartAnalysisResponse(
        job_id=job_id,
        status="pending",
        message="Analysis started. Poll /reading/analyse/status/{job_id} for progress.",
    )


@router.get(
    "/analyse/status/{job_id}",
    response_model=JobStatusResponse,
    summary="Get the status of an async analysis job",
)
async def get_analysis_status(
    job_id: str,
    user_id: str,
):
    """
    Get the current status of an analysis job.

    Poll this endpoint every 2-3 seconds after starting an async analysis.
    When status is 'completed', use session_id to fetch full results.
    """
    job = get_job_status(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Redis returns empty strings for unset hash fields — normalise to None
    session_id = job.get("session_id")
    if session_id == "":
        session_id = None
    error = job.get("error")
    if error == "":
        error = None

    return JobStatusResponse(
        job_id=job_id,
        status=job.get("status", "unknown"),
        progress=int(job.get("progress", 0)),
        progress_message=job.get("progress_message", ""),
        session_id=session_id,
        error=error,
        created_at=job.get("created_at", ""),
        updated_at=job.get("updated_at", ""),
    )
