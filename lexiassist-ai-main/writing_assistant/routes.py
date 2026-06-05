# writing_assistant/routes.py
import uuid
from collections import deque
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from groq import Groq
import os

from database import UserSession, SessionType, get_db
from job_queue import enqueue_job

router = APIRouter(prefix="/writing", tags=["Writing Assistant"])

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Rolling context per session — stores cleaned note chunks
# so /notes can attend over the full lecture history
_session_contexts: dict[str, deque] = {}
CONTEXT_WINDOW_SIZE = 20


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_or_create_session(session_id: Optional[str]) -> tuple[str, deque]:
    sid = session_id or str(uuid.uuid4())
    if sid not in _session_contexts:
        _session_contexts[sid] = deque(maxlen=CONTEXT_WINDOW_SIZE)
    return sid, _session_contexts[sid]


def transcribe_with_groq(audio_bytes: bytes, filename: str, language: str = "en") -> str:
    """
    Transcribes audio using Groq's Whisper large-v3-turbo.
    Groq accepts: mp3, mp4, mpeg, mpga, m4a, wav, webm — max 25MB per chunk.
    For live lecture chunks (5–15s), file size will be well under 1MB.
    """
    transcription = groq_client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model="whisper-large-v3-turbo",
        language=language,
        response_format="text",
    )
    return transcription.strip() if isinstance(transcription, str) else transcription.text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class TranscribeResponse(BaseModel):
    session_id: str = Field(..., description="Store this — pass it with every subsequent chunk and to /notes")
    raw_text: str   = Field(..., description="Raw transcription exactly as Groq Whisper heard it")


class NotesRequest(BaseModel):
    session_id: str  = Field(..., description="Session ID from /transcribe")
    raw_text: str    = Field(..., description="Full raw transcript the user has accumulated and approved")
    subject: str     = Field(default="General", description="Lecture subject e.g. Biology, Physics")
    save: bool       = Field(default=True, description="Persist notes to DB for later retrieval")
    user_id: str     = Field(..., description="ID of the authenticated user")


class NotesResponse(BaseModel):
    session_id: str
    user_id: str
    structured_notes: str = Field(..., description="Clean structured markdown notes")


class NotesSessionDetail(BaseModel):
    session_id: str
    user_id: str
    subject: str
    created_at: str
    structured_notes: str


class NotesSummary(BaseModel):
    session_id: str
    subject: str
    created_at: str


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/transcribe",
    response_model=TranscribeResponse,
    summary="Transcribe a live audio chunk via Groq Whisper",
)
async def transcribe(
    audio: UploadFile = File(..., description="5–15 second audio chunk (webm/wav/mp3/m4a). Max 25MB."),
    session_id: Optional[str] = Form(None, description="Omit on first chunk to start a new session"),
    language: str = Form("en", description="BCP-47 language code e.g. 'en', 'fr', 'es'"),
):
    """
    Transcribes a short audio chunk using **Groq Whisper large-v3-turbo**.

    **Client recording pattern:**
    Every 5–15 seconds the client stops the current recording chunk, sends it
    here, and starts the next chunk. The response streams back immediately.

    - First call: omit `session_id` — a new one is generated and returned
    - All subsequent chunks: pass the same `session_id` to keep them linked
    - When the lecture ends: pass the accumulated `raw_text` to `POST /writing/notes`

    **Streamed response:**
    Tokens are streamed back as SSE so the UI can render text as it arrives
    rather than waiting for the full chunk to process.

    ```
    event: session
    data: <session_id>

    data: The mitochondria
    data:  is the powerhouse
    data:  of the cell.
    data: [DONE]
    ```
    """
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="Audio file is empty.")

    sid, _ = get_or_create_session(session_id)

    try:
        raw_text = transcribe_with_groq(
            audio_bytes=audio_bytes,
            filename=audio.filename or f"chunk.{(audio.content_type or 'audio/webm').split('/')[-1]}",
            language=language,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq Whisper transcription failed: {e}")

    if not raw_text:
        raise HTTPException(
            status_code=422,
            detail="No speech detected in this audio chunk.",
        )

    # Stream the raw transcription token by token as SSE
    async def stream_tokens():
        yield f"event: session\ndata: {sid}\n\n"

        # Stream word by word so the UI feels live
        words = raw_text.split(" ")
        for i, word in enumerate(words):
            token = word if i == 0 else f" {word}"
            yield f"data: {token}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_tokens(), media_type="text/event-stream")


@router.post(
    "/notes",
    response_model=NotesResponse,
    summary="Convert accumulated raw transcript into structured notes",
)
async def generate_notes(
    req: NotesRequest,
    db: Session = Depends(get_db),
):
    """
    Takes the **full raw transcript** the user has accumulated across all chunks
    in this session and converts it into clean, structured markdown notes.

    Uses the rolling session context so Gemini can:
    - Correct misheared words using surrounding lecture content
    - Resolve sentences cut off at chunk boundaries
    - Maintain consistent terminology across the whole lecture
    - Organise content under correct topic headings

    If `save=true` (default), the notes are persisted to PostgreSQL under the
    `session_id` so the user can retrieve them later from their history.

    **Call this once at the end of a recording session** (or per section if the
    lecture is long and the student wants incremental notes).
    """
    if not req.raw_text.strip():
        raise HTTPException(status_code=422, detail="raw_text cannot be empty.")

    sid = req.session_id
    if sid not in _session_contexts:
        _session_contexts[sid] = deque(maxlen=CONTEXT_WINDOW_SIZE)
    ctx = _session_contexts[sid]

    system = SystemMessage(content=f"""You are an expert academic note-taker transcribing a live {req.subject} lecture.

You will receive a raw speech-to-text transcript which may contain:
- Filler words (um, uh, like, you know)
- Misheared words or homophones — use lecture context to correct these
- Repeated or restarted sentences from the speaker
- Incomplete thoughts cut off at chunk recording boundaries

Convert this into well-organised markdown notes.

Format rules:
- ## for main topics, ### for subtopics
- Bullet points for key facts and explanations
- **Bold** all definitions and key terms
- > Blockquote examples or analogies the teacher gives
- [unclear: ...] for anything genuinely unresolvable from context
- Do NOT add any information not present in the transcript
- Do NOT summarise — preserve all content, just clean and organise it""")

    # Prior cleaned chunks as alternating Human/AI turns so Gemini attends
    # over the full lecture history when resolving mishears
    messages = [system]
    for i, prior_chunk in enumerate(ctx):
        if i % 2 == 0:
            messages.append(HumanMessage(content=prior_chunk))
        else:
            messages.append(AIMessage(content=prior_chunk))

    messages.append(
        HumanMessage(content=f"Subject: {req.subject}\n\nRaw transcript:\n{req.raw_text}")
    )

    try:
        response = llm.invoke(messages)
        structured_notes = response.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Notes generation failed: {e}")

    # Save cleaned notes into rolling context for future chunks
    ctx.append(structured_notes)

    # Persist to PostgreSQL if requested
    if req.save:
        existing = db.query(UserSession).filter(
            UserSession.session_id   == sid,
            UserSession.session_type == SessionType.notes,
        ).first()

        if existing:
            # Update in place if the user is adding more to the same session
            existing.structured_notes = structured_notes
            existing.subject          = req.subject
        else:
            db.add(UserSession(
                session_id       = sid,
                user_id          = req.user_id,
                session_type     = SessionType.notes,
                subject          = req.subject,
                structured_notes = structured_notes,
            ))
        db.commit()

    return NotesResponse(
        session_id       = sid,
        user_id          = req.user_id,
        structured_notes = structured_notes,
    )


@router.get(
    "/notes/session/{session_id}",
    response_model=NotesSessionDetail,
    summary="Retrieve past notes by session ID",
)
def get_notes_session(
    session_id: str,
    user_id: str,
    db: Session = Depends(get_db),
):
    """
    Returns the structured notes from a previous session.
    `user_id` must match the session owner.
    """
    row = db.query(UserSession).filter(
        UserSession.session_id   == session_id,
        UserSession.user_id      == user_id,
        UserSession.session_type == SessionType.notes,
    ).first()

    if not row:
        raise HTTPException(status_code=404, detail="Notes session not found.")

    return NotesSessionDetail(
        session_id       = row.session_id,
        user_id          = row.user_id,
        subject          = row.subject or "General",
        created_at       = row.created_at.isoformat(),
        structured_notes = row.structured_notes or "",
    )


@router.get(
    "/notes/history",
    response_model=list[NotesSummary],
    summary="Get all notes sessions for a user",
)
def get_notes_history(
    user_id: str,
    db: Session = Depends(get_db),
):
    """
    Returns metadata for all past notes sessions, newest first.
    Use this to populate the user's notes history page.
    """
    rows = db.query(UserSession).filter(
        UserSession.user_id      == user_id,
        UserSession.session_type == SessionType.notes,
    ).order_by(UserSession.created_at.desc()).all()

    return [
        NotesSummary(
            session_id = r.session_id,
            subject    = r.subject or "General",
            created_at = r.created_at.isoformat(),
        )
        for r in rows
    ]

# ─────────────────────────────────────────────────────────────────────────────
# Async Job-based Route
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/notes/async")
async def generate_notes_async(req: NotesRequest):
    """Enqueue note generation and return a job_id immediately."""
    if not req.raw_text.strip():
        raise HTTPException(status_code=422, detail="raw_text cannot be empty.")

    job_id = enqueue_job(
        "writing_notes",
        req.model_dump(),
        req.user_id,
    )
    return {"job_id": job_id, "status": "pending"}
