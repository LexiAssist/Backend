from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Header, Request, HTTPException, status
from writing_assistant.routes import router as writing_router
from reading_assistant.routes import router as reading_router
from study_buddy.routes import router as study_router
from database import init_db
from fastapi.middleware.cors import CORSMiddleware
from worker import AIWorker
import os


worker = AIWorker()


INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "dev-internal-key")

async def verify_internal_key(request: Request):
    if request.url.path in ("/health", "/"):
        return
    key = request.headers.get("X-Internal-Key")
    if key != INTERNAL_API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid internal key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and start background worker on startup."""
    init_db()
    worker.start()
    yield
    worker.stop()


app = FastAPI(
    title="EdTech AI API",
    description="""
## Writing Assistant
Live lecture transcription and note generation.

| Endpoint | Description |
|---|---|
| `POST /writing/transcribe` | Audio → raw transcription via Gemini STT |
| `POST /writing/notes` | Raw transcription → clean structured markdown notes |

## Study Tools
Personalised flashcard and quiz generation from uploaded notes.

| Endpoint | Description |
|---|---|
| `POST /study/flashcards` | Upload notes → personalised flashcards |
| `POST /study/quiz` | Upload notes → multiple choice or theory quiz |
    """,
    version="1.0.0",
    lifespan=lifespan,
    dependencies=[Depends(verify_internal_key)],
)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
ALLOWED_ORIGINS = [o.strip() for o in ALLOWED_ORIGINS if o.strip() and o.strip() != "*"]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(study_router)
app.include_router(writing_router)
app.include_router(reading_router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ai-service"}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    from job_queue import get_job_status
    job = get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
