from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from writing_assistant.routes import router as writing_router
from reading_assistant.routes import router as reading_router
from study_buddy.routes import router as study_router
from database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    init_db()
    yield

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
)

import os

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)