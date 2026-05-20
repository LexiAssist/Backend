# database.py
import enum
import os
from datetime import datetime
from dotenv import load_dotenv


from sqlalchemy import (
    Column, DateTime, Enum as SAEnum,
    Integer, JSON, String, Text, create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker
load_dotenv()  # Load environment variables from .env file
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://lexiassist:lexiassist_secret@localhost:5432/lexiassist")

engine       = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base         = declarative_base()


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class SessionType(str, enum.Enum):
    notes     = "notes"       # writing assistant
    reading   = "reading"     # reading assistant
    flashcard = "flashcard"   # study tools
    quiz      = "quiz"        # study tools


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class UserSession(Base):
    """
    One row per user action (transcription session, reading upload,
    flashcard set, or quiz). All features share this table — unused
    columns for a given session_type are left NULL.
    """
    __tablename__ = "user_sessions"
    __table_args__ = {"schema": "ai"}

    session_id           = Column(String,          primary_key=True)
    user_id              = Column(String,          nullable=False, index=True)
    session_type         = Column(SAEnum(SessionType), nullable=False)
    filename             = Column(String,          nullable=True)
    created_at           = Column(DateTime,        default=datetime.utcnow)

    # ── Writing assistant (notes) ──────────────────────────────────────────
    subject              = Column(String,          nullable=True)
    structured_notes     = Column(Text,            nullable=True)

    # ── Reading assistant ──────────────────────────────────────────────────
    weaviate_collection  = Column(String,          nullable=True)
    summary              = Column(Text,            nullable=True)
    summary_type         = Column(String,          nullable=True)
    tts_audio_b64        = Column(Text,            nullable=True)
    vocab_terms          = Column(JSON,            nullable=True)   # list[{term, definition, context_snippet}]

    # ── Flashcards ─────────────────────────────────────────────────────────
    flashcards           = Column(JSON,            nullable=True)   # list[{front, back, topic}]
    num_cards            = Column(Integer,         nullable=True)

    # ── Quiz ───────────────────────────────────────────────────────────────
    quiz_type            = Column(String,          nullable=True)   # "multiple_choice" | "theory"
    questions            = Column(JSON,            nullable=True)   # list[question dicts]
    num_questions        = Column(Integer,         nullable=True)


# Ensure the 'ai' schema exists before creating tables
from sqlalchemy import text
with engine.connect() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS ai"))
    conn.commit()

Base.metadata.create_all(bind=engine)


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI dependency
# ─────────────────────────────────────────────────────────────────────────────

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database schema and tables."""
    from sqlalchemy import text
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS ai"))
        conn.commit()
    Base.metadata.create_all(bind=engine)
