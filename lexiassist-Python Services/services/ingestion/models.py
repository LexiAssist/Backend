"""
Ingestion Service — Storage Layer (PostgreSQL + pgvector)

Stores document chunks with vector embeddings in PostgreSQL using pgvector.
Requires a running PostgreSQL instance with the pgvector extension.
"""

from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from pgvector.sqlalchemy import Vector
from datetime import datetime, timezone
import os
import uuid


class Base(DeclarativeBase):
    pass


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True)
    material_id = Column(String, index=True, nullable=False)
    user_id = Column(String, index=True, nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(384), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ─── Database Connection (required) ─────────────────────────────────────

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://lexiassist:lexiassist_secret@localhost:5432/lexiassist"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# Create tables on startup
try:
    from sqlalchemy import text as sa_text
    with engine.connect() as conn:
        conn.execute(sa_text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=engine)
    print("✅ Ingestion: PostgreSQL + pgvector connected, tables ready")
except Exception as e:
    print(f"❌ Ingestion: Database setup failed: {e}")
    raise RuntimeError(
        f"Ingestion service requires PostgreSQL with pgvector extension. Error: {e}"
    ) from e


def save_chunks(chunks_data: list, material_id: str, user_id: str) -> str:
    """
    Save document chunks with embeddings to PostgreSQL.
    Returns the storage method used (always 'postgresql').
    """
    db = SessionLocal()
    try:
        # Delete existing chunks for this material (re-ingestion)
        db.query(DocumentChunk).filter(
            DocumentChunk.material_id == material_id,
            DocumentChunk.user_id == user_id,
        ).delete()

        for chunk in chunks_data:
            db_chunk = DocumentChunk(
                id=str(uuid.uuid4()),
                material_id=material_id,
                user_id=user_id,
                chunk_text=chunk["text"],
                embedding=chunk["embedding"],
                chunk_index=chunk["index"],
            )
            db.add(db_chunk)

        db.commit()
        print(f"✅ Saved {len(chunks_data)} chunks to PostgreSQL")
        return "postgresql"
    except Exception as e:
        db.rollback()
        print(f"❌ Failed to save chunks: {e}")
        raise
    finally:
        db.close()


def get_chunks_by_material(material_id: str) -> list:
    """Retrieve all chunks for a given material."""
    db = SessionLocal()
    try:
        chunks = db.query(DocumentChunk).filter(
            DocumentChunk.material_id == material_id
        ).order_by(DocumentChunk.chunk_index).all()
        return [
            {
                "id": c.id,
                "text": c.chunk_text,
                "index": c.chunk_index,
                "material_id": c.material_id,
            }
            for c in chunks
        ]
    finally:
        db.close()
