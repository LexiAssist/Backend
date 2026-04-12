"""
Retrieval Service — Vector Search Layer (PostgreSQL + pgvector)

Provides semantic vector similarity search using pgvector's cosine distance.
Requires a running PostgreSQL instance with the pgvector extension and
document chunks ingested by the Ingestion service.
"""

from sqlalchemy import create_engine, Column, String, Integer, Text, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from pgvector.sqlalchemy import Vector
from typing import List, Dict, Optional
import os


class Base(DeclarativeBase):
    pass


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True)
    material_id = Column(String, index=True)
    user_id = Column(String, index=True)
    chunk_text = Column(Text)
    embedding = Column(Vector(384))
    chunk_index = Column(Integer)


# ─── Database Connection (required) ─────────────────────────────────────

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://lexiassist:lexiassist_secret@localhost:5432/lexiassist"
)

try:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    print("✅ Retrieval: PostgreSQL + pgvector connected")
except Exception as e:
    print(f"❌ Retrieval: Database connection failed: {e}")
    raise RuntimeError(
        f"Retrieval service requires PostgreSQL with pgvector. Error: {e}"
    ) from e


# ─── Vector Search ───────────────────────────────────────────────────────

def search_similar_chunks(
    query_vector: List[float],
    user_id: str,
    material_id: Optional[str] = None,
    top_k: int = 5,
) -> List[Dict]:
    """
    Find document chunks most similar to the query vector using pgvector
    cosine distance. Filters by user_id for security, optionally by material_id.
    """
    db = SessionLocal()
    try:
        # Build query with cosine distance
        from sqlalchemy import select

        query = select(
            DocumentChunk,
            DocumentChunk.embedding.cosine_distance(query_vector).label("distance"),
        ).where(
            DocumentChunk.user_id == user_id
        )

        if material_id:
            query = query.where(DocumentChunk.material_id == material_id)

        query = query.order_by("distance").limit(top_k)

        results = db.execute(query).all()

        return [
            {
                "chunk_id": row.DocumentChunk.id,
                "material_id": row.DocumentChunk.material_id,
                "chunk_text": row.DocumentChunk.chunk_text,
                "similarity_score": round(1.0 - row.distance, 6),
                "chunk_index": row.DocumentChunk.chunk_index,
            }
            for row in results
        ]

    except Exception as e:
        print(f"❌ Vector search error: {e}")
        raise
    finally:
        db.close()


def get_search_mode() -> str:
    """Return the current search mode for health check reporting."""
    return "postgresql+pgvector"
