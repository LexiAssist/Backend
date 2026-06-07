"""
Retrieval Service — Database & Search Layer

Provides vector similarity search via:
1. PostgreSQL + pgvector (when Docker/production DB is available)
2. JSON fallback search using cosine similarity (for local development)

The JSON fallback reads the mock DB files produced by the Ingestion service,
so the full RAG pipeline works end-to-end even without PostgreSQL.
"""

from sqlalchemy import create_engine, Column, String, Integer, Text, select
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from typing import List, Dict, Optional
import numpy as np
import json
import glob
import os

# Try to import pgvector
try:
    from pgvector.sqlalchemy import Vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
    print("⚠️  pgvector not available — will use JSON fallback search")


class Base(DeclarativeBase):
    pass


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True)
    material_id = Column(String, index=True)
    user_id = Column(String, index=True)
    chunk_text = Column(Text)
    embedding = Column(Vector(384)) if PGVECTOR_AVAILABLE else Column(Text)
    chunk_index = Column(Integer)


# Database connection
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://lexiassist:password@localhost:5432/lexiassist_db"
)

engine = None
SessionLocal = None
DB_CONNECTED = False

if PGVECTOR_AVAILABLE:
    try:
        engine = create_engine(DATABASE_URL)
        # Test the connection
        with engine.connect() as conn:
            conn.execute(select(1))
        SessionLocal = sessionmaker(bind=engine)
        DB_CONNECTED = True
        print("✅ PostgreSQL + pgvector connected")
        
        # Initialize pgvector extension and create tables if they do not exist
        try:
            with engine.connect() as conn:
                from sqlalchemy import text
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                conn.commit()
            Base.metadata.create_all(bind=engine)
            print("✅ Database tables verified/created for Retrieval Service")
        except Exception as e:
            print(f"⚠️  Failed to verify/create database tables for Retrieval: {e}")
            
    except Exception as e:
        print(f"⚠️  PostgreSQL not available: {e}")
        print("   Using JSON fallback search instead")
        DB_CONNECTED = False


# ─── JSON Fallback Search (cosine similarity) ───────────────────────

# Directory where ingestion service saves mock DB files
INGESTION_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "ingestion"
)


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _load_json_chunks() -> List[dict]:
    """
    Load all chunks from ingestion's JSON mock files.
    These files are produced by the ingestion service when PostgreSQL is not available.
    """
    all_chunks = []
    pattern = os.path.join(INGESTION_DATA_DIR, "db_mock_*.json")
    files = glob.glob(pattern)

    if not files:
        print(f"   No JSON mock files found at: {pattern}")
        return []

    for filepath in files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            material_id = data.get("material_id", os.path.basename(filepath))
            user_id = data.get("user_id", "unknown")
            chunks = data.get("chunks", [])

            for chunk in chunks:
                if "embedding" in chunk:
                    all_chunks.append({
                        "material_id": material_id,
                        "user_id": user_id,
                        "chunk_text": chunk.get("text", ""),
                        "embedding": chunk["embedding"],
                        "chunk_index": chunk.get("index", 0)
                    })
        except (json.JSONDecodeError, IOError) as e:
            print(f"   Warning: Could not load {filepath}: {e}")

    print(f"   Loaded {len(all_chunks)} chunks from {len(files)} JSON file(s)")
    return all_chunks


def search_json_fallback(
    query_vector: List[float],
    user_id: str,
    material_id: Optional[str] = None,
    top_k: int = 5
) -> List[Dict]:
    """
    Search through JSON mock files using cosine similarity.
    This provides real semantic search without PostgreSQL.
    """
    chunks = _load_json_chunks()

    if not chunks:
        print("   No chunks available for search — returning empty results")
        return []

    # Filter by user_id (security)
    filtered = [c for c in chunks if c["user_id"] == user_id]

    # If no chunks for this user, search all (dev convenience)
    if not filtered:
        print(f"   No chunks for user '{user_id}', searching all available chunks")
        filtered = chunks

    # Filter by material_id if specified
    if material_id:
        material_filtered = [c for c in filtered if c["material_id"] == material_id]
        if material_filtered:
            filtered = material_filtered
        else:
            print(f"   No chunks for material '{material_id}', searching across all materials")

    # Compute similarity scores
    scored = []
    for chunk in filtered:
        score = cosine_similarity(query_vector, chunk["embedding"])
        scored.append({
            "chunk_id": f"json-{chunk['material_id']}-{chunk['chunk_index']}",
            "material_id": chunk["material_id"],
            "chunk_text": chunk["chunk_text"],
            "similarity_score": round(score, 6),
            "chunk_index": chunk["chunk_index"]
        })

    # Sort by similarity descending, take top_k
    scored.sort(key=lambda x: x["similarity_score"], reverse=True)
    return scored[:top_k]


# ─── PostgreSQL pgvector Search ──────────────────────────────────────

def search_pgvector(
    query_vector: List[float],
    user_id: str,
    material_id: Optional[str] = None,
    top_k: int = 5
) -> List[Dict]:
    """
    Search using pgvector's cosine distance operator in PostgreSQL.
    """
    try:
        db = SessionLocal()

        query = select(
            DocumentChunk,
            DocumentChunk.embedding.cosine_distance(query_vector).label("distance")
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
                "chunk_index": row.DocumentChunk.chunk_index
            }
            for row in results
        ]

    except Exception as e:
        print(f"   ⚠️  pgvector search error: {e}")
        print("   Falling back to JSON search...")
        return search_json_fallback(query_vector, user_id, material_id, top_k)
    finally:
        if 'db' in locals():
            db.close()


# ─── Public API ──────────────────────────────────────────────────────

def search_similar_chunks(
    query_vector: List[float],
    user_id: str,
    material_id: str = None,
    top_k: int = 5
) -> List[Dict]:
    """
    Find chunks with vectors most similar to the query vector.
    Uses pgvector when PostgreSQL is available, otherwise falls back to
    JSON-based cosine similarity search.
    """
    if DB_CONNECTED and SessionLocal:
        print("   🔍 Using PostgreSQL + pgvector search")
        return search_pgvector(query_vector, user_id, material_id, top_k)
    else:
        print("   🔍 Using JSON fallback search (cosine similarity)")
        return search_json_fallback(query_vector, user_id, material_id, top_k)


def get_search_mode() -> str:
    """Return the current search mode for health check reporting."""
    if DB_CONNECTED:
        return "postgresql+pgvector"
    return "json-fallback"
