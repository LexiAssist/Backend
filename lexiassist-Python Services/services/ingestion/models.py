from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from datetime import datetime, timezone
import os
import json

# Try to import pgvector. If not available, we'll handle gracefully.
try:
    from pgvector.sqlalchemy import Vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
    print("Warning: pgvector not available. Will use mock storage for now.")
    # Create a dummy Vector class for code compatibility
    class Vector:
        def __init__(self, dim):
            self.dim = dim

class Base(DeclarativeBase):
    pass

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True)
    material_id = Column(String, index=True)
    user_id = Column(String, index=True)
    chunk_text = Column(Text)
    # For pgvector - dimension matches your embedding model (384)
    embedding = Vector(384) if PGVECTOR_AVAILABLE else Column(Text)
    chunk_index = Column(Integer)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

# Database setup (will work when docker-compose provides PostgreSQL)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://lexiassist:password@localhost:5432/lexiassist_db"
)

engine = None
SessionLocal = None

if PGVECTOR_AVAILABLE:
    try:
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(bind=engine)
        print("✅ Database connection configured")
    except Exception as e:
        print(f"⚠️  Database not available yet: {e}")
        print("   Will use mock JSON storage instead")

def init_database():
    """Create tables - run this once when DB is ready"""
    if PGVECTOR_AVAILABLE and engine:
        Base.metadata.create_all(bind=engine)
        print("Database tables created!")
    else:
        print("Cannot create tables - PostgreSQL not connected")

def save_chunks(chunks_data: list, material_id: str, user_id: str):
    """
    Saves chunks to database (or JSON if DB not available).
    This function works NOW and will work later with Docker!
    """
    import uuid

    # Try database first, fallback to JSON
    if PGVECTOR_AVAILABLE and SessionLocal:
        try:
            db = SessionLocal()
            for chunk in chunks_data:
                db_chunk = DocumentChunk(
                    id=str(uuid.uuid4()),
                    material_id=material_id,
                    user_id=user_id,
                    chunk_text=chunk["text"],
                    embedding=chunk["embedding"],
                    chunk_index=chunk["index"]
                )
                db.add(db_chunk)
            db.commit()
            print(f"✅ Saved {len(chunks_data)} chunks to PostgreSQL")
            return "database"
        except Exception as e:
            print(f"Database save failed: {e}")
            print("Falling back to JSON storage...")

    # Fallback: Save to JSON file (for testing without Docker)
    filename = f"db_mock_{material_id}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({
            "material_id": material_id,
            "user_id": user_id,
            "chunks": chunks_data,
            "saved_at": datetime.now().isoformat()
        }, f, indent=2)

    print(f"💾 Saved {len(chunks_data)} chunks to JSON mock: {filename}")
    return "json"

# Test function
if __name__ == "__main__":
    print("Models Module - Test Mode")
    print("=" * 50)

    # Try to load the embeddings we just created
    try:
        with open("chunks_with_embeddings.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Loaded {len(data)} chunks with embeddings")
        print(f"Sample embedding length: {len(data[0]['embedding'])} dimensions")

        # Try to save (will use JSON fallback since no PostgreSQL yet)
        result = save_chunks(data, "ibm-hackathon-001", "student-123")
        print(f"\nStorage method used: {result}")

    except FileNotFoundError:
        print("Error: chunks_with_embeddings.json not found!")
        print("Please run embedder.py first.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
