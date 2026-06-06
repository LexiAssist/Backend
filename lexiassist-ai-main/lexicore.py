import os
from collections import deque
import google.generativeai as genai
import argparse
from pathlib import Path
import uuid
import logging
import httpx
from dotenv import load_dotenv
load_dotenv()

from database import SessionLocal, LexiChunk

COHERE_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-multilingual-v3.0")

logger = logging.getLogger(__name__)

# Lazily initialized after env var validation.
_model = None
_initialized = False


def _get_cohere_embeddings(texts: list[str], input_type: str = "search_document") -> list[list[float]]:
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        raise RuntimeError("Missing env var: COHERE_API_KEY")

    headers = {
        "Authorization": f"Bearer {cohere_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "texts": texts,
        "model": COHERE_MODEL,
        "input_type": input_type,
    }
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post("https://api.cohere.ai/v1/embed", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["embeddings"]
    except Exception as e:
        logger.error(f"Cohere embedding API call failed: {e}")
        raise e


def _ensure_initialized():
    """Lazy init — connects to Gemini on first use."""
    global _model, _initialized
    if _initialized:
        return
    _init_clients()
    _initialized = True


def _init_clients():
    global _model, COHERE_MODEL

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    cohere_api_key = os.getenv("COHERE_API_KEY")
    COHERE_MODEL = os.getenv("COHERE_EMBED_MODEL", COHERE_MODEL)

    if not gemini_api_key:
        raise RuntimeError("Missing env var: GEMINI_API_KEY")
    if not cohere_api_key:
        raise RuntimeError("Missing env var: COHERE_API_KEY")

    genai.configure(api_key=gemini_api_key)
    _model = genai.GenerativeModel("gemini-2.5-flash")

class LexiEngine:
    def __init__(self):
        self.history = deque(maxlen=8)

    @staticmethod
    def chunk_text(text, size=800, overlap=120):
        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + size, n)
            chunks.append(text[start:end])
            if end == n:
                break
            start = end - overlap
        return chunks

    def ingest_material(self, doc_id, text, course_code, source="uploaded_note"):
        _ensure_initialized()
        chunks = self.chunk_text(text)
        course_code = (course_code or "").strip()

        # Generate embeddings in batch via Cohere
        vectors = _get_cohere_embeddings(chunks, input_type="search_document")

        db = SessionLocal()
        try:
            # Deterministic IDs so re-upload overwrites same chunks.
            for i, chunk in enumerate(chunks):
                obj_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}::chunk::{i}"))
                
                # Check if exists, overwrite or insert
                existing = db.query(LexiChunk).filter(LexiChunk.id == obj_uuid).first()
                if existing:
                    existing.chunk_text = chunk
                    existing.doc_id = doc_id
                    existing.course = course_code
                    existing.chunk_index = i
                    existing.source = source
                    existing.embedding = vectors[i]
                else:
                    db_chunk = LexiChunk(
                        id=obj_uuid,
                        doc_id=doc_id,
                        course=course_code,
                        chunk_index=i,
                        chunk_text=chunk,
                        source=source,
                        embedding=vectors[i]
                    )
                    db.add(db_chunk)
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to ingest materials in Postgres: {e}")
            raise e
        finally:
            db.close()

        return f"Ingested {len(chunks)} chunks for {doc_id} in {course_code}"

    def _retrieve(self, user_query, course_code, top_k=5, min_score=0.0, alpha=0.5):
        _ensure_initialized()
        course_code = (course_code or "").strip()

        db = SessionLocal()
        try:
            # 1. Vector Search
            query_vector = _get_cohere_embeddings([user_query], input_type="search_query")[0]
            distance = LexiChunk.embedding.cosine_distance(query_vector)
            vector_results = db.query(LexiChunk, distance.label("distance")).filter(
                LexiChunk.course == course_code
            ).order_by("distance").limit(top_k * 2).all()

            # Map ID to (object, score)
            vector_hits = {}
            for row in vector_results:
                sim = 1.0 - row.distance
                vector_hits[row.LexiChunk.id] = (row.LexiChunk, sim)

            # 2. Keyword Search using Postgres tsvector
            from sqlalchemy import text as sql_text
            keyword_query = sql_text("""
                SELECT id, ts_rank_cd(to_tsvector('english', chunk_text), plainto_tsquery('english', :query)) AS rank
                FROM ai.lexi_chunks
                WHERE course = :course
                ORDER BY rank DESC
                LIMIT :limit
            """)
            keyword_results = db.execute(keyword_query, {
                "query": user_query,
                "course": course_code,
                "limit": top_k * 2
            }).all()

            keyword_hits = {}
            for row in keyword_results:
                # fetch the object from database
                chunk_obj = db.query(LexiChunk).filter(LexiChunk.id == row.id).first()
                if chunk_obj:
                    keyword_hits[row.id] = (chunk_obj, row.rank)

            # 3. Combine scores using RRF or simple weighted sum
            # If alpha = 1.0, vector only. If alpha = 0.0, keyword only.
            all_ids = set(vector_hits.keys()) | set(keyword_hits.keys())
            combined_scores = []

            # RRF Parameters
            k = 60
            for cid in all_ids:
                vector_rank = list(vector_hits.keys()).index(cid) + 1 if cid in vector_hits else None
                keyword_rank = list(keyword_hits.keys()).index(cid) + 1 if cid in keyword_hits else None

                rrf_score = 0.0
                if vector_rank is not None:
                    rrf_score += alpha * (1.0 / (k + vector_rank))
                if keyword_rank is not None:
                    rrf_score += (1.0 - alpha) * (1.0 / (k + keyword_rank))

                obj, vec_score = vector_hits.get(cid, (None, 0.0))
                if obj is None:
                    obj, _ = keyword_hits.get(cid)
                
                score = vec_score if vec_score > 0 else 0.0

                combined_scores.append((obj, rrf_score, score))

            # Sort by combined RRF score descending
            combined_scores.sort(key=lambda x: x[1], reverse=True)

            matches = []
            for obj, rrf_score, score in combined_scores[:top_k]:
                if score >= min_score:
                    matches.append({
                        "score": score,
                        "text": obj.chunk_text,
                        "doc_id": obj.doc_id,
                        "chunk_index": obj.chunk_index,
                        "source": obj.source,
                        "explain_score": f"RRF score: {rrf_score:.5f}, Vector similarity: {score:.4f}"
                    })

            # Fallback: if matches is empty, fetch objects
            if not matches:
                fallback = db.query(LexiChunk).filter(
                    LexiChunk.course == course_code
                ).order_by(LexiChunk.chunk_index.asc()).limit(top_k).all()
                for chunk in fallback:
                    matches.append({
                        "score": 0.0,
                        "text": chunk.chunk_text,
                        "doc_id": chunk.doc_id,
                        "chunk_index": chunk.chunk_index,
                        "source": chunk.source,
                        "explain_score": "Fallback fetch"
                    })

            return matches

        except Exception as e:
            logger.error(f"Error in lexicore _retrieve: {e}")
            return []
        finally:
            db.close()

    def contextual_chat(self, user_query, course_code):
        matches = self._retrieve(user_query, course_code)
        if not matches:
            return "I do not know based on the provided course materials."

        context_blocks = []
        for i, m in enumerate(matches, start=1):
            context_blocks.append(
                f"[{i}] doc={m['doc_id']} chunk={m['chunk_index']} score={m['score']:.3f}\n{m['text']}"
            )
        context = "\n\n---\n\n".join(context_blocks)

        memory_text = "\n".join([f"{t['role']}: {t['text']}" for t in self.history])

        prompt = (
            "You are an academic RAG assistant.\n"
            "Rules:\n"
            "1) Use only CONTEXT evidence.\n"
            "2) If evidence is insufficient, say: I do not know based on provided materials.\n"
            "3) Give concise explanation and include citations like [1], [2].\n\n"
            f"RECENT_CHAT:\n{memory_text}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{user_query}"
        )
        answer = _model.generate_content(prompt).text

        self.history.append({"role": "user", "text": user_query})
        self.history.append({"role": "assistant", "text": answer})
        return answer
    

def _check_env():
    missing = []
    if not os.getenv("GEMINI_API_KEY"):
        missing.append("GEMINI_API_KEY")
    if not os.getenv("COHERE_API_KEY"):
        missing.append("COHERE_API_KEY")
    if missing:
        raise RuntimeError("Missing env vars: " + ", ".join(missing))

def _read_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")

def _read_pdf(path: Path) -> str:
    try:
        import fitz  # pymupdf
    except ImportError as e:
        raise RuntimeError("Install pymupdf: pip install pymupdf") from e

    doc = fitz.open(str(path))
    text = "\n".join(page.get_text("text") for page in doc)
    return text.strip()

def _read_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError as e:
        raise RuntimeError("Install python-docx: pip install python-docx") from e

    doc = Document(str(path))
    lines = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(lines).strip()

def _read_document(path_str: str) -> str:
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    ext = p.suffix.lower()
    if ext == ".txt":
        text = _read_txt(p)
    elif ext == ".pdf":
        text = _read_pdf(p)
    elif ext == ".docx":
        text = _read_docx(p)
    else:
        raise ValueError("Unsupported file type. Use .txt, .pdf, or .docx")

    if not text.strip():
        raise ValueError(f"No extractable text found in {p.name}")
    return text

def _iter_upload_files(path_str: str, recursive: bool):
    p = Path(path_str)
    if p.is_file():
        yield p
        return
    if not p.is_dir():
        raise FileNotFoundError(f"Path not found: {p}")

    pattern = "**/*" if recursive else "*"
    for f in p.glob(pattern):
        if f.is_file() and f.suffix.lower() in {".txt", ".pdf", ".docx"}:
            yield f

def run_cli():
    parser = argparse.ArgumentParser(description="LexiAssist upload-first CLI")
    parser.add_argument("--course", required=True, help="Namespace, e.g. BIO101")
    parser.add_argument("--path", required=True, help="File or folder to upload")
    parser.add_argument("--doc-prefix", default="doc", help="Prefix for generated doc ids")
    parser.add_argument("--recursive", action="store_true", help="Include nested folders")
    parser.add_argument("--ask", help="Optional question to run immediately after upload")
    args = parser.parse_args()

    _check_env()
    _ensure_initialized()
    lexi = LexiEngine()

    uploaded = 0
    failed = 0
    for i, f in enumerate(_iter_upload_files(args.path, args.recursive), start=1):
        try:
            text = _read_document(str(f))
            doc_id = f"{args.doc_prefix}_{i:04d}_{f.stem}"
            msg = lexi.ingest_material(
                doc_id=doc_id,
                text=text,
                course_code=args.course,
                source=f.name
            )
            print(f"[OK] {f.name} -> {msg}")
            uploaded += 1
        except Exception as ex:
            print(f"[FAIL] {f}: {ex}")
            failed += 1

    print(f"\nUpload complete. success={uploaded}, failed={failed}, course={args.course}")

    if args.ask and uploaded > 0:
        print("\nQuestion:", args.ask)
        print("Answer:", lexi.contextual_chat(args.ask, args.course))



if __name__ == "__main__":
    run_cli()