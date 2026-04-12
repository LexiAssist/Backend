import os
from collections import deque
import google.generativeai as genai
import argparse
from pathlib import Path
import uuid

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter, MetadataQuery
from dotenv import load_dotenv
load_dotenv()

COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION", "LexiChunks")
VECTOR_NAME = os.getenv("WEAVIATE_VECTOR_NAME", "chunk_vector")
COHERE_MODEL = os.getenv("COHERE_EMBED_MODEL")  # e.g. "embed-multilingual-v3.0"

# Lazily initialized after env var validation.
_wclient = None
_collection = None
_model = None
_initialized = False


def _ensure_initialized():
    """Lazy init — connects to Weaviate and Gemini on first use."""
    global _wclient, _collection, _model, _initialized
    if _initialized:
        return
    _init_clients()
    _initialized = True



def _init_clients():
    global _wclient, _collection, _model, COLLECTION_NAME, VECTOR_NAME, COHERE_MODEL

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    cohere_api_key = os.getenv("COHERE_API_KEY")
    COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION", COLLECTION_NAME)
    VECTOR_NAME = os.getenv("WEAVIATE_VECTOR_NAME", VECTOR_NAME)
    COHERE_MODEL = os.getenv("COHERE_EMBED_MODEL", COHERE_MODEL)

    genai.configure(api_key=gemini_api_key)
    _model = genai.GenerativeModel("gemini-2.5-flash")

    headers = {}
    if cohere_api_key:
        headers["X-Cohere-Api-Key"] = cohere_api_key

    if not weaviate_url:
        raise RuntimeError("Missing env var: WEAVIATE_URL")

    if weaviate_api_key:
        _wclient = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key),
            headers=headers,
        )
    else:
        # Some WCD setups may be public / not require an API key.
        _wclient = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            headers=headers,
        )

    if not _wclient.collections.exists(COLLECTION_NAME):
        _wclient.collections.create(
            name=COLLECTION_NAME,
            # Use vector_config (new API) to vectorize only the chunk_text property.
            vector_config=[
                Configure.Vectors.text2vec_cohere(
                    name=VECTOR_NAME,
                    source_properties=["chunk_text"],
                    model=COHERE_MODEL,
                )
            ],
            properties=[
                Property(name="chunk_text", data_type=DataType.TEXT),
                Property(name="doc_id", data_type=DataType.TEXT),
                Property(name="course", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
                Property(name="source", data_type=DataType.TEXT),
            ],
        )

    _collection = _wclient.collections.use(COLLECTION_NAME)

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

        # Deterministic IDs so re-upload overwrites same chunks.
        with _collection.batch.dynamic() as batch:
            for i, chunk in enumerate(chunks):
                obj_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}::chunk::{i}")
                batch.add_object(
                    uuid=obj_uuid,
                    properties={
                        "chunk_text": chunk,
                        "doc_id": doc_id,
                        "course": course_code,
                        "chunk_index": i,
                        "source": source,
                    },
                )

        return f"Ingested {len(chunks)} chunks for {doc_id} in {course_code}"

    def _retrieve(self, user_query, course_code, top_k=5, min_score=0.0, alpha=0.5):
        _ensure_initialized()
        course_code = (course_code or "").strip()

        # Hybrid search combines keyword + vector. alpha=0 -> keyword only, alpha=1 -> vector only.
        hybrid_kwargs = dict(
            query=user_query,
            alpha=alpha,
            limit=top_k,
            filters=Filter.by_property("course").equal(course_code),
            query_properties=["chunk_text"],
            return_metadata=MetadataQuery(score=True, explain_score=True),
            return_properties=["chunk_text", "doc_id", "course", "chunk_index", "source"],
        )

        # If the collection has a named vector, Weaviate requires target_vector.
        try:
            resp = _collection.query.hybrid(
                target_vector=VECTOR_NAME,
                **hybrid_kwargs,
            )
        except Exception:
            resp = _collection.query.hybrid(
                **hybrid_kwargs,
            )

        matches = []
        for obj in resp.objects:
            props = obj.properties or {}
            chunk_text = props.get("chunk_text")
            score = getattr(obj.metadata, "score", 0.0) or 0.0
            if chunk_text and score >= min_score:
                matches.append(
                    {
                        "score": score,
                        "text": chunk_text,
                        "doc_id": props.get("doc_id", "unknown"),
                        "chunk_index": props.get("chunk_index", -1),
                        "source": props.get("source", "unknown"),
                        "explain_score": getattr(obj.metadata, "explain_score", None),
                    }
                )

        # Fallback: if hybrid returns nothing (common for generic questions like "summarize"),
        # fetch a few chunks from this course so the LLM has something to work with.
        if not matches:
            try:
                fallback = _collection.query.fetch_objects(
                    limit=top_k,
                    filters=Filter.by_property("course").equal(course_code),
                    return_properties=["chunk_text", "doc_id", "course", "chunk_index", "source"],
                )
                for obj in fallback.objects:
                    props = obj.properties or {}
                    chunk_text = props.get("chunk_text")
                    if chunk_text:
                        matches.append(
                            {
                                "score": 0.0,
                                "text": chunk_text,
                                "doc_id": props.get("doc_id", "unknown"),
                                "chunk_index": props.get("chunk_index", -1),
                                "source": props.get("source", "unknown"),
                                "explain_score": None,
                            }
                        )
            except Exception:
                pass

        return matches

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
    if not os.getenv("WEAVIATE_URL"):
        missing.append("WEAVIATE_URL")
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

    if _wclient is not None:
        try:
            _wclient.close()
        except Exception:
            pass

if __name__ == "__main__":
    run_cli()