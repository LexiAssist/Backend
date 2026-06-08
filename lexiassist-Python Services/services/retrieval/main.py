import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from pydantic import BaseModel, Field
from typing import List, Optional
from database import search_similar_chunks, get_search_mode
import os
import uvicorn

import httpx

COHERE_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-multilingual-v3.0")

def _get_cohere_embeddings(texts: List[str], input_type: str = "search_query") -> List[List[float]]:
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        print("⚠️ COHERE_API_KEY is not configured in environment variables. Cohere embedding will fail!")
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
        print(f"❌ Cohere embedding API call failed: {e}")
        raise e


def verify_internal_key(request: Request, x_internal_key: str = Header(None)):
    if request.url.path in ("/", "/health"):
        return
    expected = os.getenv("INTERNAL_API_KEY", "dev-internal-key")
    if not x_internal_key or x_internal_key != expected:
        raise HTTPException(status_code=403, detail="Invalid or missing internal key")


app = FastAPI(
    title="LexiAssist Retrieval Service",
    description="Semantic search & vector retrieval for RAG",
    version="2.1.0",
    dependencies=[Depends(verify_internal_key)],
)

# Pydantic models
class RetrieveRequest(BaseModel):
    query: str = Field(..., description="User's question")
    user_id: str = Field(..., description="For security filtering")
    material_id: Optional[str] = Field(None, description="Optional: filter by specific material")
    top_k: int = Field(default=5, le=10, description="Number of chunks to return")

class ChunkResult(BaseModel):
    chunk_id: str
    material_id: str
    chunk_text: str
    similarity_score: float
    chunk_index: int

class RetrieveResponse(BaseModel):
    query: str
    query_embedding_preview: List[float]
    results: List[ChunkResult]
    search_mode: str
    cached: bool = False
    note: str

# Health check
@app.get("/")
async def root():
    search_mode = get_search_mode()
    return {
        "status": "healthy",
        "service": "retrieval",
        "port": 5003,
        "version": "2.1.0",
        "model": COHERE_MODEL,
        "embedding_dim": 1024,
        "search_mode": search_mode
    }

@app.get("/health")
async def health():
    search_mode = get_search_mode()
    return {
        "status": "ok",
        "model_loaded": True,
        "search_mode": search_mode,
        "cache": "disabled (waiting for redis)"
    }

def generate_query_embedding(query: str) -> List[float]:
    """
    Convert user query to a 1024-dimensional vector using Cohere API.
    """
    print(f"\n🔍 Generating Cohere embedding for query: '{query}'")
    try:
        embeddings = _get_cohere_embeddings([query], input_type="search_query")
        embedding = embeddings[0]
        print(f"   ✓ Generated {len(embedding)}-dim vector")
        print(f"   Sample values: {embedding[0]:.6f}, {embedding[1]:.6f}, {embedding[2]:.6f}")
        return embedding
    except Exception as e:
        print(f"❌ Failed to generate query embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Query embedding generation failed: {str(e)}")

@app.post("/api/v1/ai/retrieve", response_model=RetrieveResponse)
async def retrieve_context(request: RetrieveRequest):
    """
    Main RAG retrieval endpoint.
    
    1. Generates a real embedding from the user's query
    2. Searches for similar document chunks (pgvector or JSON fallback)
    3. Returns top-k most relevant chunks for the Orchestrator to use
    """
    try:
        # Generate real embedding
        query_vector = generate_query_embedding(request.query)

        # Search for similar chunks
        results_data = search_similar_chunks(
            query_vector=query_vector,
            user_id=request.user_id,
            material_id=request.material_id,
            top_k=request.top_k
        )

        # Convert to Pydantic models
        chunk_results = [ChunkResult(**r) for r in results_data]
        search_mode = get_search_mode()

        return RetrieveResponse(
            query=request.query,
            query_embedding_preview=query_vector[:5],
            results=chunk_results,
            search_mode=search_mode,
            cached=False,
            note=f"Returned {len(chunk_results)} chunks via {search_mode}."
        )

    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5003, reload=True)
