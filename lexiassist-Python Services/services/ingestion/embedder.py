import os
import httpx
import json
from typing import List, Dict

COHERE_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-multilingual-v3.0")

def _get_cohere_embeddings(texts: List[str], input_type: str = "search_document") -> List[List[float]]:
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

def generate_embeddings(chunks: List[Dict]) -> List[Dict]:
    """
    Converts text chunks into 1024-dimensional vectors using Cohere.

    Args:
        chunks: List of dicts with 'text' key

    Returns:
        Same list with 'embedding' key added (1024 floats per chunk)
    """
    # Extract just the text from all chunks
    texts = [chunk["text"] for chunk in chunks]

    print(f"Generating embeddings for {len(texts)} chunks via Cohere...")

    all_embeddings = []
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = _get_cohere_embeddings(batch, input_type="search_document")
        all_embeddings.extend(batch_embeddings)

    # Add embeddings back to chunks
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = all_embeddings[i]
        # Store dimension count for verification
        chunk["embedding_dim"] = len(all_embeddings[i])

    print(f"✅ Generated {len(chunks)} embeddings ({len(all_embeddings[0]) if all_embeddings else 1024} dimensions each)")
    return chunks

def verify_embeddings(chunks: List[Dict]):
    """Check that embeddings were created correctly"""
    print("\nVerification:")
    print("-" * 50)

    for i in range(min(2, len(chunks))):
        emb = chunks[i]["embedding"]
        print(f"Chunk {i}:")
        print(f"  Text preview: {chunks[i]['text'][:60]}...")
        print(f"  Embedding: {emb[:5]}... (showing 5 of {len(emb)} numbers)")
        print(f"  Sample values: {emb[0]:.6f}, {emb[1]:.6f}, {emb[2]:.6f}")
        print()

# Self-test when run directly
if __name__ == "__main__":
    print("Embedder Module - Test Mode")
    print("=" * 50)

    try:
        # Load the chunks we just created
        with open("chunks.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)

        print(f"Loaded {len(chunks)} chunks from chunks.json")

        # Generate embeddings
        chunks_with_embeddings = generate_embeddings(chunks)

        # Verify
        verify_embeddings(chunks_with_embeddings)

        # Save to file (this is what goes to database later)
        output_file = "chunks_with_embeddings.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks_with_embeddings, f, indent=2)

        print(f"\n💾 Saved chunks with embeddings to: {output_file}")
        print("Each chunk now has: text, index, word_count, embedding (384 floats)")

    except FileNotFoundError:
        print("Error: 'chunks.json' not found!")
        print("Please run chunker.py first.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
