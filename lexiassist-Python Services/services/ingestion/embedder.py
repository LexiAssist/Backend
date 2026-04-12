import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

from sentence_transformers import SentenceTransformer
import json
import numpy as np
from typing import List, Dict

# Load the embedding model (384 dimensions for all-MiniLM-L6-v2)
# First run downloads ~80MB model, then it's cached
print("Loading AI embedding model... (one-time download, ~400MB)")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!")

def generate_embeddings(chunks: List[Dict]) -> List[Dict]:
    """
    Converts text chunks into 384-dimensional vectors using all-MiniLM-L6-v2.

    Args:
        chunks: List of dicts with 'text' key

    Returns:
        Same list with 'embedding' key added (384 floats per chunk)
    """
    # Extract just the text from all chunks
    texts = [chunk["text"] for chunk in chunks]

    print(f"Generating embeddings for {len(texts)} chunks...")

    # Batch encode (faster than one-by-one)
    embeddings = model.encode(texts, show_progress_bar=True)

    # Add embeddings back to chunks
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()  # Convert numpy to list for JSON
        # Store dimension count for verification
        chunk["embedding_dim"] = len(embeddings[i])

    print(f"✅ Generated {len(chunks)} embeddings ({len(embeddings[0]) if chunks else 384} dimensions each)")
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
