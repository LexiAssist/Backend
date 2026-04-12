from typing import List, Dict

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """
    Breaks text into overlapping chunks for AI processing.

    Args:
        text: The full extracted text
        chunk_size: Number of words per chunk (default 500)
        overlap: Number of words to overlap between chunks (default 50)
                 This ensures context continuity between chunks

    Returns:
        List of dictionaries with: text, index, word_count
    """
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append({
            "text": chunk_text,
            "index": len(chunks),  # 0, 1, 2, 3...
            "word_count": len(chunk_words)
        })

        # Move forward by chunk_size minus overlap
        start += (chunk_size - overlap)

    return chunks

def print_chunk_info(chunks: List[Dict]):
    """Helper to display chunk information"""
    print(f"\nTotal chunks created: {len(chunks)}")
    print("-" * 50)

    for chunk in chunks[:3]:  # Show first 3
        preview = chunk["text"][:100].replace("\n", " ")
        print(f"Chunk {chunk['index']}: {chunk['word_count']} words")
        print(f"  Preview: {preview}...")
        print()

    if len(chunks) > 3:
        print(f"... and {len(chunks) - 3} more chunks")

# Self-test when run directly
if __name__ == "__main__":
    print("Chunker Module - Test Mode")
    print("=" * 50)

    # Test with the text we just extracted
    try:
        with open("extracted_text.txt", "r", encoding="utf-8") as f:
            text = f.read()

        print(f"Loaded text: {len(text)} characters, {len(text.split())} words")

        # Create chunks
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        print_chunk_info(chunks)

        # Save chunks to file for verification
        import json
        with open("chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)

        print(f"\nSaved all chunks to: chunks.json")

    except FileNotFoundError:
        print("Error: 'extracted_text.txt' not found!")
        print("Please run parser.py first to extract text from a PDF.")
    except Exception as e:
        print(f"Error: {e}")
