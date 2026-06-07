import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from pydantic import BaseModel
import os
import sys
import uuid
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our pipeline modules
from parser import extract_text_from_file
from chunker import chunk_text
from embedder import generate_embeddings
from models import save_chunks, init_database
try:
    init_database()
except Exception as e:
    print(f"⚠️ Database initialization failed at startup: {e}")


def verify_internal_key(request: Request, x_internal_key: str = Header(None)):
    if request.url.path in ("/", "/health"):
        return
    expected = os.getenv("INTERNAL_API_KEY", "dev-internal-key-change-in-production")
    if not x_internal_key or x_internal_key != expected:
        raise HTTPException(status_code=403, detail="Invalid or missing internal key")


app = FastAPI(
    title="LexiAssist Ingestion Service",
    description="Document processing pipeline - extracts, chunks, embeds, and stores PDFs",
    version="2.0.0",
    dependencies=[Depends(verify_internal_key)],
)

# Pydantic models
class DocumentProcessRequest(BaseModel):
    material_id: str
    user_id: str
    file_url: str  # Can be S3 URL or local file path for testing

class ProcessResponse(BaseModel):
    task_id: str
    status: str
    message: str
    chunks_created: int = 0
    storage_method: str = "unknown"

# Health check
@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "ingestion",
        "port": 5002,
        "pipeline": "parser → chunker → embedder → storage",
        "version": "2.0.0"
    }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "pipeline_modules": {
            "parser": "loaded",
            "chunker": "loaded",
            "embedder": "loaded",
            "storage": "json_fallback (waiting for postgres)"
        }
    }

def process_pipeline(material_id: str, user_id: str, file_path: str) -> dict:
    """
    THE REAL PIPELINE: PDF → Text → Chunks → Embeddings → Storage
    """
    print(f"\n🚀 Starting pipeline for: {material_id}")
    print(f"   User: {user_id}")
    print(f"   File: {file_path}")

    # Step 1: Extract text from document
    print("\n📄 Step 1: Extracting text...")
    text = extract_text_from_file(file_path)
    if not text:
        raise Exception("No text extracted from document")
    print(f"   ✓ Extracted {len(text)} characters")

    # Step 2: Chunk text
    print("\n✂️ Step 2: Chunking text...")
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    print(f"   ✓ Created {len(chunks)} chunks")

    # Step 3: Generate embeddings
    print("\n🤖 Step 3: Generating AI embeddings...")
    chunks_with_embeddings = generate_embeddings(chunks)
    print(f"   ✓ Generated {len(chunks_with_embeddings)} embeddings")

    # Step 4: Save to storage (JSON for now, DB later)
    print("\n💾 Step 4: Saving to storage...")
    storage_method = save_chunks(chunks_with_embeddings, material_id, user_id)
    print(f"   ✓ Saved using: {storage_method}")

    return {
        "chunks_created": len(chunks),
        "storage_method": storage_method,
        "text_length": len(text)
    }

@app.post("/process", response_model=ProcessResponse)
async def process_document(request: DocumentProcessRequest):
    """
    Process a PDF document through the AI pipeline.

    For testing: Use local file path like "C:\\Users\\USER\\Documents\\file.pdf"
    For production: Will use S3 URLs when docker-compose is ready
    """
    task_id = str(uuid.uuid4())

    try:
        # Check if it's a local file path (for testing)
        if os.path.exists(request.file_url):
            # LOCAL FILE: Process immediately
            print(f"\n🧪 TEST MODE: Processing local file")
            result = process_pipeline(
                request.material_id,
                request.user_id,
                request.file_url
            )

            return ProcessResponse(
                task_id=task_id,
                status="completed",
                message=f"Document processed successfully! Created {result['chunks_created']} chunks.",
                chunks_created=result['chunks_created'],
                storage_method=result['storage_method']
            )

        else:
            # S3/REMOTE URL: Mock for now (until S3 credentials setup)
            print(f"\n☁️ S3 MODE: Would download from {request.file_url}")
            print("   (S3 processing mocked - waiting for S3 credentials)")

            return ProcessResponse(
                task_id=task_id,
                status="queued",
                message="S3 download not yet implemented. Use local file path for testing.",
                chunks_created=0,
                storage_method="mocked"
            )

    except Exception as e:
        print(f"\n❌ Error processing document: {e}")
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """For now, just returns mock status"""
    return {
        "task_id": task_id,
        "status": "completed",
        "note": "Synchronous processing (Redis not connected)"
    }


# Pydantic model for processing from storage
class ProcessFromStorageRequest(BaseModel):
    material_id: str
    user_id: str
    filename: str


@app.post("/process-from-storage")
async def process_from_storage(request: ProcessFromStorageRequest):
    """
    Process a document that has been uploaded to MinIO storage.
    Downloads the file from MinIO, then processes it through the pipeline.
    """
    import httpx
    import tempfile
    
    task_id = str(uuid.uuid4())
    material_id = request.material_id
    user_id = request.user_id
    filename = request.filename
    
    # Construct MinIO URL
    minio_endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
    minio_bucket = os.getenv("MINIO_BUCKET", "lexiassist-materials")
    
    # Use internal Docker network for MinIO
    minio_url = f"http://{minio_endpoint}/{minio_bucket}/materials/{user_id}/{material_id}"
    
    print(f"\n🚀 Processing from storage: {material_id}")
    print(f"   Downloading from: {minio_url}")
    
    try:
        # Download file from MinIO
        response = httpx.get(minio_url, timeout=30.0)
        if response.status_code != 200:
            raise HTTPException(
                status_code=404, 
                detail=f"File not found in storage: {response.status_code}"
            )
        
        # Determine the file extension from the filename parameter
        _, ext = os.path.splitext(filename.lower())
        if not ext:
            ext = ".pdf"

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        
        print(f"   ✓ Downloaded {len(response.content)} bytes")
        
        try:
            # Process the file
            result = process_pipeline(material_id, user_id, tmp_path)
            
            # Notify content service of completion
            await notify_content_service(material_id, "completed", result['chunks_created'])
            
            return ProcessResponse(
                task_id=task_id,
                status="completed",
                message=f"Document processed successfully! Created {result['chunks_created']} chunks.",
                chunks_created=result['chunks_created'],
                storage_method=result['storage_method']
            )
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n❌ Error processing from storage: {e}")
        import traceback
        traceback.print_exc()
        
        # Notify content service of failure
        await notify_content_service(material_id, "failed", 0, str(e))
        
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


async def notify_content_service(material_id: str, status: str, chunks: int = 0, error: str = None):
    """Notify content service about processing status."""
    import httpx
    
    webhook_url = os.getenv("CONTENT_WEBHOOK_URL")
    if not webhook_url:
        print("⚠️ No CONTENT_WEBHOOK_URL configured, skipping notification")
        return
    
    internal_key = os.getenv("INTERNAL_API_KEY", "dev-internal-key")
    
    payload = {
        "material_id": material_id,
        "status": status,
        "chunks_created": chunks,
    }
    if error:
        payload["error"] = error
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                webhook_url,
                json=payload,
                headers={"X-Internal-API-Key": internal_key},
                timeout=10.0
            )
            print(f"   ✓ Notified content service: {response.status_code}")
    except Exception as e:
        print(f"   ⚠️ Failed to notify content service: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5002, reload=True)
