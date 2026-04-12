import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import uuid
from datetime import datetime
from typing import Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our pipeline modules
from parser import extract_text_from_pdf
from chunker import chunk_text
from embedder import generate_embeddings
from models import save_chunks

app = FastAPI(
    title="LexiAssist Ingestion Service",
    description="Document processing pipeline - extracts, chunks, embeds, and stores PDFs",
    version="3.0.0"
)

# CORS
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ProcessResponse(BaseModel):
    task_id: str
    status: str
    message: str
    chunks_created: int = 0

# Health check
@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "ingestion",
        "port": 5002,
        "pipeline": "parser → chunker → embedder → postgresql",
        "version": "3.0.0"
    }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "storage": "postgresql+pgvector",
        "pipeline_modules": {
            "parser": "loaded",
            "chunker": "loaded",
            "embedder": "loaded",
            "storage": "postgresql"
        }
    }

def process_pipeline(material_id: str, user_id: str, file_path: str) -> dict:
    """
    Full pipeline: PDF → Text → Chunks → Embeddings → PostgreSQL
    """
    print(f"\n🚀 Starting pipeline for: {material_id}")
    print(f"   User: {user_id}")
    print(f"   File: {file_path}")

    # Step 1: Extract text from PDF
    print("\n📄 Step 1: Extracting text...")
    text = extract_text_from_pdf(file_path)
    if not text:
        raise Exception("No text extracted from PDF")
    print(f"   ✓ Extracted {len(text)} characters")

    # Step 2: Chunk text
    print("\n✂️ Step 2: Chunking text...")
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    print(f"   ✓ Created {len(chunks)} chunks")

    # Step 3: Generate embeddings
    print("\n🤖 Step 3: Generating AI embeddings...")
    chunks_with_embeddings = generate_embeddings(chunks)
    print(f"   ✓ Generated {len(chunks_with_embeddings)} embeddings")

    # Step 4: Save to PostgreSQL
    print("\n💾 Step 4: Saving to PostgreSQL...")
    save_chunks(chunks_with_embeddings, material_id, user_id)
    print(f"   ✓ Saved to PostgreSQL")

    return {
        "chunks_created": len(chunks),
        "text_length": len(text)
    }


@app.post("/process", response_model=ProcessResponse)
async def process_document(
    file: UploadFile = File(..., description="PDF file to process"),
    material_id: str = Form(..., description="Unique material identifier"),
    user_id: str = Form(..., description="User identifier"),
):
    """
    Process a PDF document through the AI pipeline.
    Accepts multipart file upload, extracts text, generates embeddings,
    and stores everything in PostgreSQL.
    """
    task_id = str(uuid.uuid4())

    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Save uploaded file to temp location
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            if len(content) == 0:
                raise HTTPException(status_code=400, detail="Empty file uploaded")
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = process_pipeline(material_id, user_id, tmp_path)

            return ProcessResponse(
                task_id=task_id,
                status="completed",
                message=f"Document processed successfully! Created {result['chunks_created']} chunks.",
                chunks_created=result['chunks_created'],
            )
        finally:
            # Clean up temp file
            os.unlink(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        print(f"\n❌ Error processing document: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Check processing task status."""
    return {
        "task_id": task_id,
        "status": "completed",
        "note": "Synchronous processing — task completes immediately"
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
    import requests
    import tempfile
    
    task_id = str(uuid.uuid4())
    material_id = request.material_id
    user_id = request.user_id
    filename = request.filename
    
    # Construct MinIO URL
    minio_endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
    minio_bucket = os.getenv("MINIO_BUCKET", "lexiassist-materials")
    
    # Use internal Docker network for MinIO
    minio_url = f"http://{minio_endpoint}/{minio_bucket}/materials/{material_id}/{filename}"
    
    print(f"\n🚀 Processing from storage: {material_id}")
    print(f"   Downloading from: {minio_url}")
    
    try:
        # Download file from MinIO
        response = requests.get(minio_url, timeout=30)
        if response.status_code != 200:
            raise HTTPException(
                status_code=404, 
                detail=f"File not found in storage: {response.status_code}"
            )
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
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
