from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import google.generativeai as genai
import os
import uvicorn
from datetime import datetime, timedelta
import uuid
import json
import asyncio
import time
import hashlib
from google.api_core.exceptions import ResourceExhausted

# Initialize FastAPI
app = FastAPI(
    title="LexiAssist AI Orchestrator",
    description="Manages Gemini AI calls, prompts, and conversation history",
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

@app.middleware("http")
async def internal_auth_middleware(request: Request, call_next):
    if request.url.path in ("/", "/health"):
        return await call_next(request)
    key = request.headers.get("X-Internal-Key")
    expected = os.getenv("INTERNAL_API_KEY", "dev-internal-key")
    if key != expected:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=401, content={"detail": "Invalid internal key"})
    return await call_next(request)

# ─── Model Router & Cost Tracking ───────────────────────────────────────

MODEL_PRICING = {
    "gemini-2.5-flash-lite": {"input": 0.00010, "output": 0.00040, "description": "Cheapest, fast"},
    "gemini-2.5-flash": {"input": 0.00030, "output": 0.00250, "description": "Balanced"},
    "gemini-2.5-pro": {"input": 0.00125, "output": 0.01000, "description": "Best quality"},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.00030, "description": "Legacy Flash"},
}

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-2.5-flash-lite")

class ModelRouter:
    def select_model(self, task_type: str, query: str = "", context_chunks: List[str] = None, override: str = None) -> str:
        if override and override in MODEL_PRICING:
            return override

        # Estimate input size (chars ≈ tokens * 4 for English)
        context_len = sum(len(c) for c in (context_chunks or []))
        estimated_input_chars = len(query) + context_len

        # Small, simple tasks → cheapest model
        if estimated_input_chars < 1500 and task_type in ("chat", "generate_summary"):
            return "gemini-2.5-flash-lite"

        # Large context or complex generation → pro
        if estimated_input_chars > 12000 or task_type in ("generate_quiz", "generate_flashcards"):
            # For very large contexts (> ~15k chars ≈ 4k tokens), use pro; otherwise default flash
            if estimated_input_chars > 15000:
                return "gemini-2.5-pro"
            return DEFAULT_MODEL

        # Default balanced model
        return DEFAULT_MODEL

router = ModelRouter()

_user_costs: Dict[str, List[Dict]] = {}

# Simple in-memory response cache: {hash: (timestamp, response_data)}
_response_cache: Dict[str, tuple] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes

def _get_cache_key(prompt: str, model_name: str) -> str:
    return hashlib.sha256(f"{model_name}:{prompt}".encode()).hexdigest()

def _get_cached_response(cache_key: str):
    if cache_key not in _response_cache:
        return None
    timestamp, data = _response_cache[cache_key]
    if time.time() - timestamp > CACHE_TTL_SECONDS:
        del _response_cache[cache_key]
        return None
    return data

def _set_cached_response(cache_key: str, data):
    _response_cache[cache_key] = (time.time(), data)

def _calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model_name)
    if not pricing:
        # Fallback to default model or gemini-2.5-flash if default model is not configured
        fallback_model = DEFAULT_MODEL if DEFAULT_MODEL in MODEL_PRICING else "gemini-2.5-flash"
        pricing = MODEL_PRICING.get(fallback_model, MODEL_PRICING["gemini-2.5-flash"])
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    return input_cost + output_cost

def _track_cost(user_id: str, task_type: str, model_name: str, input_tokens: int, output_tokens: int, cost_usd: float):
    if user_id not in _user_costs:
        _user_costs[user_id] = []
    _user_costs[user_id].append({
        "timestamp": datetime.now().isoformat(),
        "task_type": task_type,
        "model": model_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
    })
    # Cap at last 1000 entries
    if len(_user_costs[user_id]) > 1000:
        _user_costs[user_id] = _user_costs[user_id][-1000:]

# ─── Configure Gemini AI (required) ─────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is required. "
        "Get one from https://aistudio.google.com/app/apikey"
    )

genai.configure(api_key=GEMINI_API_KEY)
_model_cache: Dict[str, genai.GenerativeModel] = {}

def _get_model(model_name: str) -> genai.GenerativeModel:
    if model_name not in _model_cache:
        _model_cache[model_name] = genai.GenerativeModel(model_name)
    return _model_cache[model_name]

print(f"✅ Gemini AI configured (model: {MODEL_NAME})")

# ─── Conversation History (in-memory with TTL) ──────────────────────────

conversation_history: Dict[str, List[Dict]] = {}
conversation_last_active: Dict[str, datetime] = {}
CONVERSATION_TTL = timedelta(hours=1)
MAX_CONVERSATIONS = 1000


def _evict_stale_conversations():
    """Remove conversations idle for more than CONVERSATION_TTL."""
    now = datetime.now()
    stale = [cid for cid, ts in conversation_last_active.items() if now - ts > CONVERSATION_TTL]
    for cid in stale:
        conversation_history.pop(cid, None)
        conversation_last_active.pop(cid, None)
    # Hard cap: if still too many, remove oldest
    if len(conversation_history) > MAX_CONVERSATIONS:
        sorted_convos = sorted(conversation_last_active.items(), key=lambda x: x[1])
        for cid, _ in sorted_convos[:len(sorted_convos) - MAX_CONVERSATIONS]:
            conversation_history.pop(cid, None)
            conversation_last_active.pop(cid, None)


# ─── Pydantic Models ────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., description="User's question", max_length=10000)
    user_id: str = Field(..., description="For conversation history")
    material_id: Optional[str] = Field(None, description="Specific document context")
    context_chunks: List[str] = Field(default=[], description="Retrieved text chunks from Retrieval Service")
    conversation_id: Optional[str] = Field(None, description="Continue existing conversation")
    model: Optional[str] = Field(None, description="Override model: gemini-2.5-flash-lite, gemini-2.5-flash, gemini-2.5-pro")

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    tokens_used: int
    model: str
    model_used: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    sources: List[str]

class ContextItem(BaseModel):
    text: str
    source: str
    relevance_score: float


# ─── Health Check ────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "ai-orchestrator",
        "port": 5005,
        "version": "3.0.0",
        "ai_model": MODEL_NAME,
    }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "conversations_active": len(conversation_history)
    }


# ─── Prompt Builder ─────────────────────────────────────────────────────

def build_prompt(query: str, context_chunks: List[str], chat_history: List[Dict]) -> str:
    """Build a smart prompt for Gemini with context and history."""
    has_context = bool(context_chunks)
    
    if has_context:
        context_text = "\n\n".join([
            f"[Document {i+1}]\n{chunk}"
            for i, chunk in enumerate(context_chunks[:5])
        ])
        context_section = f"""DOCUMENT CONTEXT:
{context_text}

IMPORTANT: Use the above document context to answer the question. If the answer is not in the documents, say "I don't have enough information in the provided documents."""
    else:
        context_section = """NO DOCUMENTS UPLOADED:
Answer this question using your general knowledge. The user has not uploaded any study materials, so provide a helpful, accurate answer based on what you know."""

    history_text = ""
    if chat_history:
        history_text = "\n\nPrevious conversation:\n"
        for msg in chat_history[-3:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

    prompt = f"""You are LexiAssist, a helpful AI study assistant for students.

{context_section}
{history_text}

USER QUESTION: {query}

INSTRUCTIONS:
- Be concise but thorough
- Format your response using clean, professional Markdown
- Use **bold text** for section headers and key terms (e.g., **Summary:**, **Key Points:**)
- Use bullet points (*) for lists of items or facts
- Use numbered lists (1. 2. 3.) for sequential steps or ranked items
- Add blank lines between paragraphs and sections for readability
- For document-based questions: cite which document [Document X]
- For general knowledge: provide accurate, helpful information
- Start with a brief overview/summary when appropriate
- Group related information under clear, bold subheadings
- Never use wall-of-text paragraphs; always break into digestible chunks

Your response:"""

    return prompt


async def call_gemini(prompt: str, model_name: str = None, task_type: str = "unknown", user_id: str = None) -> tuple:
    """Call Gemini with retry, caching, and return (response_text, input_tokens, output_tokens, cost_usd, model_used)."""
    target_model = model_name or MODEL_NAME
    
    # Check cache first
    cache_key = _get_cache_key(prompt, target_model)
    cached = _get_cached_response(cache_key)
    if cached:
        print(f"♻️  Cache hit | {task_type} | {target_model}")
        return cached
    
    model = _get_model(target_model)
    
    # Retry with exponential backoff for 429 rate limit errors
    max_retries = 3
    base_delay = 2.0
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            break
        except ResourceExhausted as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"⏳ Rate limit hit (429). Retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)
            else:
                print(f"❌ Rate limit exceeded after {max_retries} retries: {e}")
                raise HTTPException(status_code=429, detail="AI service rate limit exceeded. Please try again in a moment.")
        except Exception as e:
            print(f"❌ Gemini API error: {e}")
            raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")
    
    input_tokens = 0
    output_tokens = 0
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
        output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
    
    cost_usd = _calculate_cost(target_model, input_tokens, output_tokens)
    if user_id:
        _track_cost(user_id, task_type, target_model, input_tokens, output_tokens, cost_usd)
    
    result = (response.text, input_tokens, output_tokens, cost_usd, target_model)
    _set_cached_response(cache_key, result)
    
    print(f"💰 {task_type} | {target_model} | ${cost_usd:.6f} | {input_tokens}+{output_tokens} tokens")
    return result


# ─── Chat Endpoint ──────────────────────────────────────────────────────

@app.post("/api/v1/ai/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint — receives query + context, returns AI response."""
    conversation_id = request.conversation_id or str(uuid.uuid4())

    # Evict stale conversations
    _evict_stale_conversations()

    if conversation_id not in conversation_history:
        conversation_history[conversation_id] = []

    chat_history = conversation_history[conversation_id]
    prompt = build_prompt(request.query, request.context_chunks, chat_history)

    print(f"\n🤖 Processing chat for user {request.user_id}")
    print(f"   Query: {request.query[:50]}...")
    print(f"   Context chunks: {len(request.context_chunks)}")
    print(f"   History length: {len(chat_history)}")

    model_name = router.select_model("chat", request.query, request.context_chunks, request.model)

    try:
        ai_response, input_tokens, output_tokens, cost_usd, model_used = await call_gemini(
            prompt, model_name=model_name, task_type="chat", user_id=request.user_id
        )

        # Update conversation history
        chat_history.append({
            "role": "user",
            "content": request.query,
            "timestamp": datetime.now().isoformat()
        })
        chat_history.append({
            "role": "assistant",
            "content": ai_response,
            "timestamp": datetime.now().isoformat()
        })

        # Keep only last 10 messages
        conversation_history[conversation_id] = chat_history[-10:]
        conversation_last_active[conversation_id] = datetime.now()

        total_tokens = input_tokens + output_tokens
        print(f"   ✅ Response generated ({len(ai_response)} chars, {total_tokens} tokens)")

        return ChatResponse(
            response=ai_response,
            conversation_id=conversation_id,
            tokens_used=total_tokens,
            model=MODEL_NAME,
            model_used=model_used,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            sources=[f"chunk_{i}" for i in range(len(request.context_chunks))]
        )

    except Exception as e:
        print(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")


@app.get("/api/v1/ai/conversation/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """Retrieve conversation history."""
    if conversation_id not in conversation_history:
        return {"conversation_id": conversation_id, "messages": []}
    return {
        "conversation_id": conversation_id,
        "messages": conversation_history[conversation_id]
    }

@app.delete("/api/v1/ai/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Clear conversation history."""
    if conversation_id in conversation_history:
        del conversation_history[conversation_id]
        conversation_last_active.pop(conversation_id, None)
        return {"message": "Conversation cleared"}
    return {"message": "Conversation not found"}


# ─── Streaming Chat Endpoint ──────────────────────────────────────────────

@app.post("/api/v1/ai/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat responses for real-time display."""
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    # Evict stale conversations
    _evict_stale_conversations()
    
    if conversation_id not in conversation_history:
        conversation_history[conversation_id] = []
    
    chat_history = conversation_history[conversation_id]
    prompt = build_prompt(request.query, request.context_chunks, chat_history)
    
    print(f"\n🤖 Processing streaming chat for user {request.user_id}")
    print(f"   Query: {request.query[:50]}...")
    print(f"   Context chunks: {len(request.context_chunks)}")
    
    model_name = router.select_model("chat", request.query, request.context_chunks, request.model)
    model = _get_model(model_name)
    
    async def generate_stream():
        try:
            # Start with conversation_id event
            yield f"data: {json.dumps({'conversation_id': conversation_id})}\n\n"
            
            # Call Gemini with streaming
            response = model.generate_content(prompt, stream=True)
            
            full_response = ""
            input_tokens = 0
            output_tokens = 0
            
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    # Send each chunk as SSE
                    yield f"data: {json.dumps({'token': chunk.text})}\n\n"
                    # Small delay to prevent overwhelming the client
                    await asyncio.sleep(0.01)
            
            # Get token count if available
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
            
            cost_usd = _calculate_cost(model_name, input_tokens, output_tokens)
            if request.user_id:
                _track_cost(request.user_id, "chat_stream", model_name, input_tokens, output_tokens, cost_usd)
            
            # Update conversation history
            chat_history.append({
                "role": "user",
                "content": request.query,
                "timestamp": datetime.now().isoformat()
            })
            chat_history.append({
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep only last 10 messages
            conversation_history[conversation_id] = chat_history[-10:]
            conversation_last_active[conversation_id] = datetime.now()
            
            # Send completion event
            yield f"data: {json.dumps({{
                'complete': True,
                'conversation_id': conversation_id,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'cost_usd': cost_usd,
                'model': MODEL_NAME,
                'model_used': model_name,
                'sources': [f"chunk_{i}" for i in range(len(request.context_chunks))]
            }})}\n\n"
            
            # End of stream
            yield "data: [DONE]\n\n"
            
            print(f"   ✅ Stream complete ({len(full_response)} chars)")
            
        except Exception as e:
            print(f"❌ Streaming chat error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# ─── Generate Endpoints ─────────────────────────────────────────────────

@app.post("/api/v1/ai/generate/quiz")
async def generate_quiz(request: ChatRequest):
    """Generate a quiz from content using RAG context chunks."""
    try:
        context_section = ""
        if request.context_chunks:
            context_section = "Based on these document excerpts:\n\n" + "\n\n".join(
                f"[Section {i+1}] {chunk}" for i, chunk in enumerate(request.context_chunks[:5])
            ) + "\n\n"

        prompt = f"""{context_section}Generate a 5-question multiple choice quiz about the following topic.

Topic/Content: {request.query}

For each question:
1. Write a clear question
2. Provide 4 options labeled A, B, C, D
3. Mark the correct answer
4. Add a brief explanation

Format as valid JSON with this structure:
{{
    "questions": [
        {{
            "id": "q1",
            "question": "...",
            "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
            "correct_answer": "B",
            "explanation": "..."
        }}
    ]
}}"""

        model_name = router.select_model("generate_quiz", request.query, request.context_chunks, request.model)
        ai_response, input_tokens, output_tokens, cost_usd, model_used = await call_gemini(
            prompt, model_name=model_name, task_type="generate_quiz", user_id=request.user_id
        )

        return {
            "quiz": ai_response,
            "type": "quiz",
            "model": MODEL_NAME,
            "model_used": model_used,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "tokens_used": input_tokens + output_tokens,
            "context_chunks_used": len(request.context_chunks)
        }

    except Exception as e:
        print(f"❌ Quiz generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Quiz generation failed: {str(e)}")


@app.post("/api/v1/ai/generate/summary")
async def generate_summary(request: ChatRequest):
    """Generate a summary of content using RAG context chunks."""
    try:
        context_section = ""
        if request.context_chunks:
            context_section = "Document content to summarize:\n\n" + "\n\n".join(
                f"[Section {i+1}] {chunk}" for i, chunk in enumerate(request.context_chunks[:5])
            ) + "\n\n"

        prompt = f"""{context_section}Provide a concise but comprehensive summary of the following content.

Content/Topic: {request.query}

Include:
- Key concepts and main ideas
- Important details and relationships
- Any conclusions or implications

Write in clear, student-friendly language with bullet points for key takeaways."""

        model_name = router.select_model("generate_summary", request.query, request.context_chunks, request.model)
        ai_response, input_tokens, output_tokens, cost_usd, model_used = await call_gemini(
            prompt, model_name=model_name, task_type="generate_summary", user_id=request.user_id
        )

        return {
            "summary": ai_response,
            "type": "summary",
            "model": MODEL_NAME,
            "model_used": model_used,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "tokens_used": input_tokens + output_tokens,
            "context_chunks_used": len(request.context_chunks)
        }

    except Exception as e:
        print(f"❌ Summary generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")


@app.post("/api/v1/ai/generate/flashcards")
async def generate_flashcards(request: ChatRequest):
    """Generate flashcards from content using RAG context chunks."""
    try:
        context_section = ""
        if request.context_chunks:
            context_section = "Source material:\n\n" + "\n\n".join(
                f"[Section {i+1}] {chunk}" for i, chunk in enumerate(request.context_chunks[:5])
            ) + "\n\n"

        prompt = f"""{context_section}Create 5 flashcards (question and answer format) from the following content.

Content/Topic: {request.query}

For each flashcard:
- Front: A clear, specific question
- Back: A concise but complete answer

Format as valid JSON:
{{
    "flashcards": [
        {{
            "id": "f1",
            "front": "What is...?",
            "back": "It is..."
        }}
    ]
}}"""

        model_name = router.select_model("generate_flashcards", request.query, request.context_chunks, request.model)
        ai_response, input_tokens, output_tokens, cost_usd, model_used = await call_gemini(
            prompt, model_name=model_name, task_type="generate_flashcards", user_id=request.user_id
        )

        return {
            "flashcards": ai_response,
            "type": "flashcards",
            "model": MODEL_NAME,
            "model_used": model_used,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "tokens_used": input_tokens + output_tokens,
            "context_chunks_used": len(request.context_chunks)
        }

    except Exception as e:
        print(f"❌ Flashcards generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Flashcards generation failed: {str(e)}")


# ─── Analytics / Cost Endpoints ─────────────────────────────────────────

@app.get("/analytics/costs")
async def get_system_costs():
    """Admin: total costs across all users today."""
    today = datetime.now().date().isoformat()
    total = 0.0
    for entries in _user_costs.values():
        for entry in entries:
            if entry["timestamp"].startswith(today):
                total += entry["cost_usd"]
    return {
        "date": today,
        "total_cost_usd": round(total, 6),
        "total_users": len(_user_costs),
    }

@app.get("/analytics/costs/{user_id}")
async def get_user_costs(user_id: str):
    """Per-user cost breakdown."""
    entries = _user_costs.get(user_id, [])
    today = datetime.now().date().isoformat()
    today_total = sum(e["cost_usd"] for e in entries if e["timestamp"].startswith(today))
    return {
        "user_id": user_id,
        "total_entries": len(entries),
        "today_cost_usd": round(today_total, 6),
        "entries": entries[-100:],  # Return last 100 entries
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5005, reload=True)
