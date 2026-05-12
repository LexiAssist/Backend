# reading_assistant/reading_graph.py
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
import json, base64
import uuid
import sys
import os
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lexicore import LexiEngine
import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter
from google import genai
from google.genai import types
import time
import logging

logger = logging.getLogger(__name__)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

class ReadingState(TypedDict):
    document_text: str
    summary: str
    vocab_terms: list[dict]   # [{term, definition, context_snippet}]
    tts_audio_b64: str
    tts_config: dict          # {voice, speed, pitch}
    stored_doc_id: str
    summary_type: Optional[str]  # "detailed", "concise", or "brief"
    audio: Optional[bytes]


COLLECTION_NAME = 'New_reading_docs'
_weaviate_client = None


def _get_weaviate_client():
    """Lazy singleton — connects to Weaviate on first call."""
    global _weaviate_client
    if _weaviate_client is not None:
        return _weaviate_client

    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    if not weaviate_url:
        raise RuntimeError("WEAVIATE_URL env var is not set")

    headers = {}
    cohere_key = os.getenv("COHERE_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    if cohere_key:
        headers["X-Cohere-Api-Key"] = cohere_key
    if google_key:
        headers["X-Goog-Studio-Api-Key"] = google_key

    if weaviate_api_key:
        _weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key),
            headers=headers,
        )
    else:
        _weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            headers=headers,
        )

    if not _weaviate_client.collections.exists(COLLECTION_NAME):
        _weaviate_client.collections.create(
            name=COLLECTION_NAME,
            generative_config=Configure.Generative.google_gemini(
                model="gemini-2.5-flash"
            ),
            properties=[
                Property(name="chunk_text", data_type=DataType.TEXT),
                Property(name="doc_id", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
            ],
        )

    logger.info("Connected to Weaviate Cloud")
    return _weaviate_client
    



class ReaadingEngine:
    def __init__(self):
        from .tts_engine import TTSGenerator
        self.tts_generator = TTSGenerator()

    def store_document(self, state: ReadingState) -> ReadingState:
        """Embed and store the document in Weaviate for RAG reuse."""
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        client = _get_weaviate_client()
        docs = client.collections.use(COLLECTION_NAME)
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        
        # Chunk and store
        chunks = LexiEngine.chunk_text(state["document_text"])
        doc_id = str(uuid.uuid4())
        
        vectors = embeddings.embed_documents(chunks)
        
        with docs.batch.stream() as batch:
            for i, chunk in enumerate(chunks):
                obj_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}::chunk::{i}")
                batch.add_object(
                    uuid=obj_uuid,
                    properties={
                        "chunk_text": chunk,
                        "doc_id": doc_id,
                        "chunk_index": i
                    },
                    vector=vectors[i]
                )

        if len(docs.batch.failed_objects) > 0:
            time.sleep(5)
            print(f"Failed to store {len(docs.batch.failed_objects)} chunks")

            for failed in docs.batch.failed_objects:
                print(f"{failed.message}")
            #     print(f"{failed.object_.properties}")

        state["stored_doc_id"] = doc_id
        return state

    def generate_summary(self, state: ReadingState) -> ReadingState:
        doc_id = state.get("stored_doc_id")
        summary_type = state.get("summary_type", "concise").lower()

        if not doc_id:
            raise ValueError("Error: Document not stored properly.")
        
        client = _get_weaviate_client()
        
        # If Weaviate is available, retrieve representative chunks from it
        if client is not None:
            docs = client.collections.use(COLLECTION_NAME)
            filtered_chunks = docs.query.fetch_objects(
                filters=Filter.by_property("doc_id").equal(doc_id),
                include_vector=False
            )
            if not filtered_chunks.objects:
                state["summary"] = "Error: No document chunks found for summarization."
                return state
            
            sorted_chunks = sorted(filtered_chunks.objects, key=lambda x: x.properties.get("chunk_index"))
            chunk_texts = [x.properties.get("chunk_text", "") for x in sorted_chunks]
            selected = self._select_summary_chunks(chunk_texts, max_chunks=12)
        else:
            # Weaviate not available — chunk original document text
            logger.info("Weaviate not available — using chunked original document text")
            doc_text = state.get("document_text", "")
            chunks = []
            start = 0
            n = len(doc_text)
            while start < n:
                end = min(start + 1200, n)
                chunks.append(doc_text[start:end])
                if end == n:
                    break
                start = end - 150
            selected = self._select_summary_chunks(chunks, max_chunks=12)
    
        context = "\n\n---\n\n".join(f"[Excerpt {i+1}] {c}" for i, c in enumerate(selected))
        
        max_chars = {"brief": 4000, "concise": 6000, "detailed": 8000}.get(summary_type, 6000)

        system = SystemMessage(content=f"""
        You are an academic summariser. 
        Produce a clear, structured summary suitable for being read aloud as an audio overview.
        Use natural spoken language, not written academic style. 
        Adjust length of summary based on summary type: {summary_type} (either Detailed, Concise, or Brief). 
        Organise into: main topics, key points and explanations, key arguments/findings, conclusion.
        """)

        human = HumanMessage(
            content=f"Summarise the following excerpts from an academic text. Provide a {summary_type} summary suitable for audio narration:\n\n{context[:max_chars]}"
        )

        response = llm.invoke([system, human])
        state["summary"] = response.content
        state["summary_type"] = summary_type
        return state
    
    def extract_vocab(self, state: ReadingState) -> ReadingState:
        system = SystemMessage(content="""You are an academic vocabulary assistant.
        Extract complex or domain-specific terms from the text and their meanings/explanations. 
        Return ONLY a valid JSON array with objects: 
        {"term": str, "definition": str, "context_snippet": str (max 50 words from text)}
        Include 5-15 terms. No markdown fences.""")

        # Context compression: use first ~8000 chars instead of full document
        text_sample = state["document_text"][:8000]
        human = HumanMessage(content=f"Extract vocabulary from:\n\n{text_sample}")
        response = llm.invoke([system, human])
        
        try:
            state["vocab_terms"] = json.loads(response.content)
        except json.JSONDecodeError:
            state["vocab_terms"] = []
        return state

    def _select_summary_chunks(self, chunks: list[str], max_chunks: int = 12) -> list[str]:
        """
        Select a representative subset of chunks for summarisation.
        Strategy: keep first + last + evenly spaced middle chunks.
        This preserves narrative arc while capping token usage.
        """
        if len(chunks) <= max_chunks:
            return chunks

        selected = [chunks[0]]  # introduction
        remaining = max_chunks - 2
        if remaining > 0:
            step = max(1, (len(chunks) - 2) // remaining)
            for i in range(1, len(chunks) - 1, step):
                selected.append(chunks[i])
                if len(selected) >= max_chunks - 1:
                    break
        selected.append(chunks[-1])  # conclusion
        return selected[:max_chunks]

    def synthesise_tts(self, state: ReadingState, include_metadata: bool = True, temperature=1) -> ReadingState:
                
        summary= state.get("summary", "No summary available for TTS.")
        cfg = state.get("tts_config", {"voice": "Zephyr", "speed": 1.0, "pitch": 0.0, "speaker_label": "Reader"})
        
        print(f"TTS Config: {cfg}")
        print(f"generating audio with voice: {cfg['voice']}, speed: {cfg['speed']}, pitch: {cfg['pitch']}")

        audio_result= self.tts_generator.generate_audio(
            text=summary,
            voice=cfg.get("voice", "Zephyr"),
            speaker_label=cfg.get("speaker_label", "Reader"),
            temperature=temperature

        )

        raw_bytes=audio_result.get("audio_data",b"")
        state["audio"] = audio_result
        state["tts_audio_b64"] = base64.b64encode(raw_bytes).decode("utf-8")

        print(f"audio generation complete: {audio_result['audio_path']}")

        return state
    
# Build reading graph — sequential but could be parallelised with Send API
reading= ReaadingEngine()
reading_graph_builder = StateGraph(ReadingState)
reading_graph_builder.add_node("store_document", reading.store_document)
reading_graph_builder.add_node("generate_summary", reading.generate_summary)
reading_graph_builder.add_node("extract_vocab", reading.extract_vocab)
reading_graph_builder.add_node("synthesise_tts", reading.synthesise_tts)

reading_graph_builder.set_entry_point("store_document")
reading_graph_builder.add_edge("store_document", "generate_summary")
reading_graph_builder.add_edge("generate_summary", "extract_vocab")
reading_graph_builder.add_edge("extract_vocab", "synthesise_tts")
reading_graph_builder.add_edge("synthesise_tts", END)
reading_graph = reading_graph_builder.compile()