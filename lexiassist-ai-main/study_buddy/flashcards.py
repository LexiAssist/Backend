# study_tools/flashcard_graph.py
import json
from typing import TypedDict
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

GOOGLE_API_KEY= os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4, api_key = GOOGLE_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

class FlashcardState(TypedDict):
    document_text: str
    num_cards: int
    flashcards: list[dict]   # [{front, back, topic}]


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────

def generate_flashcards(state: FlashcardState) -> FlashcardState:
    """
    Generates flashcards strictly from the content of the uploaded document.
    Each card has a front (question/term) and back (answer/definition),
    plus a topic tag so the client can group or filter cards.
    """
    system = SystemMessage(content="""You are an expert study tool that generates flashcards 
strictly from the provided academic text. 

Rules:
- Every flashcard must be directly based on content in the provided text
- Do NOT add outside knowledge or information not present in the text
- Front: a clear question, key term, or concept prompt
- Back: the precise answer, definition, or explanation from the text
- Topic: a short label grouping the card under a subtopic from the text
- Distribute cards evenly across the major topics in the document
- Vary card types: definitions, concept questions, cause-and-effect, examples

Return ONLY a valid JSON array. Each object must have exactly:
{
  "front": str,
  "back": str,
  "topic": str
}
No markdown fences, no preamble, no trailing text — raw JSON array only.""")

    human = HumanMessage(content=f"""Generate exactly {state['num_cards']} flashcards from the following notes.
Every flashcard must come strictly from this text:

{state['document_text']}""")

    response = llm.invoke([system, human])

    try:
        clean = response.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        cards = json.loads(clean)
        # Enforce schema — drop any malformed cards
        state["flashcards"] = [
            c for c in cards
            if all(k in c for k in ("front", "back", "topic"))
        ]
    except json.JSONDecodeError:
        state["flashcards"] = []

    return state


# ─────────────────────────────────────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────────────────────────────────────

flashcard_graph = (
    StateGraph(FlashcardState)
    .add_node("generate_flashcards", generate_flashcards)
    .set_entry_point("generate_flashcards")
    .add_edge("generate_flashcards", END)
    .compile()
)