# study_tools/quiz_graph.py
import json
from typing import Literal, TypedDict
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

class QuizState(TypedDict):
    document_text: str
    quiz_type: str          # "multiple_choice" or "theory"
    num_questions: int
    questions: list[dict]


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────

def generate_multiple_choice(state: QuizState) -> QuizState:
    """
    Generates multiple choice questions strictly from the uploaded document.
    Each question has 4 options (A–D), one correct answer, and an explanation.
    """
    system = SystemMessage(content="""You are an expert quiz generator for academic study.
Generate multiple choice questions strictly from the provided text.

Rules:
- Every question must be directly answerable from the provided text
- Do NOT add outside knowledge or information not present in the text
- Each question has exactly 4 options labelled A, B, C, D
- Exactly one option is correct
- Distractors (wrong options) must be plausible but clearly wrong based on the text
- Explanation must cite why the correct answer is right using the text
- Vary difficulty: mix straightforward recall and deeper comprehension questions
- Distribute questions across all major topics in the document

Return ONLY a valid JSON array. Each object must have exactly:
{
  "question": str,
  "options": {"A": str, "B": str, "C": str, "D": str},
  "correct_answer": str,   (one of "A", "B", "C", "D")
  "explanation": str,
  "topic": str
}
No markdown fences, no preamble — raw JSON array only.""")

    human = HumanMessage(content=f"""Generate exactly {state['num_questions']} multiple choice questions 
from the following notes. Every question must come strictly from this text:

{state['document_text']}""")

    response = llm.invoke([system, human])

    try:
        clean = response.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        questions = json.loads(clean)
        state["questions"] = [
            q for q in questions
            if all(k in q for k in ("question", "options", "correct_answer", "explanation", "topic"))
            and isinstance(q["options"], dict)
            and set(q["options"].keys()) == {"A", "B", "C", "D"}
            and q["correct_answer"] in ("A", "B", "C", "D")
        ]
    except json.JSONDecodeError:
        state["questions"] = []

    return state


def generate_theory(state: QuizState) -> QuizState:
    """
    Generates theory/essay questions strictly from the uploaded document.
    Each question includes a model answer and marking guide based on the text.
    """
    system = SystemMessage(content="""You are an expert quiz generator for academic study.
Generate theory/essay questions strictly from the provided text.

Rules:
- Every question must be directly answerable from the provided text
- Do NOT add outside knowledge or information not present in the text
- Questions should require the student to explain, analyse, or discuss concepts from the text
- Model answer must be based strictly on the text content
- Marking guide lists the key points a student must cover to score full marks
- Vary question depth: some short-answer (2–3 sentences), some extended response

Return ONLY a valid JSON array. Each object must have exactly:
{
  "question": str,
  "model_answer": str,
  "marking_guide": [str],   (list of key points required for full marks)
  "marks": int,             (suggested marks: 2 for short, 5–10 for extended)
  "topic": str
}
No markdown fences, no preamble — raw JSON array only.""")

    human = HumanMessage(content=f"""Generate exactly {state['num_questions']} theory questions 
from the following notes. Every question must come strictly from this text:

{state['document_text']}""")

    response = llm.invoke([system, human])

    try:
        clean = response.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        questions = json.loads(clean)
        state["questions"] = [
            q for q in questions
            if all(k in q for k in ("question", "model_answer", "marking_guide", "marks", "topic"))
            and isinstance(q["marking_guide"], list)
        ]
    except json.JSONDecodeError:
        state["questions"] = []

    return state


def route_quiz_type(state: QuizState) -> str:
    return state["quiz_type"]


# ─────────────────────────────────────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────────────────────────────────────

quiz_graph = (
    StateGraph(QuizState)
    .add_node("generate_multiple_choice", generate_multiple_choice)
    .add_node("generate_theory", generate_theory)
    .set_conditional_entry_point(
        route_quiz_type,
        {
            "multiple_choice": "generate_multiple_choice",
            "theory":          "generate_theory",
        },
    )
    .add_edge("generate_multiple_choice", END)
    .add_edge("generate_theory",          END)
    .compile()
)