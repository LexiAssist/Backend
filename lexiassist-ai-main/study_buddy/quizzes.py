# study_tools/quiz_graph.py
import json
import logging
from typing import Literal, TypedDict
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langgraph.graph import END, StateGraph
from dotenv import load_dotenv
from lexicore import LexiEngine

logger = logging.getLogger(__name__)

load_dotenv()  # Load environment variables from .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4,
    api_key=GOOGLE_API_KEY,
    response_mime_type="application/json",
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    }
)


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

    course_code = state.get("course_code")
    if course_code:
        matches = LexiEngine()._retrieve(
            user_query="important concepts, facts, and details for quiz questions",
            course_code=course_code,
            top_k=8
        )
        context_text = "\n\n".join(m["text"] for m in matches) if matches else state["document_text"][:8000]
    else:
        context_text = state["document_text"][:8000]

    human = HumanMessage(content=f"""Generate exactly {state['num_questions']} multiple choice questions 
from the following notes. Every question must come strictly from this text:

{context_text}""")

    response = llm.invoke([system, human])

    try:
        clean = response.content.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        elif clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()
        
        parsed = json.loads(clean)
        if isinstance(parsed, dict) and "questions" in parsed:
            parsed = parsed["questions"]
            
        valid_questions = []
        if isinstance(parsed, list):
            for q in parsed:
                if not isinstance(q, dict):
                    continue
                if "question" not in q or "options" not in q:
                    continue
                
                options = q["options"]
                if not isinstance(options, dict):
                    continue
                
                # Normalize option keys to uppercase
                normalized_options = {}
                for k, v in options.items():
                    normalized_options[str(k).upper()] = str(v)
                
                # Ensure options contain A, B, C, D
                if not all(k in normalized_options for k in ("A", "B", "C", "D")):
                    logger.warning(f"Multiple choice question missing A, B, C, or D option: {normalized_options.keys()}")
                    continue
                
                correct_ans = str(q.get("correct_answer", "")).upper()
                if correct_ans not in ("A", "B", "C", "D"):
                    # Attempt text match
                    matched = False
                    for opt_key, opt_val in normalized_options.items():
                        if opt_val.strip().lower() == correct_ans.strip().lower():
                            correct_ans = opt_key
                            matched = True
                            break
                    if not matched:
                        logger.warning(f"Multiple choice correct_answer is invalid: {correct_ans}")
                        continue
                
                explanation = q.get("explanation", "")
                if not explanation:
                    explanation = f"The correct answer is {correct_ans}."
                    
                topic = q.get("topic", "General")
                
                valid_questions.append({
                    "question": q["question"],
                    "options": normalized_options,
                    "correct_answer": correct_ans,
                    "explanation": explanation,
                    "topic": topic
                })
        state["questions"] = valid_questions
        logger.info(f"Successfully generated and parsed {len(valid_questions)} multiple choice questions.")
    except Exception as e:
        logger.error(f"Error parsing multiple choice quiz JSON: {e}", exc_info=True)
        logger.error(f"Raw response content: {repr(response.content)}")
        logger.error(f"Raw response metadata: {repr(getattr(response, 'response_metadata', None))}")
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

    course_code = state.get("course_code")
    if course_code:
        matches = LexiEngine()._retrieve(
            user_query="important concepts, arguments, and explanations for theory questions",
            course_code=course_code,
            top_k=8
        )
        context_text = "\n\n".join(m["text"] for m in matches) if matches else state["document_text"][:8000]
    else:
        context_text = state["document_text"][:8000]

    human = HumanMessage(content=f"""Generate exactly {state['num_questions']} theory questions 
from the following notes. Every question must come strictly from this text:

{context_text}""")

    response = llm.invoke([system, human])

    try:
        clean = response.content.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        elif clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()
        
        parsed = json.loads(clean)
        if isinstance(parsed, dict) and "questions" in parsed:
            parsed = parsed["questions"]
            
        valid_questions = []
        if isinstance(parsed, list):
            for q in parsed:
                if not isinstance(q, dict):
                    continue
                if "question" not in q or "model_answer" not in q:
                    continue
                
                marking_guide = q.get("marking_guide", [])
                if not isinstance(marking_guide, list):
                    if isinstance(marking_guide, str):
                        marking_guide = [marking_guide]
                    else:
                        marking_guide = []
                
                marks = q.get("marks", 5)
                try:
                    marks = int(marks)
                except (ValueError, TypeError):
                    marks = 5
                    
                topic = q.get("topic", "General")
                
                valid_questions.append({
                    "question": q["question"],
                    "model_answer": q["model_answer"],
                    "marking_guide": marking_guide,
                    "marks": marks,
                    "topic": topic
                })
        state["questions"] = valid_questions
        logger.info(f"Successfully generated and parsed {len(valid_questions)} theory questions.")
    except Exception as e:
        logger.error(f"Error parsing theory quiz JSON: {e}", exc_info=True)
        logger.error(f"Raw response content: {repr(response.content)}")
        logger.error(f"Raw response metadata: {repr(getattr(response, 'response_metadata', None))}")
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