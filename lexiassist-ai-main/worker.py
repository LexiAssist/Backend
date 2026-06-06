# lexiassist-ai-main/worker.py
"""Background worker that consumes the Redis task queue."""
import base64
import json
import os
import threading
import traceback
import uuid

from langchain_core.messages import HumanMessage, SystemMessage
from shared.llm_utils import get_llm, safe_llm_invoke

from database import SessionLocal, UserSession, SessionType
from job_queue import dequeue_job, update_job_status, requeue_job
from reading_assistant.reading_engine import reading_graph
from reading_assistant.tts_engine import TTSGenerator
from study_buddy.routes import _generate_flashcards_cached, _generate_quiz_cached

llm = get_llm(temperature=0.2, model=os.getenv("DEFAULT_MODEL", "gemini-2.5-flash"))
fallback_llm = get_llm(temperature=0.2, model="gemini-2.5-flash-lite")


class AIWorker:
    """Daemon-thread worker that polls Redis and executes AI tasks."""

    def __init__(self):
        self._thread = None
        self._stop_event = threading.Event()
        self._tts_generator = TTSGenerator()

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            print("AIWorker already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("AI Worker started")

    def stop(self):
        print("AI Worker stopping...")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        print("AI Worker stopped")

    def _run(self):
        import time
        while not self._stop_event.is_set():
            try:
                job = dequeue_job(timeout=2)
                if job is None:
                    continue
                self._process_job(job)
            except Exception as e:
                print(f"Worker loop error: {e}")
                traceback.print_exc()
                # Prevent CPU spike and log flood when Redis/DB are down
                time.sleep(5)

    def _process_job(self, job: dict):
        job_id = job["job_id"]
        task_type = job.get("task_type")
        payload = job.get("payload") or {}
        user_id = job.get("user_id")

        print(f"[{job_id}] Processing {task_type}")
        update_job_status(
            job_id,
            status="processing",
            progress=10,
            progress_message="Starting job...",
        )

        try:
            if task_type == "reading_analyse":
                self._handle_reading_analyse(job_id, payload, user_id)
            elif task_type == "study_flashcards":
                self._handle_study_flashcards(job_id, payload, user_id)
            elif task_type == "study_quiz":
                self._handle_study_quiz(job_id, payload, user_id)
            elif task_type == "writing_notes":
                self._handle_writing_notes(job_id, payload, user_id)
            else:
                raise ValueError(f"Unknown task_type: {task_type}")
        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            print(f"[{job_id}] FAILED: {error_msg}")

            # Attempt retry before marking permanently failed
            requeued = requeue_job(job_id)
            if requeued:
                print(f"[{job_id}] Requeued for retry")
                update_job_status(
                    job_id,
                    status="pending",
                    progress=0,
                    progress_message="Retrying...",
                    error=str(e),
                )
            else:
                print(f"[{job_id}] Max retries exceeded — moved to dead-letter queue")
                update_job_status(
                    job_id,
                    status="failed",
                    progress=0,
                    progress_message="Job failed",
                    error=str(e),
                )

    def _handle_reading_analyse(self, job_id: str, payload: dict, user_id: str):
        document_text = payload["document_text"]
        filename = payload.get("filename", "unknown")
        summary_type = payload.get("summary_type", "concise")
        voice = payload.get("voice", "Zephyr")
        speaker_label = payload.get("speaker_label", "Reader")
        temperature = payload.get("temperature", 1.0)

        update_job_status(
            job_id,
            progress=20,
            progress_message="Generating summary...",
        )

        session_id = str(uuid.uuid4())
        collection_name = f"reading_{session_id.replace('-', '_')}"

        result = reading_graph.invoke(
            {
                "document_text": document_text,
                "summary": "",
                "vocab_terms": [],
                "tts_audio_b64": "",
                "tts_config": {
                    "voice": voice,
                    "speaker_label": speaker_label,
                    "temperature": temperature,
                },
                "stored_doc_id": "",
                "summary_type": summary_type,
                "audio": None,
                "collection_name": collection_name,
            }
        )

        update_job_status(
            job_id,
            progress=80,
            progress_message="Finalizing results...",
        )

        audio_result = result.get("audio") or {}
        raw_bytes = audio_result.get("audio_data", b"")
        mime_type = audio_result.get("mime_type", "audio/wav")
        tts_b64 = base64.b64encode(raw_bytes).decode() if raw_bytes else result.get("tts_audio_b64")
        tts_error = result.get("tts_error")

        vocab_raw = result.get("vocab_terms", [])
        vocab_list = []
        for item in vocab_raw:
            try:
                if all(k in item for k in ("term", "definition", "context_snippet")):
                    vocab_list.append(item)
            except Exception:
                continue

        db = SessionLocal()
        try:
            db_session = UserSession(
                session_id=session_id,
                user_id=user_id,
                session_type=SessionType.reading,
                filename=filename,
                weaviate_collection=collection_name,
                summary=result.get("summary", ""),
                summary_type=result.get("summary_type", summary_type),
                tts_audio_b64=tts_b64,
                vocab_terms=vocab_list,
            )
            db.add(db_session)
            db.commit()
        finally:
            db.close()

        result_data = {
            "session_id": session_id,
            "summary": result.get("summary", ""),
            "summary_type": result.get("summary_type", summary_type),
            "vocab_terms": vocab_list,
            "tts_audio_b64": tts_b64,
            "audio_mime_type": mime_type if tts_b64 else None,
            "voice": voice,
            "tts_error": tts_error,
        }
        update_job_status(
            job_id,
            status="completed",
            progress=100,
            progress_message="Analysis complete",
            result=result_data,
            session_id=session_id,
        )
        print(f"[{job_id}] Completed reading_analyse -> {session_id}")

    def _handle_study_flashcards(self, job_id: str, payload: dict, user_id: str):
        document_text = payload["document_text"]
        num_cards = payload.get("num_cards", 10)
        filename = payload.get("filename", "unknown")

        update_job_status(
            job_id,
            progress=30,
            progress_message="Generating flashcards...",
        )

        result = _generate_flashcards_cached(document_text, num_cards)

        cards_raw = result.get("flashcards", [])
        cards = [c for c in cards_raw if all(k in c for k in ("front", "back", "topic"))]

        session_id = str(uuid.uuid4())

        db = SessionLocal()
        try:
            db.add(
                UserSession(
                    session_id=session_id,
                    user_id=user_id,
                    session_type=SessionType.flashcard,
                    filename=filename,
                    flashcards=cards,
                    num_cards=len(cards),
                )
            )
            db.commit()
        finally:
            db.close()

        update_job_status(
            job_id,
            status="completed",
            progress=100,
            progress_message="Flashcards generated",
            result={
                "session_id": session_id,
                "num_generated": len(cards),
                "flashcards": cards,
            },
            session_id=session_id,
        )
        print(f"[{job_id}] Completed study_flashcards -> {session_id}")

    def _handle_study_quiz(self, job_id: str, payload: dict, user_id: str):
        document_text = payload["document_text"]
        quiz_type = payload.get("quiz_type", "multiple_choice")
        num_questions = payload.get("num_questions", 5)
        filename = payload.get("filename", "unknown")

        update_job_status(
            job_id,
            progress=30,
            progress_message="Generating quiz...",
        )

        result = _generate_quiz_cached(document_text, quiz_type, num_questions)

        questions_raw = result.get("questions", [])

        if quiz_type == "multiple_choice":
            questions = [
                q
                for q in questions_raw
                if all(k in q for k in ("question", "options", "correct_answer", "explanation", "topic"))
                and isinstance(q.get("options"), dict)
                and set(q["options"].keys()) == {"A", "B", "C", "D"}
                and q["correct_answer"] in ("A", "B", "C", "D")
            ]
        else:
            questions = [
                q
                for q in questions_raw
                if all(k in q for k in ("question", "model_answer", "marking_guide", "marks", "topic"))
                and isinstance(q.get("marking_guide"), list)
            ]

        session_id = str(uuid.uuid4())

        db = SessionLocal()
        try:
            db.add(
                UserSession(
                    session_id=session_id,
                    user_id=user_id,
                    session_type=SessionType.quiz,
                    filename=filename,
                    quiz_type=quiz_type,
                    questions=questions,
                    num_questions=len(questions),
                )
            )
            db.commit()
        finally:
            db.close()

        update_job_status(
            job_id,
            status="completed",
            progress=100,
            progress_message="Quiz generated",
            result={
                "session_id": session_id,
                "quiz_type": quiz_type,
                "num_generated": len(questions),
                "questions": questions,
            },
            session_id=session_id,
        )
        print(f"[{job_id}] Completed study_quiz -> {session_id}")

    def _handle_writing_notes(self, job_id: str, payload: dict, user_id: str):
        raw_text = payload.get("raw_text", "")
        subject = payload.get("subject", "General")
        session_id = payload.get("session_id", str(uuid.uuid4()))
        save = payload.get("save", True)

        if not raw_text.strip():
            raise ValueError("raw_text cannot be empty.")

        update_job_status(
            job_id,
            progress=30,
            progress_message="Generating notes...",
        )

        system = SystemMessage(
            content=f"""You are an expert academic note-taker transcribing a live {subject} lecture.

You will receive a raw speech-to-text transcript which may contain:
- Filler words (um, uh, like, you know)
- Misheared words or homophones — use lecture context to correct these
- Repeated or restarted sentences from the speaker
- Incomplete thoughts cut off at chunk recording boundaries

Convert this into well-organised markdown notes.

Format rules:
- ## for main topics, ### for subtopics
- Bullet points for key facts and explanations
- **Bold** all definitions and key terms
- > Blockquote examples or analogies the teacher gives
- [unclear: ...] for anything genuinely unresolvable from context
- Do NOT add any information not present in the transcript
- Do NOT summarise — preserve all content, just clean and organise it"""
        )

        messages = [
            system,
            HumanMessage(content=f"Subject: {subject}\n\nRaw transcript:\n{raw_text}"),
        ]

        response = safe_llm_invoke(llm, messages, fallback_llm=fallback_llm)
        structured_notes = response.content.strip()

        update_job_status(
            job_id,
            progress=80,
            progress_message="Saving notes...",
        )

        if save:
            db = SessionLocal()
            try:
                existing = (
                    db.query(UserSession)
                    .filter(
                        UserSession.session_id == session_id,
                        UserSession.session_type == SessionType.notes,
                    )
                    .first()
                )

                if existing:
                    existing.structured_notes = structured_notes
                    existing.subject = subject
                else:
                    db.add(
                        UserSession(
                            session_id=session_id,
                            user_id=user_id,
                            session_type=SessionType.notes,
                            subject=subject,
                            structured_notes=structured_notes,
                        )
                    )
                db.commit()
            finally:
                db.close()

        update_job_status(
            job_id,
            status="completed",
            progress=100,
            progress_message="Notes generated",
            result={
                "session_id": session_id,
                "structured_notes": structured_notes,
                "subject": subject,
            },
            session_id=session_id,
        )
        print(f"[{job_id}] Completed writing_notes -> {session_id}")
