# lexiassist-ai-main/queue.py
"""Redis-backed async task queue for the AI Monolith."""
import json
import os
import uuid
from datetime import datetime, timezone

import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
_redis_client = None


def _get_redis():
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def enqueue_job(task_type, payload, user_id, max_retries: int = 2) -> str:
    """Push a new job onto the pending queue and persist its metadata."""
    r = _get_redis()
    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    job_data = {
        "status": "pending",
        "task_type": task_type,
        "payload": json.dumps(payload),
        "user_id": user_id,
        "created_at": now,
        "updated_at": now,
        "result": "",
        "error": "",
        "progress": "0",
        "progress_message": "",
        "session_id": "",
        "retry_count": "0",
        "max_retries": str(max_retries),
    }
    r.hset(f"ai:job:{job_id}", mapping=job_data)
    r.lpush("ai:queue:pending", job_id)
    return job_id


def get_job_status(job_id) -> dict | None:
    """Return the full job hash or None if missing."""
    r = _get_redis()
    data = r.hgetall(f"ai:job:{job_id}")
    if not data:
        return None
    if data.get("payload"):
        try:
            data["payload"] = json.loads(data["payload"])
        except json.JSONDecodeError:
            pass
    return dict(data)


def dequeue_job(timeout=5) -> dict | None:
    """Blocking pop from the pending queue. Returns job dict or None."""
    r = _get_redis()
    result = r.brpop("ai:queue:pending", timeout=timeout)
    if result is None:
        return None
    job_id = result[1]
    data = r.hgetall(f"ai:job:{job_id}")
    if not data:
        return None
    data["job_id"] = job_id
    if data.get("payload"):
        try:
            data["payload"] = json.loads(data["payload"])
        except json.JSONDecodeError:
            pass
    return dict(data)


def update_job_status(
    job_id,
    status,
    result=None,
    error=None,
    progress=None,
    progress_message=None,
    session_id=None,
):
    """Patch one or more fields on an existing job hash."""
    r = _get_redis()
    key = f"ai:job:{job_id}"
    updates = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
    }
    if result is not None:
        updates["result"] = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
    if error is not None:
        updates["error"] = str(error)
    if progress is not None:
        updates["progress"] = str(progress)
    if progress_message is not None:
        updates["progress_message"] = progress_message
    if session_id is not None:
        updates["session_id"] = session_id
    r.hset(key, mapping=updates)


def requeue_job(job_id: str) -> bool:
    """Requeue a failed job for retry. Returns True if requeued, False if max retries exceeded."""
    r = _get_redis()
    key = f"ai:job:{job_id}"
    data = r.hgetall(key)
    if not data:
        return False
    retry_count = int(data.get("retry_count", "0"))
    max_retries = int(data.get("max_retries", "2"))
    if retry_count >= max_retries:
        # Move to dead-letter queue
        r.hset(key, mapping={
            "status": "failed_permanently",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        r.lpush("ai:queue:dead_letter", job_id)
        return False
    r.hset(key, mapping={
        "status": "pending",
        "retry_count": str(retry_count + 1),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    r.lpush("ai:queue:pending", job_id)
    return True


def get_job_result(job_id) -> dict | None:
    """Return the job hash with the result field deserialized."""
    r = _get_redis()
    data = r.hgetall(f"ai:job:{job_id}")
    if not data:
        return None
    result_raw = data.get("result", "")
    if result_raw:
        try:
            data["result"] = json.loads(result_raw)
        except json.JSONDecodeError:
            data["result"] = result_raw
    return dict(data)
