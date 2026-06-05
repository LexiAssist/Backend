import hashlib
import json
import os
import asyncio
import logging
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger(__name__)

# Lazy-initialized Redis client
_redis_client = None


def _get_redis():
    global _redis_client
    if _redis_client is None:
        try:
            import redis
            url = os.getenv("REDIS_URL", "redis://localhost:6379")
            _redis_client = redis.from_url(url, decode_responses=True)
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            _redis_client = False  # sentinel: unavailable
    return _redis_client if _redis_client is not False else None


def _is_empty_result(result: Any) -> bool:
    """Check if the result is empty or represents a failed/empty generation."""
    if not result:
        return True
    if isinstance(result, dict):
        # If it's a dictionary, check if the main payload lists are empty
        for key in ("questions", "flashcards", "vocab_terms"):
            if key in result and not result[key]:
                return True
    return False


def ai_cache(namespace: str, ttl: int = 86400):
    """
    Decorator that caches the result of an AI graph invocation in Redis.

    Cache key = ai_cache:{namespace}:sha256(json.dumps(args + kwargs))

    On hit: returns deserialized JSON.
    On miss: executes wrapped function, stores JSON result with expiration, returns it.
    If Redis is unavailable or the result is empty: fails open (does not cache).
    """
    def decorator(func: Callable[..., Any]):
        is_async = asyncio.iscoroutinefunction(func)

        if is_async:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                r = _get_redis()
                key = None
                if r is not None:
                    try:
                        payload = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
                        key = f"ai_cache:{namespace}:{hashlib.sha256(payload.encode()).hexdigest()}"
                        cached = r.get(key)
                        if cached:
                            return json.loads(cached)
                    except Exception as e:
                        logger.warning(f"Redis cache read failed: {e}")

                result = await func(*args, **kwargs)

                if r is not None and key is not None and not _is_empty_result(result):
                    try:
                        r.setex(key, ttl, json.dumps(result, default=str))
                    except Exception as e:
                        logger.warning(f"Redis cache write failed: {e}")

                return result
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                r = _get_redis()
                key = None
                if r is not None:
                    try:
                        payload = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
                        key = f"ai_cache:{namespace}:{hashlib.sha256(payload.encode()).hexdigest()}"
                        cached = r.get(key)
                        if cached:
                            return json.loads(cached)
                    except Exception as e:
                        logger.warning(f"Redis cache read failed: {e}")

                result = func(*args, **kwargs)

                if r is not None and key is not None and not _is_empty_result(result):
                    try:
                        r.setex(key, ttl, json.dumps(result, default=str))
                    except Exception as e:
                        logger.warning(f"Redis cache write failed: {e}")

                return result
            return sync_wrapper

    return decorator
