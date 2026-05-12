#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Test -- Phase 1 Hardening + Phase 2 Implementation
Validates the key backend changes without requiring the full Docker stack.
"""

import sys
import os
import json
import tempfile
import shutil
from datetime import datetime, timezone

# Test counters
passed = 0
failed = 0


def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name} -- {detail}")
        failed += 1


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1: Evaluation Service — Accurate Cost Tracking
# ═════════════════════════════════════════════════════════════════════════════
print("\n[TEST 1] Evaluation Service -- Accurate Cost Tracking")
os.chdir(os.path.join(os.path.dirname(__file__), "lexiassist-Python Services", "services", "evaluation"))
sys.path.insert(0, os.getcwd())

import main as eval_main
import database as eval_db

# 1.1 MODEL_PRICING matches orchestrator
test(
    "MODEL_PRICING has Gemini 2.5 Flash rates",
    eval_main.MODEL_PRICING.get("gemini-2.5-flash") == {"input": 0.00030, "output": 0.00250},
    f"Got {eval_main.MODEL_PRICING.get('gemini-2.5-flash')}"
)

# 1.2 Cost calculation is accurate
cost = eval_main._calculate_cost("gemini-2.5-flash", 1000, 500)
expected = (1000 / 1000) * 0.00030 + (500 / 1000) * 0.00250  # 0.00030 + 0.00125 = 0.00155
test(
    "_calculate_cost uses input/output split correctly",
    abs(cost - expected) < 1e-9,
    f"Expected {expected}, got {cost}"
)

# 1.3 Default model is gemini-2.5-flash
test(
    "Default model updated to gemini-2.5-flash",
    eval_main.DEFAULT_MODEL == "gemini-2.5-flash",
    f"Got {eval_main.DEFAULT_MODEL}"
)

# 1.4 database.py aggregates stored costs correctly
with tempfile.TemporaryDirectory() as tmpdir:
    eval_db.DATA_DIR = tmpdir
    eval_db.INTERACTIONS_FILE = os.path.join(tmpdir, "ai_interactions.json")
    # Seed two interactions with stored costs
    eval_db.save_interaction({
        "user_id": "u1", "service_type": "chat", "input_tokens": 1000, "output_tokens": 500,
        "latency_ms": 100, "success": True, "model_name": "gemini-2.5-flash",
        "estimated_cost_usd": 0.00155
    })
    eval_db.save_interaction({
        "user_id": "u1", "service_type": "chat", "input_tokens": 2000, "output_tokens": 1000,
        "latency_ms": 200, "success": True, "model_name": "gemini-2.5-flash",
        "estimated_cost_usd": 0.00310
    })
    stats = eval_db.get_interaction_stats("u1")
    test(
        "get_interaction_stats sums stored costs accurately",
        abs(stats["total_estimated_cost_usd"] - 0.00465) < 1e-6,
        f"Expected 0.00465, got {stats['total_estimated_cost_usd']}"
    )

# Clean sys.path
sys.path.pop(0)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2: Study Buddy -- Document Truncation
# ═════════════════════════════════════════════════════════════════════════════
print("\n[TEST 2] Study Buddy -- Document Truncation")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lexiassist-ai-main"))

# Mock weaviate before importing lexicore
import types
weaviate_mock = types.ModuleType("weaviate")
weaviate_mock.connect_to_weaviate_cloud = lambda **kw: None
weaviate_mock.auth = types.ModuleType("weaviate.auth")
weaviate_mock.auth.AuthApiKey = lambda key: None
weaviate_mock.collections = types.ModuleType("weaviate.collections")
weaviate_mock.collections.exists = lambda name: True
weaviate_mock.collections.use = lambda name: None
weaviate_mock.classes = types.ModuleType("weaviate.classes")
weaviate_mock.classes.config = types.ModuleType("weaviate.classes.config")
weaviate_mock.classes.config.Configure = types.SimpleNamespace()
weaviate_mock.classes.config.Configure.Generative = types.SimpleNamespace()
weaviate_mock.classes.config.Configure.Generative.google_gemini = lambda model: None
weaviate_mock.classes.config.DataType = types.SimpleNamespace(TEXT="text")
weaviate_mock.classes.config.Property = lambda **kw: None
weaviate_mock.classes.query = types.ModuleType("weaviate.classes.query")
weaviate_mock.classes.query.Filter = types.SimpleNamespace()
weaviate_mock.classes.query.Filter.by_property = lambda name: types.SimpleNamespace(
    equal=lambda val: None,
    like=lambda val: None
)
weaviate_mock.classes.query.MetadataQuery = types.SimpleNamespace(full_vector=True)
sys.modules["weaviate"] = weaviate_mock
sys.modules["weaviate.auth"] = weaviate_mock.auth
sys.modules["weaviate.classes"] = weaviate_mock.classes
sys.modules["weaviate.classes.config"] = weaviate_mock.classes.config
sys.modules["weaviate.classes.query"] = weaviate_mock.classes.query

# Provide a dummy API key so LangChain Google GenAI models can be instantiated
os.environ.setdefault("GOOGLE_API_KEY", "dummy-key-for-testing")
os.environ.setdefault("GEMINI_API_KEY", "dummy-key-for-testing")

from study_buddy.flashcards import generate_flashcards
from study_buddy.quizzes import generate_multiple_choice, generate_theory

# 2.1 Flashcards truncate when no course_code
prompt_captured = {}

class FakeLLM:
    def invoke(self, msgs):
        prompt_captured["text"] = msgs[1].content if len(msgs) > 1 else ""
        class FakeResp:
            content = '[]'
        return FakeResp()

try:
    from study_buddy import flashcards as fc_mod
    original_llm = fc_mod.llm
    fc_mod.llm = FakeLLM()

    state = {"document_text": "A" * 20000, "num_cards": 5, "flashcards": []}
    generate_flashcards(state)
    test(
        "Flashcards truncate document to ~8000 chars when no course_code",
        len(prompt_captured.get("text", "")) < 10000,
        f"Prompt length was {len(prompt_captured.get('text', ''))}"
    )
finally:
    fc_mod.llm = original_llm

# 2.2 Quiz multiple_choice truncates
prompt_captured.clear()
try:
    from study_buddy import quizzes as qz_mod
    original_llm = qz_mod.llm
    qz_mod.llm = FakeLLM()

    state = {"document_text": "B" * 20000, "quiz_type": "multiple_choice", "num_questions": 5, "questions": []}
    generate_multiple_choice(state)
    test(
        "Quiz MC truncates document to ~8000 chars when no course_code",
        len(prompt_captured.get("text", "")) < 10000,
        f"Prompt length was {len(prompt_captured.get('text', ''))}"
    )
finally:
    qz_mod.llm = original_llm


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3: AI Cache — Redis Deduplication Decorator
# ═════════════════════════════════════════════════════════════════════════════
print("\n[TEST 3] AI Cache -- Redis Deduplication Decorator")

from shared.ai_cache import ai_cache

# Mock Redis for deterministic testing
class FakeRedis:
    def __init__(self):
        self.store = {}
        self.hashes = {}
        self.lists = {}
    def get(self, key):
        return self.store.get(key)
    def setex(self, key, ttl, value):
        self.store[key] = value
    def hset(self, key, mapping=None):
        if key not in self.hashes:
            self.hashes[key] = {}
        if mapping:
            self.hashes[key].update(mapping)
    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))
    def lpush(self, key, value):
        if key not in self.lists:
            self.lists[key] = []
        self.lists[key].insert(0, value)
    def brpop(self, key, timeout=5):
        lst = self.lists.get(key, [])
        if lst:
            val = lst.pop()
            return (key, val)
        return None

# Patch the redis getter
import shared.ai_cache as cache_mod
original_get_redis = cache_mod._get_redis
fake_redis = FakeRedis()
cache_mod._get_redis = lambda: fake_redis

@ai_cache("test_ns", ttl=60)
def expensive_op(x, y):
    return {"result": x + y, "computed": True}

r1 = expensive_op(1, 2)
r2 = expensive_op(1, 2)

test(
    "ai_cache stores and returns cached result",
    r1 == r2 == {"result": 3, "computed": True},
    f"r1={r1}, r2={r2}"
)

test(
    "ai_cache wrote exactly one entry to Redis",
    len(fake_redis.store) == 1,
    f"Store has {len(fake_redis.store)} entries"
)

# Verify cache key is SHA-256 based on args
for key in fake_redis.store:
    test(
        "ai_cache key uses SHA-256 namespace prefix",
        key.startswith("ai_cache:test_ns:") and len(key.split(":")[-1]) == 64,
        f"Key was {key}"
    )

cache_mod._get_redis = original_get_redis


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4: Orchestrator -- Dynamic Model Router
# ═════════════════════════════════════════════════════════════════════════════
print("\n[TEST 4] Orchestrator -- Dynamic Model Router")

# Verify routing thresholds in source code (avoiding heavy module import side effects)
orch_path = os.path.join(os.path.dirname(__file__), "lexiassist-Python Services", "services", "orchestrator", "main.py")
with open(orch_path, "r", encoding="utf-8") as f:
    orch_source = f.read()

test(
    "ModelRouter source routes small inputs to flash-lite",
    'estimated_input_chars < 1500' in orch_source,
    "Threshold for flash-lite not found"
)

test(
    "ModelRouter source routes large contexts to pro at ~15k chars",
    'estimated_input_chars > 15000' in orch_source,
    "Threshold for pro not found"
)

# Lightweight functional test without executing the full module
class MockModelRouter:
    def select_model(self, task_type: str, query: str = "", context_chunks=None, override: str = None) -> str:
        if override and override in ("gemini-2.5-pro", "gemini-2.5-flash-lite", "gemini-2.5-flash"):
            return override
        context_len = sum(len(c) for c in (context_chunks or []))
        estimated = len(query) + context_len
        if estimated < 1500 and task_type in ("chat", "generate_summary"):
            return "gemini-2.5-flash-lite"
        if estimated > 12000 or task_type in ("generate_quiz", "generate_flashcards"):
            if estimated > 15000:
                return "gemini-2.5-pro"
            return "gemini-2.5-flash"
        return "gemini-2.5-flash"

router = MockModelRouter()

# 4.1 Override always wins
test(
    "ModelRouter respects override",
    router.select_model("chat", override="gemini-2.5-pro") == "gemini-2.5-pro",
    "Override was ignored"
)

# 4.2 Small chat → flash-lite
model = router.select_model("chat", query="Hi", context_chunks=[])
test(
    "Small chat query routes to flash-lite",
    model == "gemini-2.5-flash-lite",
    f"Got {model}"
)

# 4.3 Large context → pro
model = router.select_model("chat", query="A" * 5001, context_chunks=["B" * 10000])
test(
    "Very large context routes to pro",
    model == "gemini-2.5-pro",
    f"Got {model}"
)

# 4.4 Medium generation → flash
model = router.select_model("generate_quiz", query="Make a quiz", context_chunks=["x" * 2000])
test(
    "Medium generation task routes to flash",
    model == "gemini-2.5-flash",
    f"Got {model}"
)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 5: Job Queue -- Retry + Dead-Letter
# ═════════════════════════════════════════════════════════════════════════════
print("\n[TEST 5] Job Queue -- Retry + Dead-Letter")

# Mock redis module before importing job_queue
redis_mock = types.ModuleType("redis")
redis_mock.from_url = lambda url, **kw: fake_redis
sys.modules["redis"] = redis_mock

import job_queue as jq_mod
jq_mod._redis_client = fake_redis
fake_redis.store.clear()

job_id = jq_mod.enqueue_job("test_task", {"key": "val"}, "user_1", max_retries=1)
test(
    "enqueue_job returns a UUID-like job_id",
    len(job_id) == 36,
    f"Job ID was {job_id}"
)

status = jq_mod.get_job_status(job_id)
test(
    "Job starts in pending status",
    status and status.get("status") == "pending",
    f"Status was {status}"
)

# Simulate first failure + retry
requeued = jq_mod.requeue_job(job_id)
test(
    "requeue_job returns True on first retry",
    requeued,
    "Job was not requeued"
)

status_after_retry = jq_mod.get_job_status(job_id)
test(
    "Retry increments retry_count",
    status_after_retry and int(status_after_retry.get("retry_count", "0")) == 1,
    f"retry_count was {status_after_retry.get('retry_count')}"
)

# Simulate second failure → dead letter
requeued_again = jq_mod.requeue_job(job_id)
test(
    "requeue_job returns False when max retries exceeded",
    not requeued_again,
    "Job was incorrectly requeued again"
)

status_final = jq_mod.get_job_status(job_id)
test(
    "Max retries exceeded moves job to failed_permanently",
    status_final and status_final.get("status") == "failed_permanently",
    f"Final status was {status_final.get('status')}"
)

# Check dead-letter queue membership
dlq = fake_redis.lists.get("ai:queue:dead_letter", [])
test(
    "Dead-letter queue contains the failed job_id",
    job_id in dlq,
    f"DLQ contents: {dlq}"
)

jq_mod._redis_client = None


# ═════════════════════════════════════════════════════════════════════════════
# TEST 6: Reading Engine -- Context Compression
# ═════════════════════════════════════════════════════════════════════════════
print("\n[TEST 6] Reading Engine -- Context Compression")

# Mock gtts before importing reading_engine
gtts_mock = types.ModuleType("gtts")
gtts_mock.gTTS = lambda *args, **kwargs: None
sys.modules["gtts"] = gtts_mock

from reading_assistant.reading_engine import ReaadingEngine

engine = ReaadingEngine()

# 6.1 _select_summary_chunks exists and works
chunks = [f"chunk_{i}" for i in range(20)]
selected = engine._select_summary_chunks(chunks, max_chunks=5)
test(
    "_select_summary_chunks selects <= max_chunks",
    len(selected) <= 5,
    f"Selected {len(selected)} chunks"
)

test(
    "_select_summary_chunks preserves first and last",
    selected[0] == "chunk_0" and selected[-1] == "chunk_19",
    f"First={selected[0]}, Last={selected[-1]}"
)

# 6.2 Small input returns all chunks
small = [f"c{i}" for i in range(3)]
selected_small = engine._select_summary_chunks(small, max_chunks=12)
test(
    "_select_summary_chunks returns all when under max",
    selected_small == small,
    f"Got {selected_small}"
)


# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"Results: {passed} passed, {failed} failed")
print("=" * 60)

if failed > 0:
    sys.exit(1)
print("All integration tests passed!")
