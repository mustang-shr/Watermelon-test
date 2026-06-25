import shutil
import pytest

from memory.store import MemoryStore, ExecutionRecord
from memory.embeddings import LocalTFIDFEmbedder, FallbackEmbedder
from memory.schema import init_schema


@pytest.fixture
def memory(tmp_path):
    db_path = str(tmp_path / "memory_db")
    return MemoryStore(db_path, LocalTFIDFEmbedder())


def test_schema_init_is_idempotent(tmp_path):
    db_path = str(tmp_path / "memory_db")
    init_schema(db_path)
    init_schema(db_path)  # must not raise on second call (agent restart scenario)


def test_log_and_retrieve_similar_instruction(memory):
    memory.log_execution(ExecutionRecord(
        instruction_id="i1",
        text="find all stale issues with no activity in 30 days",
        steps=[{"description": "list open issues", "status": "success",
                "tool_used": "list_issues", "outcome": "success", "latency_ms": 100}],
        status="success", total_api_calls=1, total_time_ms=100,
    ))

    matches = memory.find_similar_past_instructions(
        "find all stale issues with no activity in the last 30 days"
    )
    assert len(matches) == 1
    assert matches[0]["id"] == "i1"
    assert matches[0]["decomposition"][0]["description"] == "list open issues"


def test_dissimilar_instruction_does_not_match(memory):
    memory.log_execution(ExecutionRecord(
        instruction_id="i1", text="find stale issues",
        steps=[], status="success", total_api_calls=1, total_time_ms=50,
    ))
    matches = memory.find_similar_past_instructions("deploy the production database")
    assert matches == []


def test_tool_stats_accumulate(memory):
    memory.log_execution(ExecutionRecord(
        instruction_id="i1", text="instr a",
        steps=[{"description": "s1", "status": "success", "tool_used": "list_issues",
                "outcome": "success", "latency_ms": 10}],
        status="success", total_api_calls=1, total_time_ms=10,
    ))
    memory.log_execution(ExecutionRecord(
        instruction_id="i2", text="instr b",
        steps=[{"description": "s1", "status": "failed", "tool_used": "list_issues",
                "outcome": "failed", "latency_ms": 10, "error_detail": "timeout"}],
        status="failed", total_api_calls=1, total_time_ms=10,
    ))
    stats = memory.get_tool_stats("list_issues")
    assert stats["success_count"] == 1
    assert stats["failure_count"] == 1


def test_fallback_embedder_uses_local_provider_when_primary_fails():
    class FailingProvider:
        def embed(self, texts):
            raise RuntimeError("remote provider unavailable")

    embedder = FallbackEmbedder(FailingProvider(), LocalTFIDFEmbedder())
    with pytest.warns(RuntimeWarning, match="remote provider unavailable"):
        vecs = embedder.embed(["alpha beta", "beta gamma"])

    assert vecs.shape[0] == 2
    assert vecs.ndim == 2
    assert vecs.dtype.kind in {"f", "i"}
