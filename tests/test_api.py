import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from api import create_app

from memory.store import MemoryStore
from memory.embeddings import LocalTFIDFEmbedder
from tools.registry import ToolRegistry
from tools.synthesis import CapabilitySynthesisEngine, MockCodeGen
from agent.planner import MockPlanner, Plan
from agent.executor import Step
from agent.agent import Agent


def _fake_agent(tmp_path):
    memory = MemoryStore(str(tmp_path / "memory_db"), LocalTFIDFEmbedder())
    registry = ToolRegistry(memory_store=memory)
    registry.register("ok_tool", lambda: "fine", kind="builtin", description="always succeeds")
    planner = MockPlanner(plans=[Plan(steps=[Step(description="do the thing", tool_name="ok_tool")],
                                        reasoning="simple plan")])
    engine = CapabilitySynthesisEngine(codegen=MockCodeGen(responses=[""]))
    return Agent(memory=memory, registry=registry, planner=planner,
                 synthesis_engine=engine, embedder=LocalTFIDFEmbedder())


def test_health(tmp_path):
    client = TestClient(create_app(_fake_agent(tmp_path)))
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["tools_registered"] == 1


def test_list_tools(tmp_path):
    client = TestClient(create_app(_fake_agent(tmp_path)))
    resp = client.get("/tools")
    assert resp.status_code == 200
    assert resp.json()[0]["name"] == "ok_tool"


def test_run_success(tmp_path):
    client = TestClient(create_app(_fake_agent(tmp_path)))
    resp = client.post("/run", json={"instruction": "do the thing"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["total_api_calls"] == 1
    assert body["steps"][0]["status"] == "success"


def test_run_rejects_empty_instruction(tmp_path):
    client = TestClient(create_app(_fake_agent(tmp_path)))
    resp = client.post("/run", json={"instruction": "   "})
    assert resp.status_code == 422


def _crashing_agent(tmp_path):
    memory = MemoryStore(str(tmp_path / "memory_db"), LocalTFIDFEmbedder())
    registry = ToolRegistry(memory_store=memory)
    engine = CapabilitySynthesisEngine(codegen=MockCodeGen(responses=[""]))

    class CrashingPlanner:
        def plan(self, instruction, available_tools, similar_past):
            raise RuntimeError("simulated unexpected internal failure")

    return Agent(memory=memory, registry=registry, planner=CrashingPlanner(),
                 synthesis_engine=engine, embedder=LocalTFIDFEmbedder())


def test_unhandled_exception_returns_structured_error_not_bare_string(tmp_path):
    """Previously an unhandled crash returned a bare 'Internal Server Error'
    with zero structured detail - broke the 'structured report every run'
    contract at the API boundary even though Agent.run() honored it
    internally. This is the regression test for that fix."""
    client = TestClient(create_app(_crashing_agent(tmp_path)), raise_server_exceptions=False)
    resp = client.post("/run", json={"instruction": "anything"})

    assert resp.status_code == 500
    body = resp.json()
    assert body["status"] == "internal_error"
    assert body["error_type"] == "RuntimeError"
    assert "error_id" in body
    assert "GitHub API or tool failure" in body["detail"]


def test_validation_error_unaffected_by_global_exception_handler(tmp_path):
    """The global Exception handler must not swallow FastAPI's own
    HTTPException handling - confirmed explicitly, not assumed."""
    client = TestClient(create_app(_fake_agent(tmp_path)))
    resp = client.post("/run", json={"instruction": ""})
    assert resp.status_code == 422
    assert "instruction must not be empty" in resp.json()["detail"]
