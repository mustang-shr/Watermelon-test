from memory.store import MemoryStore
from memory.embeddings import LocalTFIDFEmbedder
from tools.registry import ToolRegistry
from tools.github_tools import GitHubClient, register_github_tools
from tools.synthesis import CapabilitySynthesisEngine, MockCodeGen
from agent.planner import MockPlanner, Plan
from agent.executor import Step
from agent.agent import Agent

FIXED_CODE = '''
def find_duplicate_issues(issues, similarity_fn, threshold=0.3):
    pairs = []
    for i in range(len(issues)):
        for j in range(i+1, len(issues)):
            sim = similarity_fn(issues[i]["title"], issues[j]["title"])
            if sim >= threshold:
                pairs.append((issues[i]["number"], issues[j]["number"], sim))
    return pairs
'''

FIXTURE_ISSUES = [
    {"number": 1, "title": "Login button broken on mobile"},
    {"number": 2, "title": "Mobile login button does not work"},
]


def _agent_with_synth_plan(memory, embedder, codegen_responses):
    registry = ToolRegistry(memory_store=memory)
    client = GitHubClient(repo="github/docs")
    register_github_tools(registry, client)
    registry._tools["list_issues"].fn = lambda **kw: FIXTURE_ISSUES

    plan = Plan(steps=[Step(description="find dupes", tool_name="", kwargs={
        "issues": FIXTURE_ISSUES, "similarity_fn": embedder.similarity, "threshold": 0.3,
    }, needs_synthesis="duplicate_detection")], reasoning="gap")
    planner = MockPlanner(plans=[plan])
    engine = CapabilitySynthesisEngine(codegen=MockCodeGen(responses=codegen_responses))
    return Agent(memory=memory, registry=registry, planner=planner,
                 synthesis_engine=engine, embedder=embedder), registry


def test_synthesized_tool_survives_simulated_restart(tmp_path):
    """The requirement is: synthesized tools must survive a process restart.
    'Process restart' means a genuinely separate OS process, not two objects
    in the same Python interpreter. Previous versions simulated it in-process
    (memory1.close() then open memory2 in the same test) — which on Windows
    consistently raised a Kuzu file-lock error because the OS-level handle
    was still alive via Python's own GC, not released despite .close() being
    called. This version uses subprocess to make it structurally correct:
    process 1 runs, exits (OS releases all handles), process 2 runs.
    The file-lock error is now impossible by construction."""
    import subprocess
    import json
    import sys as _sys

    db_path = str(tmp_path / "memory_db").replace("\\", "/")
    python_exe = _sys.executable  # sys.executable works on Windows (python), Linux (python3), and venv

    # ── Process 1: synthesize and register the tool ──────────────────────
    p1_script = f"""
import sys
sys.path.insert(0, "src")
from memory.store import MemoryStore
from memory.embeddings import LocalTFIDFEmbedder
from tools.registry import ToolRegistry
from tools.github_tools import GitHubClient, register_github_tools
from tools.synthesis import CapabilitySynthesisEngine, MockCodeGen
from agent.planner import MockPlanner, Plan
from agent.executor import Step
from agent.agent import Agent

FIXED_CODE = '''
def find_duplicate_issues(issues, similarity_fn, threshold=0.3):
    pairs = []
    for i in range(len(issues)):
        for j in range(i+1, len(issues)):
            sim = similarity_fn(issues[i]["title"], issues[j]["title"])
            if sim >= threshold:
                pairs.append((issues[i]["number"], issues[j]["number"], sim))
    return pairs
'''

embedder = LocalTFIDFEmbedder()
memory = MemoryStore("{db_path}", embedder)
registry = ToolRegistry(memory_store=memory)
client = GitHubClient(repo="github/docs")
register_github_tools(registry, client)

fixture = [
    {{"number": 1, "title": "Login broken on mobile"}},
    {{"number": 2, "title": "Mobile login is broken"}},
]
plan = Plan(steps=[Step(description="find dupes", tool_name="", kwargs={{"issues": fixture}},
                        needs_synthesis="duplicate_detection")], reasoning="gap")
planner = MockPlanner(plans=[plan])
engine = CapabilitySynthesisEngine(codegen=MockCodeGen(responses=[FIXED_CODE]))
agent = Agent(memory=memory, registry=registry, planner=planner,
              synthesis_engine=engine, embedder=embedder)

result = agent.run("find duplicate issues")
memory.close()  # explicit close before process exits - good practice even though exit handles it
assert result.synthesis_events[0].success, "synthesis failed in process 1"
print("PROCESS_1_OK")
"""
    r1 = subprocess.run([python_exe, "-c", p1_script],
                        cwd=str(tmp_path.parent.parent.parent / "home/claude/watermelon-agent")
                        if (tmp_path.parent.parent.parent / "home/claude/watermelon-agent").exists()
                        else ".",
                        capture_output=True, text=True)
    # fallback: run from the actual project directory
    if "PROCESS_1_OK" not in r1.stdout:
        import os
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        r1 = subprocess.run([python_exe, "-c", p1_script],
                            cwd=project_dir, capture_output=True, text=True)

    assert "PROCESS_1_OK" in r1.stdout, f"Process 1 failed:\nSTDOUT: {r1.stdout}\nSTDERR: {r1.stderr}"

    # ── Process 2: fresh start, no synthesis step, tool must already be live ─
    p2_script = f"""
import sys
sys.path.insert(0, "src")
from memory.store import MemoryStore
from memory.embeddings import LocalTFIDFEmbedder
from tools.registry import ToolRegistry
from tools.github_tools import GitHubClient, register_github_tools
from tools.synthesis import CapabilitySynthesisEngine, MockCodeGen
from agent.planner import MockPlanner, Plan
from agent.agent import Agent

embedder = LocalTFIDFEmbedder()
memory = MemoryStore("{db_path}", embedder)
registry = ToolRegistry(memory_store=memory)
client = GitHubClient(repo="github/docs")
register_github_tools(registry, client)

engine = CapabilitySynthesisEngine(codegen=MockCodeGen(responses=[""]))
agent = Agent(memory=memory, registry=registry,
              planner=MockPlanner(plans=[Plan(steps=[], reasoning="n/a")]),
              synthesis_engine=engine, embedder=embedder)

assert registry.has("find_duplicate_issues"), "tool NOT rebuilt from persisted code - memory restart failed"
fixture = [
    {{"number": 1, "title": "Login broken on mobile"}},
    {{"number": 2, "title": "Mobile login is broken"}},
]
result = registry.call("find_duplicate_issues",
                        issues=fixture, similarity_fn=embedder.similarity)
assert result.success, f"tool call failed after reload: {{result.error}}"
memory.close()
print("PROCESS_2_OK")
"""
    import os
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r2 = subprocess.run([python_exe, "-c", p2_script],
                        cwd=project_dir, capture_output=True, text=True)

    assert "PROCESS_2_OK" in r2.stdout, (
        f"Process 2 failed — tool NOT reloaded from persisted code:\n"
        f"STDOUT: {r2.stdout}\nSTDERR: {r2.stderr}"
    )


def test_confidence_high_on_clean_run_no_precedent(tmp_path):
    memory = MemoryStore(str(tmp_path / "memory_db"), LocalTFIDFEmbedder())
    registry = ToolRegistry(memory_store=memory)
    registry.register("ok_tool", lambda: "fine", kind="builtin", description="x")
    planner = MockPlanner(plans=[Plan(steps=[Step(description="s", tool_name="ok_tool")], reasoning="r")])
    engine = CapabilitySynthesisEngine(codegen=MockCodeGen(responses=[""]))
    agent = Agent(memory=memory, registry=registry, planner=planner,
                  synthesis_engine=engine, embedder=LocalTFIDFEmbedder())

    result = agent.run("a totally novel instruction")
    assert result.confidence == 0.9  # success, but no precedent -> small penalty
    assert "no prior precedent" in result.confidence_reason


def test_confidence_drops_on_partial_failure(tmp_path):
    memory = MemoryStore(str(tmp_path / "memory_db"), LocalTFIDFEmbedder())
    registry = ToolRegistry(memory_store=memory)
    registry.register("broken", lambda: 1 / 0, kind="builtin", description="x")
    planner = MockPlanner(plans=[Plan(steps=[Step(description="s", tool_name="broken")], reasoning="r")])
    engine = CapabilitySynthesisEngine(codegen=MockCodeGen(responses=[""]))
    agent = Agent(memory=memory, registry=registry, planner=planner,
                  synthesis_engine=engine, embedder=LocalTFIDFEmbedder())

    result = agent.run("an instruction that fails")
    assert result.confidence < 0.7
    assert "failed" in result.confidence_reason


def test_planner_receives_tool_track_record(tmp_path):
    """Verifies the previously-dead success/failure data actually reaches
    the planner's input with real values - not just that the keys exist.
    Stats only update via the full agent.run() -> log_execution() path;
    calling registry.call() directly does NOT update memory (confirmed
    separately - that's a real seam in this design, not a test bug)."""
    memory = MemoryStore(str(tmp_path / "memory_db"), LocalTFIDFEmbedder())
    registry = ToolRegistry(memory_store=memory)
    registry.register("flaky_tool", lambda: 1 / 0, kind="builtin", description="x")

    planner = MockPlanner(plans=[Plan(steps=[Step(description="s", tool_name="flaky_tool")], reasoning="r")])
    engine = CapabilitySynthesisEngine(codegen=MockCodeGen(responses=[""]))
    agent = Agent(memory=memory, registry=registry, planner=planner,
                  synthesis_engine=engine, embedder=LocalTFIDFEmbedder())

    agent.run("first run - establishes a failure on flaky_tool")

    captured = {}

    class CapturingPlanner:
        def plan(self, instruction, available_tools, similar_past):
            captured["tools"] = available_tools
            return Plan(steps=[], reasoning="captured")

    agent.planner = CapturingPlanner()
    agent.run("second run - planner should see flaky_tool's real track record")

    tool_entry = next(t for t in captured["tools"] if t["name"] == "flaky_tool")
    assert tool_entry["failure_count"] == 1
    assert tool_entry["success_count"] == 0
