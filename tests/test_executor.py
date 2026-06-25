from agent.executor import Executor, Step
from tools.registry import ToolRegistry


def test_partial_failure_blocks_downstream_steps_and_reports_honestly():
    registry = ToolRegistry()
    registry.register("step_ok", lambda: "fine", kind="builtin", description="x")
    registry.register("step_broken", lambda: 1 / 0, kind="builtin", description="x")

    plan = [
        Step(description="step 1", tool_name="step_ok"),
        Step(description="step 2", tool_name="step_ok"),
        Step(description="step 3 breaks", tool_name="step_broken"),
        Step(description="step 4", tool_name="step_ok"),
        Step(description="step 5", tool_name="step_ok"),
    ]
    report = Executor(registry).execute("test instruction", plan)

    assert report.status == "partial"
    assert [r.status for r in report.step_results] == [
        "success", "success", "failed", "skipped", "skipped",
    ]
    assert "ZeroDivisionError" in report.step_results[2].error


def test_independent_step_runs_even_after_upstream_failure():
    registry = ToolRegistry()
    registry.register("step_ok", lambda: "fine", kind="builtin", description="x")
    registry.register("step_broken", lambda: 1 / 0, kind="builtin", description="x")

    plan = [
        Step(description="breaks", tool_name="step_broken"),
        Step(description="independent of the failure", tool_name="step_ok", independent=True),
    ]
    report = Executor(registry).execute("test instruction", plan)
    assert report.step_results[1].status == "success"


def test_all_success_reports_success_not_partial():
    registry = ToolRegistry()
    registry.register("step_ok", lambda: "fine", kind="builtin", description="x")
    plan = [Step(description="s1", tool_name="step_ok"), Step(description="s2", tool_name="step_ok")]
    report = Executor(registry).execute("test", plan)
    assert report.status == "success"


def test_inject_prev_output_as_passes_prior_step_output_to_next_step():
    """Verifies the output-chaining mechanism: a step with inject_prev_output_as
    gets the previous step's output injected as the named kwarg before execution.
    This is how list_issues → find_duplicate_issues works: the issue list
    produced by step 1 becomes 'issues' in step 2 automatically."""
    from memory.store import MemoryStore
    from memory.embeddings import LocalTFIDFEmbedder
    from tools.registry import ToolRegistry

    received = {}

    def capture_tool(**kwargs):
        received.update(kwargs)
        return "captured"

    memory = MemoryStore(str(pytest.importorskip("pathlib").Path(__file__).parent.parent / "data" / "test_inject"), LocalTFIDFEmbedder())
    registry = ToolRegistry(memory_store=memory)
    registry.register("producer", lambda: [1, 2, 3], kind="builtin", description="produces a list")
    registry.register("consumer", capture_tool, kind="builtin", description="receives the list")
    memory.close()

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        mem2 = MemoryStore(os.path.join(tmpdir, "db"), LocalTFIDFEmbedder())
        reg2 = ToolRegistry(memory_store=mem2)
        reg2.register("producer", lambda: [1, 2, 3], kind="builtin", description="produces a list")
        reg2.register("consumer", capture_tool, kind="builtin", description="receives the list")

        from agent.executor import Executor, Step
        executor = Executor(reg2)
        plan = [
            Step(description="produce list", tool_name="producer"),
            Step(description="consume list", tool_name="consumer",
                 inject_prev_output_as="items"),
        ]
        report = executor.execute("test instruction", plan)
        mem2.close()

    assert report.status == "success", f"Expected success but got {report.status}"
    assert received.get("items") == [1, 2, 3], f"Expected items=[1,2,3] but got {received}"


import pytest
