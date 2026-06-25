from memory.store import MemoryStore
from memory.embeddings import LocalTFIDFEmbedder
from tools.registry import ToolRegistry
from tools.github_tools import GitHubClient, register_github_tools
from tools.synthesis import CapabilitySynthesisEngine, MockCodeGen
from agent.planner import MockPlanner, Plan, RuleBasedPlanner, FallbackPlanner
from agent.executor import Step
from agent.agent import Agent

BROKEN_CODE = '''
def find_duplicate_issues(issues, similarity_fn, threshold=0.3):
    pairs = []
    for i in range(len(issues)):
        for j in range(i+1, len(issues)):
            sim = similarity_fn(issues[i]["text"], issues[j]["text"])
            if sim >= threshold:
                pairs.append((issues[i]["number"], issues[j]["number"], sim))
    return pairs
'''

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


def _build_agent(tmp_path, planner, codegen_responses=None):
    embedder = LocalTFIDFEmbedder()
    memory = MemoryStore(str(tmp_path / "memory_db"), embedder)
    registry = ToolRegistry(memory_store=memory)
    client = GitHubClient(repo="github/docs")
    register_github_tools(registry, client)
    registry._tools["list_issues"].fn = lambda **kw: [{"number": 1, "title": "stale issue"}]

    codegen = MockCodeGen(responses=codegen_responses or [""])
    synthesis_engine = CapabilitySynthesisEngine(codegen=codegen)
    return Agent(memory=memory, registry=registry, planner=planner,
                 synthesis_engine=synthesis_engine, embedder=embedder), memory, registry


def test_similar_instruction_reduces_call_count(tmp_path):
    plan1 = Plan(steps=[
        Step(description="broad attempt", tool_name="list_issues", kwargs={"state": "open"}),
        Step(description="retry with refined filter", tool_name="list_issues", kwargs={"state": "open"}),
    ], reasoning="no prior history")
    planner = MockPlanner(plans=[plan1])
    agent, memory, registry = _build_agent(tmp_path, planner)

    result1 = agent.run("find all stale issues with no activity in 30 days")
    assert result1.report.total_api_calls == 2

    plan2 = Plan(steps=[
        Step(description="reused from memory", tool_name="list_issues", kwargs={"state": "open"}),
    ], reasoning="reused decomposition")
    planner.plans = [plan2]
    planner.call_count = 0

    result2 = agent.run("find all stale issues with no activity in the last 30 days")
    assert len(result2.similar_past_used) == 1
    assert result2.report.total_api_calls == 1
    assert result2.report.total_api_calls < result1.report.total_api_calls


def test_capability_synthesis_through_full_agent(tmp_path):
    """Note: the threshold in the call must match what TF-IDF can actually achieve.
    The contract now uses 0.7 for live semantic embeddings (NIM), but tests use
    LocalTFIDFEmbedder which scores the same pair at ~0.38. We pass threshold=0.3
    explicitly here so TF-IDF can find the pair, matching how the contract's
    call_time_kwargs work with live embeddings (where 0.7 is appropriate)."""
    fixture_issues = [
        {"number": 1, "title": "Login button broken on mobile"},
        {"number": 2, "title": "Mobile login button does not work"},
        {"number": 3, "title": "Add dark mode support"},
    ]
    plan = Plan(steps=[
        Step(description="list open issues", tool_name="list_issues", kwargs={"state": "open"}),
        Step(description="find duplicates", tool_name="", kwargs={
            "issues": fixture_issues, "threshold": 0.3},  # explicit TF-IDF-compatible threshold
             needs_synthesis="duplicate_detection"),
    ], reasoning="no tool exists for similarity-based duplicate detection")
    planner = MockPlanner(plans=[plan])
    agent, memory, registry = _build_agent(tmp_path, planner, codegen_responses=[BROKEN_CODE, FIXED_CODE])
    registry._tools["list_issues"].fn = lambda **kw: fixture_issues

    result = agent.run("find duplicate issues among open issues")

    assert result.synthesis_events[0].success
    assert result.synthesis_events[0].attempts == 2
    assert registry.has("find_duplicate_issues")
    assert result.report.status == "success"
    assert memory.get_tool_stats("find_duplicate_issues")["success_count"] == 1


TIME_TO_CLOSE_BROKEN_CODE = '''
def average_resolution_time_by_label(closed_issues, parse_iso_to_epoch):
    totals = {}
    counts = {}
    for issue in closed_issues:
        for label in issue["label"]:
            name = label["name"]
            hours = (parse_iso_to_epoch(issue["closed_at"]) - parse_iso_to_epoch(issue["created_at"])) / 3600
            totals[name] = totals.get(name, 0) + hours
            counts[name] = counts.get(name, 0) + 1
    return {k: totals[k]/counts[k] for k in totals}
'''

TIME_TO_CLOSE_FIXED_CODE = '''
def average_resolution_time_by_label(closed_issues, parse_iso_to_epoch):
    totals = {}
    counts = {}
    for issue in closed_issues:
        for label in issue["labels"]:
            name = label["name"]
            hours = (parse_iso_to_epoch(issue["closed_at"]) - parse_iso_to_epoch(issue["created_at"])) / 3600
            totals[name] = totals.get(name, 0) + hours
            counts[name] = counts.get(name, 0) + 1
    return {k: totals[k]/counts[k] for k in totals}
'''


def test_second_gap_type_with_different_injected_capability(tmp_path):
    """Proves the synthesis pattern generalizes beyond duplicate_detection:
    a different kind of gap (date-math aggregation, not similarity), a
    different injected call-time capability (parse_iso_to_epoch, not
    similarity_fn), and - critically - no kwargs pre-wired by the test,
    since that's exactly the bug this test exists to prevent regressing on."""
    fixture = [
        {"labels": [{"name": "bug"}], "created_at": "2026-06-01T00:00:00Z", "closed_at": "2026-06-03T00:00:00Z"},
        {"labels": [{"name": "bug"}], "created_at": "2026-06-05T00:00:00Z", "closed_at": "2026-06-06T00:00:00Z"},
        {"labels": [{"name": "enhancement"}], "created_at": "2026-06-01T00:00:00Z", "closed_at": "2026-06-02T00:00:00Z"},
    ]
    plan = Plan(steps=[
        Step(description="avg resolution time by label", tool_name="", kwargs={"closed_issues": fixture},
             needs_synthesis="time_to_close_by_label"),
    ], reasoning="no endpoint computes this")
    planner = MockPlanner(plans=[plan])
    agent, memory, registry = _build_agent(
        tmp_path, planner, codegen_responses=[TIME_TO_CLOSE_BROKEN_CODE, TIME_TO_CLOSE_FIXED_CODE]
    )

    result = agent.run("average time to close issues by label?")

    assert result.synthesis_events[0].attempts == 2
    assert result.report.status == "success"
    output = result.report.step_results[0].output
    assert abs(output["bug"] - 36.0) < 0.01
    assert abs(output["enhancement"] - 24.0) < 0.01


def test_synthesis_failure_is_reported_not_silently_dropped(tmp_path):
    plan = Plan(steps=[
        Step(description="needs a gap type with no known contract", tool_name="",
             kwargs={}, needs_synthesis="some_unknown_capability"),
    ], reasoning="planner flagged a gap the agent has no contract for")
    planner = MockPlanner(plans=[plan])
    agent, memory, registry = _build_agent(tmp_path, planner)

    result = agent.run("do something the agent has no contract for")

    assert not result.synthesis_events[0].success
    assert result.report.status == "failed"
    assert "not registered" in result.report.step_results[0].error


def test_rule_based_planner_extracts_create_issue_fields():
    """Confirms the rule-based fallback correctly parses the DEMO.md
    instruction phrasings. This is NOT a test of the main NIM planner path -
    it's a regression test for the graceful-degradation path."""
    planner = RuleBasedPlanner()
    plan = planner.plan(
        "Create an issue titles ' agent smoke test' with the body 'checking the live wiring works'",
        available_tools=[],
        similar_past=[],
    )
    assert plan.steps[0].tool_name == "create_issue"
    assert plan.steps[0].kwargs["title"] == "agent smoke test"
    assert plan.steps[0].kwargs["body"] == "checking the live wiring works"


def test_fallback_planner_uses_rule_based_when_primary_fails():
    """If the primary planner raises (timeout, auth failure, etc.), the
    FallbackPlanner should transparently switch to RuleBasedPlanner and
    return a usable plan rather than crashing."""
    import warnings

    class AlwaysFailsPlanner:
        def plan(self, instruction, available_tools, similar_past):
            raise TimeoutError("NIM is down")

    planner = FallbackPlanner(primary=AlwaysFailsPlanner(), fallback=RuleBasedPlanner())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan = planner.plan("list open issues", available_tools=[], similar_past=[])

    assert plan.steps[0].tool_name == "list_issues"
    assert any("NIM is down" in str(w.message) for w in caught)


def test_rule_based_planner_handles_close_issue():
    planner = RuleBasedPlanner()
    plan = planner.plan(
        "Close issue number 7 with a comment 'Fixed in latest commit'",
        available_tools=[], similar_past=[],
    )
    assert plan.steps[0].tool_name == "close_issue"
    assert plan.steps[0].kwargs["issue_number"] == 7
    assert plan.steps[0].kwargs["comment"] == "Fixed in latest commit"


def test_rule_based_planner_handles_list_pull_requests():
    planner = RuleBasedPlanner()
    plan = planner.plan("list pull requests", available_tools=[], similar_past=[])
    assert plan.steps[0].tool_name == "list_pull_requests"


def test_rule_based_planner_duplicate_detection_emits_two_steps():
    planner = RuleBasedPlanner()
    plan = planner.plan("find duplicate issues", available_tools=[], similar_past=[])
    assert len(plan.steps) == 2
    assert plan.steps[0].tool_name == "list_issues"
    assert plan.steps[1].needs_synthesis == "duplicate_detection"
