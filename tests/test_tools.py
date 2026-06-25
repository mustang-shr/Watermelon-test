from tools.github_tools import _filter_non_pr_issues
from tools.registry import ToolRegistry


def test_filters_out_pull_requests():
    fixture = [
        {"number": 1, "title": "real bug report"},
        {"number": 2, "title": "a PR", "pull_request": {"url": "x"}},
        {"number": 3, "title": "another real issue"},
    ]
    result = _filter_non_pr_issues(fixture)
    assert [i["number"] for i in result] == [1, 3]


def test_registry_wraps_success():
    registry = ToolRegistry()
    registry.register("ok_tool", lambda: "result", kind="builtin", description="x")
    result = registry.call("ok_tool")
    assert result.success
    assert result.output == "result"


def test_registry_wraps_failure_cleanly():
    registry = ToolRegistry()
    registry.register("broken_tool", lambda: 1 / 0, kind="builtin", description="x")
    result = registry.call("broken_tool")
    assert not result.success
    assert "ZeroDivisionError" in result.error


def test_registry_reports_missing_tool():
    registry = ToolRegistry()
    result = registry.call("does_not_exist")
    assert not result.success
    assert "not registered" in result.error
