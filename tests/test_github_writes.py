"""
create_issue / add_comment / add_labels had ZERO test coverage before this -
not even mocked. These tests don't hit the network (api.github.com isn't
reliably reachable from CI either way, and this sandbox's IP is already
rate-limited), but they do verify the actual request shape this code sends -
URL, headers, JSON body - and that responses/errors are handled correctly.
That's a meaningfully different and weaker claim than a live test, and is
stated as such; it catches "this code is structurally wrong" bugs, not
"GitHub's API behaves differently than expected" bugs.
"""

import pytest
from unittest.mock import patch, MagicMock

from tools.github_tools import GitHubClient


def _mock_response(json_body, status_ok=True):
    resp = MagicMock()
    resp.json.return_value = json_body
    if status_ok:
        resp.raise_for_status.return_value = None
    else:
        import requests
        resp.raise_for_status.side_effect = requests.HTTPError("422 Client Error")
    return resp


def test_create_issue_sends_correct_request_shape():
    client = GitHubClient(token="fake-token", repo="owner/repo")
    fake_resp = _mock_response({"number": 42, "title": "test issue"})

    with patch("requests.post", return_value=fake_resp) as mock_post:
        result = client.create_issue(title="test issue", body="body text", labels=["bug"])

    assert result["number"] == 42
    call_args = mock_post.call_args
    assert call_args.args[0] == "https://api.github.com/repos/owner/repo/issues"
    assert call_args.kwargs["headers"]["Authorization"] == "Bearer fake-token"
    assert call_args.kwargs["json"] == {"title": "test issue", "body": "body text", "labels": ["bug"]}


def test_add_comment_sends_correct_request_shape():
    client = GitHubClient(token="fake-token", repo="owner/repo")
    fake_resp = _mock_response({"id": 999, "body": "a comment"})

    with patch("requests.post", return_value=fake_resp) as mock_post:
        result = client.add_comment(issue_number=7, body="a comment")

    assert result["id"] == 999
    call_args = mock_post.call_args
    assert call_args.args[0] == "https://api.github.com/repos/owner/repo/issues/7/comments"
    assert call_args.kwargs["json"] == {"body": "a comment"}


def test_add_labels_sends_correct_request_shape():
    client = GitHubClient(token="fake-token", repo="owner/repo")
    fake_resp = _mock_response([{"name": "bug"}, {"name": "priority-high"}])

    with patch("requests.post", return_value=fake_resp) as mock_post:
        result = client.add_labels(issue_number=7, labels=["bug", "priority-high"])

    assert len(result) == 2
    call_args = mock_post.call_args
    assert call_args.args[0] == "https://api.github.com/repos/owner/repo/issues/7/labels"
    assert call_args.kwargs["json"] == {"labels": ["bug", "priority-high"]}


@pytest.mark.parametrize("method_name,kwargs", [
    ("create_issue", {"title": "x"}),
    ("add_comment", {"issue_number": 1, "body": "x"}),
    ("add_labels", {"issue_number": 1, "labels": ["x"]}),
])
def test_write_operations_require_token(method_name, kwargs):
    client = GitHubClient(token=None, repo="owner/repo")  # no token
    method = getattr(client, method_name)
    with pytest.raises(PermissionError):
        method(**kwargs)


def test_create_issue_propagates_real_http_errors():
    client = GitHubClient(token="fake-token", repo="owner/repo")
    fake_resp = _mock_response({"message": "Validation Failed"}, status_ok=False)

    with patch("requests.post", return_value=fake_resp):
        import requests
        with pytest.raises(requests.HTTPError):
            client.create_issue(title="x")


def test_close_issue_sends_patch_with_state_closed():
    client = GitHubClient(token="fake-token", repo="owner/repo")
    fake_resp = _mock_response({"number": 7, "state": "closed"})

    with patch("requests.patch", return_value=fake_resp) as mock_patch:
        result = client.close_issue(issue_number=7)

    assert result["state"] == "closed"
    call_args = mock_patch.call_args
    assert call_args.args[0] == "https://api.github.com/repos/owner/repo/issues/7"
    assert call_args.kwargs["json"] == {"state": "closed"}


def test_close_issue_with_comment_sends_comment_first_then_close():
    client = GitHubClient(token="fake-token", repo="owner/repo")
    comment_resp = _mock_response({"id": 999})
    close_resp = _mock_response({"number": 7, "state": "closed"})

    with patch("requests.post", return_value=comment_resp) as mock_post, \
         patch("requests.patch", return_value=close_resp) as mock_patch:
        client.close_issue(issue_number=7, comment="Fixed in v2.0")

    assert mock_post.call_args.kwargs["json"] == {"body": "Fixed in v2.0"}
    assert mock_patch.call_args.kwargs["json"] == {"state": "closed"}


def test_update_issue_only_sends_provided_fields():
    client = GitHubClient(token="fake-token", repo="owner/repo")
    fake_resp = _mock_response({"number": 5, "title": "New title"})

    with patch("requests.patch", return_value=fake_resp) as mock_patch:
        client.update_issue(issue_number=5, title="New title")

    payload = mock_patch.call_args.kwargs["json"]
    assert payload == {"title": "New title"}
    assert "body" not in payload
    assert "state" not in payload


def test_assign_issue_sends_assignees():
    client = GitHubClient(token="fake-token", repo="owner/repo")
    fake_resp = _mock_response({"number": 3, "assignees": [{"login": "alice"}]})

    with patch("requests.post", return_value=fake_resp) as mock_post:
        result = client.assign_issue(issue_number=3, assignees=["alice"])

    assert mock_post.call_args.args[0] == "https://api.github.com/repos/owner/repo/issues/3/assignees"
    assert mock_post.call_args.kwargs["json"] == {"assignees": ["alice"]}


def test_get_issue_does_not_require_token():
    client = GitHubClient(token=None, repo="owner/repo")
    fake_resp = _mock_response({"number": 1, "title": "bug report", "state": "open"})

    with patch("requests.get", return_value=fake_resp):
        result = client.get_issue(issue_number=1)

    assert result["title"] == "bug report"


def test_create_pull_request_sends_correct_shape():
    client = GitHubClient(token="fake-token", repo="owner/repo")
    fake_resp = _mock_response({"number": 10, "title": "Add feature", "state": "open"})

    with patch("requests.post", return_value=fake_resp) as mock_post:
        result = client.create_pull_request(
            title="Add feature", head="feature/x", base="main", body="Adds X"
        )

    assert result["number"] == 10
    call_args = mock_post.call_args
    assert call_args.args[0] == "https://api.github.com/repos/owner/repo/pulls"
    assert call_args.kwargs["json"]["head"] == "feature/x"
    assert call_args.kwargs["json"]["base"] == "main"


def test_merge_pull_request_uses_squash_method():
    client = GitHubClient(token="fake-token", repo="owner/repo")
    fake_resp = _mock_response({"merged": True, "sha": "abc123"})

    with patch("requests.put", return_value=fake_resp) as mock_put:
        result = client.merge_pull_request(pr_number=10, merge_method="squash")

    assert result["merged"] is True
    assert mock_put.call_args.kwargs["json"]["merge_method"] == "squash"


def test_close_pull_request_sends_state_closed():
    client = GitHubClient(token="fake-token", repo="owner/repo")
    fake_resp = _mock_response({"number": 10, "state": "closed"})

    with patch("requests.patch", return_value=fake_resp) as mock_patch:
        result = client.close_pull_request(pr_number=10)

    assert result["state"] == "closed"
    assert mock_patch.call_args.kwargs["json"] == {"state": "closed"}


@pytest.mark.parametrize("method_name,kwargs", [
    ("close_issue", {"issue_number": 1}),
    ("update_issue", {"issue_number": 1, "title": "x"}),
    ("assign_issue", {"issue_number": 1, "assignees": ["alice"]}),
    ("create_pull_request", {"title": "x", "head": "feat/x"}),
    ("merge_pull_request", {"pr_number": 1}),
    ("close_pull_request", {"pr_number": 1}),
])
def test_all_write_operations_require_token(method_name, kwargs):
    client = GitHubClient(token=None, repo="owner/repo")
    with pytest.raises(PermissionError):
        getattr(client, method_name)(**kwargs)


def test_search_issues_auto_adds_is_issue_qualifier():
    """search_issues adds 'is:issue' automatically to avoid PR contamination.
    Verified by checking the query that actually reaches GitHub."""
    client = GitHubClient(token="fake-token", repo="owner/repo")
    fake_resp = _mock_response({"items": [{"number": 1, "title": "bug"}], "total_count": 1})

    with patch("requests.get", return_value=fake_resp) as mock_get:
        result = client.search_issues("is:open label:bug")

    actual_query = mock_get.call_args.kwargs["params"]["q"]
    assert "is:issue" in actual_query, f"is:issue missing from query: {actual_query}"
    assert "is:open" in actual_query
    assert "label:bug" in actual_query


def test_search_issues_does_not_duplicate_is_issue():
    """If the caller already included is:issue, it should not be added twice."""
    client = GitHubClient(token="fake-token", repo="owner/repo")
    fake_resp = _mock_response({"items": [], "total_count": 0})

    with patch("requests.get", return_value=fake_resp) as mock_get:
        client.search_issues("is:open is:issue")

    actual_query = mock_get.call_args.kwargs["params"]["q"]
    assert actual_query.count("is:issue") == 1, f"is:issue duplicated: {actual_query}"


def test_search_issues_converts_422_to_clear_valueerror():
    """422 from GitHub's search API (unindexed repo) raises ValueError
    with a helpful message pointing to list_issues() as the alternative."""
    import requests as req
    client = GitHubClient(token="fake-token", repo="owner/repo")

    mock_resp = _mock_response({}, status_ok=False)
    mock_resp.raise_for_status.side_effect = req.HTTPError(
        response=type("R", (), {"status_code": 422})()
    )

    with patch("requests.get", return_value=mock_resp):
        with pytest.raises(ValueError, match="list_issues"):
            client.search_issues("is:open")
