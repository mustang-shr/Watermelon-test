"""
Built-in GitHub tools. Plain `requests` against api.github.com rather than
PyGithub - keeps the exact HTTP shape visible, which matters when the
capability-synthesis engine needs to generate NEW calls in this same style.

Auth is optional on read endpoints (lower rate limit, ~60/hr unauthenticated)
and required on write endpoints. Every function raises on failure rather than
swallowing errors - ToolRegistry.call() is what catches and turns that into a
ToolResult, so partial-failure handling lives in exactly one place.
"""

import requests

GITHUB_API = "https://api.github.com"


def _filter_non_pr_issues(raw_issues: list[dict]) -> list[dict]:
    """GitHub's /issues endpoint also returns PRs (a PR IS an issue internally) -
    distinguishable only by presence of a 'pull_request' key. Pure function so
    this logic is testable without hitting the network."""
    return [i for i in raw_issues if "pull_request" not in i]


class GitHubClient:
    def __init__(self, token: str | None = None, repo: str | None = None):
        """repo is 'owner/name'. token is optional for read-only public-repo calls."""
        self.token = token
        self.repo = repo
        self.headers = {"Accept": "application/vnd.github+json"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def _repo(self, repo: str | None) -> str:
        r = repo or self.repo
        if not r:
            raise ValueError("No repo specified and no default repo set on client")
        return r

    # ---- reads ----

    def list_issues(self, repo: str | None = None, state: str = "open", per_page: int = 30) -> list[dict]:
        resp = requests.get(
            f"{GITHUB_API}/repos/{self._repo(repo)}/issues",
            headers=self.headers,
            params={"state": state, "per_page": per_page},
            timeout=15,
        )
        resp.raise_for_status()
        # GitHub's issues endpoint also returns PRs - filter those out, this is
        # exactly the kind of undocumented-until-you-hit-it behaviour the
        # constraint-discovery layer in memory is meant to capture
        return _filter_non_pr_issues(resp.json())

    def search_issues(self, query: str, per_page: int = 30) -> list[dict]:
        """Search issues using GitHub's search qualifier syntax.

        Examples:
            'is:open label:bug'
            'assignee:alice is:open'
            'no:assignee is:open'

        Note: GitHub's search index lags behind the database by minutes to
        hours for new/small repos. If you get a 422 or empty results on a
        repo that definitely has issues, use list_issues() instead - it
        queries the database directly. is:issue is added automatically to
        avoid ambiguity between issues and PRs in search results.
        """
        # Always add is:issue so results are unambiguous (search/issues returns
        # both issues and PRs by default; is:issue filters to issues only).
        if "is:issue" not in query and "is:pr" not in query:
            query = f"{query} is:issue"

        try:
            resp = requests.get(
                f"{GITHUB_API}/search/issues",
                headers=self.headers,
                params={"q": query, "per_page": per_page},
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json().get("items", [])
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 422:
                raise ValueError(
                    f"GitHub search returned 422 for query '{query}'. "
                    "This usually means the repository hasn't been indexed by GitHub's "
                    "search engine yet (common for new or recently-created repos). "
                    "Use list_issues() instead — it queries the database directly and "
                    "doesn't depend on the search index."
                ) from e
            raise

    def get_rate_limit(self) -> dict:
        resp = requests.get(f"{GITHUB_API}/rate_limit", headers=self.headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    # ---- writes (require token) ----

    def create_issue(self, title: str, body: str = "", labels: list[str] | None = None,
                      repo: str | None = None) -> dict:
        self._require_token()
        resp = requests.post(
            f"{GITHUB_API}/repos/{self._repo(repo)}/issues",
            headers=self.headers,
            json={"title": title, "body": body, "labels": labels or []},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    def add_comment(self, issue_number: int, body: str, repo: str | None = None) -> dict:
        self._require_token()
        resp = requests.post(
            f"{GITHUB_API}/repos/{self._repo(repo)}/issues/{issue_number}/comments",
            headers=self.headers,
            json={"body": body},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    def add_labels(self, issue_number: int, labels: list[str], repo: str | None = None) -> dict:
        self._require_token()
        resp = requests.post(
            f"{GITHUB_API}/repos/{self._repo(repo)}/issues/{issue_number}/labels",
            headers=self.headers,
            json={"labels": labels},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    def _require_token(self):
        if not self.token:
            raise PermissionError("This operation requires a GitHub token (write access)")

    def close_issue(self, issue_number: int, comment: str | None = None,
                    repo: str | None = None) -> dict:
        """Close an issue, optionally adding a comment first."""
        self._require_token()
        r = self._repo(repo)
        if comment:
            requests.post(
                f"https://api.github.com/repos/{r}/issues/{issue_number}/comments",
                headers={"Authorization": f"Bearer {self.token}"},
                json={"body": comment},
                timeout=30,
            ).raise_for_status()
        resp = requests.patch(
            f"https://api.github.com/repos/{r}/issues/{issue_number}",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"state": "closed"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def update_issue(self, issue_number: int, title: str | None = None,
                     body: str | None = None, state: str | None = None,
                     labels: list[str] | None = None,
                     repo: str | None = None) -> dict:
        """Update an issue's title, body, state, or labels in one call."""
        self._require_token()
        r = self._repo(repo)
        payload = {k: v for k, v in
                   {"title": title, "body": body, "state": state, "labels": labels}.items()
                   if v is not None}
        resp = requests.patch(
            f"https://api.github.com/repos/{r}/issues/{issue_number}",
            headers={"Authorization": f"Bearer {self.token}"},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def assign_issue(self, issue_number: int, assignees: list[str],
                     repo: str | None = None) -> dict:
        """Assign one or more users to an issue."""
        self._require_token()
        r = self._repo(repo)
        resp = requests.post(
            f"https://api.github.com/repos/{r}/issues/{issue_number}/assignees",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"assignees": assignees},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def get_issue(self, issue_number: int, repo: str | None = None) -> dict:
        """Fetch full details of a single issue including labels, assignees, state."""
        r = self._repo(repo)
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        resp = requests.get(
            f"https://api.github.com/repos/{r}/issues/{issue_number}",
            headers=headers, timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def create_pull_request(self, title: str, head: str, base: str = "main",
                             body: str = "", draft: bool = False,
                             repo: str | None = None) -> dict:
        """Open a pull request from head branch into base branch."""
        self._require_token()
        r = self._repo(repo)
        resp = requests.post(
            f"https://api.github.com/repos/{r}/pulls",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"title": title, "head": head, "base": base,
                  "body": body, "draft": draft},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def list_pull_requests(self, state: str = "open", per_page: int = 30,
                            repo: str | None = None) -> list[dict]:
        """List PRs. state: 'open', 'closed', or 'all'."""
        r = self._repo(repo)
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        resp = requests.get(
            f"https://api.github.com/repos/{r}/pulls",
            headers=headers,
            params={"state": state, "per_page": per_page},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def merge_pull_request(self, pr_number: int, commit_title: str | None = None,
                            merge_method: str = "merge",
                            repo: str | None = None) -> dict:
        """Merge a pull request. merge_method: 'merge', 'squash', or 'rebase'."""
        self._require_token()
        r = self._repo(repo)
        payload = {"merge_method": merge_method}
        if commit_title:
            payload["commit_title"] = commit_title
        resp = requests.put(
            f"https://api.github.com/repos/{r}/pulls/{pr_number}/merge",
            headers={"Authorization": f"Bearer {self.token}"},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def close_pull_request(self, pr_number: int, repo: str | None = None) -> dict:
        """Close (without merging) a pull request."""
        self._require_token()
        r = self._repo(repo)
        resp = requests.patch(
            f"https://api.github.com/repos/{r}/pulls/{pr_number}",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"state": "closed"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


def register_github_tools(registry, client: GitHubClient) -> None:
    """Wire the client's methods into the tool registry under stable names
    the planner can reference. Every tool here is visible to the planner as
    a named capability with a description - that's how the NIM planner knows
    what it can do when decomposing an instruction."""

    # === READ tools (no token required) ===
    registry.register("list_issues", client.list_issues, kind="builtin",
                       description="List issues for the repo, filtered by state ('open'/'closed'/'all').",
                       source="github_api")
    registry.register("search_issues", client.search_issues, kind="builtin",
                       description="Search issues using GitHub search qualifier syntax (e.g. 'is:open label:bug').",
                       source="github_api")
    registry.register("get_issue", client.get_issue, kind="builtin",
                       description="Get full details of a single issue by number: title, body, labels, assignees, state.",
                       source="github_api")
    registry.register("list_pull_requests", client.list_pull_requests, kind="builtin",
                       description="List pull requests filtered by state ('open'/'closed'/'all').",
                       source="github_api")
    registry.register("get_rate_limit", client.get_rate_limit, kind="builtin",
                       description="Check current GitHub API rate limit status.",
                       source="github_api")

    # === WRITE tools (token required) ===
    registry.register("create_issue", client.create_issue, kind="builtin",
                       description="Create a new issue with title, body, and optional labels. Requires write token.",
                       source="github_api")
    registry.register("close_issue", client.close_issue, kind="builtin",
                       description="Close an issue by number, with an optional closing comment. Requires write token.",
                       source="github_api")
    registry.register("update_issue", client.update_issue, kind="builtin",
                       description="Update an issue's title, body, state, or labels in one call. Requires write token.",
                       source="github_api")
    registry.register("assign_issue", client.assign_issue, kind="builtin",
                       description="Assign one or more users to an issue. Requires write token.",
                       source="github_api")
    registry.register("add_comment", client.add_comment, kind="builtin",
                       description="Add a comment to an issue. Requires write token.",
                       source="github_api")
    registry.register("add_labels", client.add_labels, kind="builtin",
                       description="Add labels to an issue. Requires write token.",
                       source="github_api")
    registry.register("create_pull_request", client.create_pull_request, kind="builtin",
                       description="Open a pull request from a head branch into base. Requires write token.",
                       source="github_api")
    registry.register("merge_pull_request", client.merge_pull_request, kind="builtin",
                       description="Merge a pull request by number. merge_method: 'merge'/'squash'/'rebase'. Requires write token.",
                       source="github_api")
    registry.register("close_pull_request", client.close_pull_request, kind="builtin",
                       description="Close (without merging) a pull request. Requires write token.",
                       source="github_api")
