"""
Planner: instruction (+ available tools + similar-past-instruction context
from memory) -> Plan (ordered Steps).

This is the one place "memory changes behaviour" is most visible: when
memory.find_similar_past_instructions() returns a high-confidence match with
a successful past decomposition, the planner is instructed to reuse that
shape rather than re-deriving steps from first principles. The NIM prompt
builder below makes that instruction explicit and conditional - it only
fires when a real match exists.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import re
import warnings

from agent.executor import Step


@dataclass
class Plan:
    steps: list[Step]
    reasoning: str = ""


class Planner(ABC):
    @abstractmethod
    def plan(self, instruction: str, available_tools: list[dict],
             similar_past: list[dict]) -> Plan:
        raise NotImplementedError


class MockPlanner(Planner):
    """Deterministic planner for testing the agent core's orchestration logic
    (memory lookup -> plan -> execute -> synthesis-on-gap -> log) without a
    live LLM. Takes a sequence of canned Plans, returned in order per call."""

    def __init__(self, plans: list[Plan]):
        self.plans = plans
        self.call_count = 0

    def plan(self, instruction: str, available_tools: list[dict],
             similar_past: list[dict]) -> Plan:
        result = self.plans[min(self.call_count, len(self.plans) - 1)]
        self.call_count += 1
        return result


class RuleBasedPlanner(Planner):
    """Deterministic fallback planner for offline or unreachable environments.

    This keeps the main demo flow usable even when the hosted planner is down,
    while still being explicit about what it inferred from the instruction.

    Important honesty note for the demo/interview: this planner ONLY covers
    the exact phrasings it's been written to recognize ("create an issue",
    "find duplicate issues", "list open issues"). It does not generalize to
    novel compound instructions - that's the live NIM planner's job. This
    exists purely so a network hiccup doesn't take down an otherwise-working
    demo, not as a substitute for the actual reasoning requirement.
    """

    def plan(self, instruction: str, available_tools: list[dict],
             similar_past: list[dict]) -> Plan:
        instruction_l = instruction.lower()
        steps = []

        if "create an issue" in instruction_l or "create issue" in instruction_l:
            title, body = self._extract_create_issue_fields(instruction)
            steps.append(Step(
                description=f"create a GitHub issue titled '{title}'",
                tool_name="create_issue",
                kwargs={"title": title, "body": body},
            ))
        elif "close issue" in instruction_l:
            number = self._extract_issue_number(instruction)
            comment = self._extract_quoted(instruction, ["comment", "with"])
            steps.append(Step(
                description=f"close issue #{number}",
                tool_name="close_issue",
                kwargs={"issue_number": number, **( {"comment": comment} if comment else {})},
            ))
        elif "assign issue" in instruction_l or "assign" in instruction_l and "issue" in instruction_l:
            number = self._extract_issue_number(instruction)
            steps.append(Step(
                description=f"assign issue #{number}",
                tool_name="assign_issue",
                kwargs={"issue_number": number, "assignees": []},
            ))
        elif "comment on issue" in instruction_l or "add a comment" in instruction_l:
            number = self._extract_issue_number(instruction)
            body = self._extract_quoted(instruction, ["body", "comment", "saying", "with"])
            steps.append(Step(
                description=f"add comment to issue #{number}",
                tool_name="add_comment",
                kwargs={"issue_number": number, "body": body or instruction},
            ))
        elif "find duplicate issues" in instruction_l or "duplicate" in instruction_l and "issue" in instruction_l:
            steps.append(Step(
                description="list open issues for duplicate detection",
                tool_name="list_issues",
                kwargs={"state": "open"},
            ))
            steps.append(Step(
                description="find duplicate issues by title similarity",
                tool_name="",
                kwargs={},
                needs_synthesis="duplicate_detection",
            ))
        elif "list open issues" in instruction_l or "find all open issues" in instruction_l or "show open issues" in instruction_l:
            steps.append(Step(
                description="list open issues",
                tool_name="list_issues",
                kwargs={"state": "open"},
            ))
        elif "list pull requests" in instruction_l or "show pull requests" in instruction_l or "list prs" in instruction_l:
            steps.append(Step(
                description="list open pull requests",
                tool_name="list_pull_requests",
                kwargs={"state": "open"},
            ))
        else:
            steps.append(Step(
                description="attempt to execute instruction with available tools",
                tool_name="",
                kwargs={},
            ))

        return Plan(steps=steps, reasoning="Rule-based fallback planner (NIM unavailable). Handles: create/close/assign issue, add comment, duplicate detection, list issues/PRs.")

    def _extract_create_issue_fields(self, instruction: str) -> tuple[str, str]:
        def _clean(value: str) -> str:
            return value.strip().strip("'\"")

        def _find_value(patterns):
            for pattern in patterns:
                match = re.search(pattern, instruction, flags=re.IGNORECASE)
                if match:
                    return _clean(match.group(1))
            return ""

        title = _find_value([
            r"\bissue\s+(?:title|titles?)\s*['\"]([^'\"]+)['\"]",
            r"\btitled?\s*['\"]([^'\"]+)['\"]",
            r"\btitle\s*['\"]([^'\"]+)['\"]",
            r"\bissue\s+['\"]([^'\"]+)['\"]",
        ])
        body = _find_value([
            r"\bbody\s*['\"]([^'\"]+)['\"]",
            r"\bwith\s+the\s+body\s*['\"]([^'\"]+)['\"]",
        ])
        return title or instruction.strip(), body or ""

    def _extract_issue_number(self, instruction: str) -> int:
        # handles: 'issue #7', 'issue number 7', 'issue no. 7', 'close #7', '#7'
        match = re.search(r"(?:issue\s*(?:number|no\.?)?\s*#?|#)(\d+)", instruction, re.IGNORECASE)
        return int(match.group(1)) if match else 1

    def _extract_quoted(self, instruction: str, hint_words: list[str]) -> str:
        """Extract the first quoted string that appears after one of the hint words."""
        for word in hint_words:
            pattern = rf"\b{word}\b[^'\"]*['\"]([^'\"]+)['\"]"
            match = re.search(pattern, instruction, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""


class FallbackPlanner(Planner):
    """Run a primary planner first, then fall back to a deterministic planner.

    This is useful for demo scripts and local environments where the hosted API
    may be rate limited, timed out, or otherwise unavailable.
    """

    def __init__(self, primary: Planner, fallback: Planner):
        self.primary = primary
        self.fallback = fallback

    def plan(self, instruction: str, available_tools: list[dict],
             similar_past: list[dict]) -> Plan:
        try:
            return self.primary.plan(instruction, available_tools, similar_past)
        except Exception as exc:
            warnings.warn(
                f"Primary planner failed ({exc.__class__.__name__}: {exc}); "
                f"falling back to rule-based planner.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self.fallback.plan(instruction, available_tools, similar_past)


class NvidiaNIMPlanner(Planner):
    """Production planner via NIM chat completions. Default model:
    meta/llama-3.1-70b-instruct — verified active on integrate.api.nvidia.com
    as of June 2026. JSON-mode (response_format=json_object) varies by model;
    this implementation prompts for JSON and parses defensively, which works
    across all NIM models without relying on API-enforced schema."""

    def __init__(self, api_key: str, model: str = "meta/llama-3.1-70b-instruct",
                 base_url: str = "https://integrate.api.nvidia.com/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def plan(self, instruction: str, available_tools: list[dict],
             similar_past: list[dict]) -> Plan:
        import requests

        def _tool_line(t):
            calls = t.get("success_count", 0) + t.get("failure_count", 0)
            if calls == 0:
                return f"  - {t['name']}: {t['description']}"
            rate = t["success_count"] / calls
            return (f"  - {t['name']}: {t['description']} "
                    f"[track record: {t['success_count']}/{calls}, {rate:.0%}]")

        read_tools  = [t for t in available_tools if t["name"] in
                       ("list_issues","search_issues","get_issue","list_pull_requests","get_rate_limit")]
        write_tools = [t for t in available_tools if t["name"] not in
                       ("list_issues","search_issues","get_issue","list_pull_requests","get_rate_limit")
                       and t.get("kind") == "builtin"]
        synth_tools = [t for t in available_tools if t.get("kind") == "synthesized"]

        tools_section = "READ TOOLS (no token required):\n"
        tools_section += "\n".join(_tool_line(t) for t in read_tools) or "  (none)"
        tools_section += "\n\nWRITE TOOLS (token required — will fail without GITHUB_TOKEN in .env):\n"
        tools_section += "\n".join(_tool_line(t) for t in write_tools) or "  (none)"
        if synth_tools:
            tools_section += "\n\nSYNTHESIZED TOOLS (generated this session and already tested):\n"
            tools_section += "\n".join(_tool_line(t) for t in synth_tools)

        memory_hint = ""
        if similar_past:
            best = similar_past[0]
            step_count = len(best["decomposition"])
            steps_desc = " -> ".join(s["description"] for s in best["decomposition"])
            memory_hint = (
                f"\n=== MEMORY DIRECTIVE (similarity={best['similarity']:.2f}) ===\n"
                f"A past instruction was successfully decomposed into {step_count} step(s).\n"
                f"Past instruction: '{best['text']}'\n"
                f"Past decomposition ({step_count} steps): {steps_desc}\n"
                f"\nDIRECTIVE: Reuse this EXACT decomposition. Do NOT re-derive from scratch.\n"
                f"Only modify a step if the new instruction explicitly requires something different.\n"
                f"Reusing memory is what makes the agent learn - this is measured as a call-count improvement.\n"
                f"=== END MEMORY DIRECTIVE ===\n"
            )

        prompt = f"""You are a planning agent that decomposes natural language instructions
into executable steps for a GitHub automation agent.

AVAILABLE TOOLS:
{tools_section}
{memory_hint}
SYNTHESIS RULE: If a step requires a capability no listed tool provides, you may
set "tool_name" to null and "needs_synthesis" to one of these EXACT gap type strings:
  - "duplicate_detection": find issues with similar titles (no GitHub endpoint for this)
  - "time_to_close_by_label": compute average resolution time grouped by label
CRITICAL: Only use gap types from the list above. Do NOT invent new gap type strings.
If a capability is not in the list and no existing tool covers it, break the instruction
into steps that use only listed tools, or omit that step and explain in "reasoning".

STEP RULES:
- Use exact tool names from the list above (case-sensitive)
- Multi-step instructions should have one step per distinct action
- kwargs must contain only keys the tool expects; do not invent parameters
- Prefer tools with higher track records for equivalent tasks

INSTRUCTION: {instruction}

Return ONLY valid JSON (no prose, no markdown fences, no comments) matching exactly:
{{
  "reasoning": "one or two sentences explaining the decomposition choice",
  "steps": [
    {{
      "description": "plain English description of this step",
      "tool_name": "exact_tool_name_or_null_if_synthesis_needed",
      "kwargs": {{}},
      "needs_synthesis": null
    }}
  ]
}}"""

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.1},
                timeout=120,
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"NIM request timed out after 120s (model='{self.model}'). This usually means either "
                f"the model ID doesn't exist/resolve, the model is genuinely slow/cold-starting, or "
                f"something on your network is interfering with the connection. Run "
                f"`python scripts/check_nim_connection.py` to isolate which."
            ) from None
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Could not connect to NIM at all: {e}. Run `python scripts/check_nim_connection.py` "
                f"to check DNS/network/firewall issues separately from the rest of the agent."
            ) from None
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())

        steps = [
            Step(
                description=s["description"],
                tool_name=s.get("tool_name") or "",
                kwargs=s.get("kwargs", {}),
                needs_synthesis=s.get("needs_synthesis"),
            )
            for s in parsed["steps"]
        ]
        return Plan(steps=steps, reasoning=parsed.get("reasoning", ""))
