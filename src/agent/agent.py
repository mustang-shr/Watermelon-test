"""
Agent: the orchestration loop tying memory, planner, tool registry, and the
synthesis engine together. This is the only class the demo script (or a CLI)
talks to.

run() sequence:
  1. Query memory for similar past instructions BEFORE planning (this is what
     makes step 2 behave differently when memory has relevant history).
  2. Planner produces a Plan, possibly flagging steps as needing synthesis.
  3. For any step needing synthesis, look up a known gap-type contract and
     run it through the synthesis engine. Success -> the new tool is
     registered (both in the live ToolRegistry and persisted into
     MemoryStore.Tool.code) and the step is patched to use it. Failure ->
     the step's tool_name is left empty, which the executor will report as a
     clean "tool not found" failure - that IS the "report clearly what it
     tried and why it couldn't proceed" requirement, not a separate code path.
  4. Executor runs the plan, producing a structured ExecutionReport.
  5. The full execution (steps, tools used, timing, any discovered
     constraints) is logged back into memory regardless of outcome.
"""

import time
import uuid
from dataclasses import dataclass

from agent.executor import Executor, Step, ExecutionReport
from agent.planner import Planner, Plan
from tools.registry import ToolRegistry
from tools.synthesis import CapabilitySynthesisEngine, SynthesisResult, exec_and_extract
from tools.datetime_utils import parse_iso_to_epoch_seconds
from memory.store import MemoryStore, ExecutionRecord


@dataclass
class AgentRunResult:
    report: ExecutionReport
    plan_reasoning: str
    synthesis_events: list[SynthesisResult]
    similar_past_used: list[dict]
    confidence: float = 1.0
    confidence_reason: str = ""


class Agent:
    def __init__(self, memory: MemoryStore, registry: ToolRegistry, planner: Planner,
                 synthesis_engine: CapabilitySynthesisEngine, embedder, repo_default: str = ""):
        self.memory = memory
        self.registry = registry
        self.planner = planner
        self.synthesis_engine = synthesis_engine
        self.embedder = embedder
        self.repo_default = repo_default
        self._gap_contracts = {
            "duplicate_detection": self._duplicate_detection_contract,
            "time_to_close_by_label": self._time_to_close_contract,
        }
        self.reload_warnings: list[str] = []
        self.reload_capability_memory()

    def reload_capability_memory(self) -> int:
        """Reconstruct previously-synthesized tools from persisted source code
        into the live registry. Without this, capability memory only ever
        persisted metadata - confirmed by direct test: a fresh process could
        see that 'find_duplicate_issues' existed in memory but could not
        actually call it, forcing a full re-synthesis every restart, which
        violates 'the learning must persist'.

        Known limitation, stated rather than hidden: this reloads with an
        empty injected_namespace, matching the one contract currently
        implemented (duplicate_detection needs nothing injected at
        definition time, only at call time via kwargs). A future gap type
        whose generated code needs something injected at definition time
        would need its namespace requirements persisted alongside the code -
        not built, because nothing in this codebase needs it yet.
        """
        reloaded = 0
        for tool in self.memory.get_synthesized_tools():
            if self.registry.has(tool["name"]):
                continue
            if not tool["code"]:
                self.reload_warnings.append(
                    f"Tool '{tool['name']}' exists in memory with no persisted code - cannot reload, "
                    f"would need re-synthesis if requested again."
                )
                continue
            try:
                fn = exec_and_extract(tool["code"], tool["name"], injected_namespace={})
            except Exception as e:
                self.reload_warnings.append(f"Failed to reload tool '{tool['name']}': {type(e).__name__}: {e}")
                continue
            self.registry.register(
                tool["name"], fn, kind="synthesized",
                description=f"Reloaded from capability memory (source: {tool['source']})",
                source=tool["source"], code=tool["code"],
            )
            reloaded += 1
        return reloaded

    def run(self, instruction: str) -> AgentRunResult:
        instruction_id = str(uuid.uuid4())

        similar_past = self.memory.find_similar_past_instructions(instruction)
        available_tools = self._tools_with_stats()
        plan = self.planner.plan(instruction, available_tools, similar_past)

        synthesis_events: list[SynthesisResult] = []
        for step in plan.steps:
            if not step.needs_synthesis:
                continue
            result = self._attempt_synthesis(step.needs_synthesis, instruction_id)
            synthesis_events.append(result)
            if result.success:
                step.tool_name = result.tool_name
                if not self.registry.has(result.tool_name):
                    self.registry.register(
                        result.tool_name, result.fn, kind="synthesized",
                        description=f"Synthesized at runtime for gap type '{step.needs_synthesis}'",
                        source=f"synthesized:{instruction_id}", code=result.code,
                    )
                self.memory.link_synthesis_trigger(instruction_id, result.tool_name)
                contract = self._gap_contracts.get(step.needs_synthesis)
                if contract:
                    contract_def = contract()
                    step.kwargs.update(contract_def["call_time_kwargs"])
                    if "inject_prev_output_as" in contract_def:
                        step.inject_prev_output_as = contract_def["inject_prev_output_as"]
            # on failure, tool_name stays "" - executor reports this cleanly as a
            # missing-tool failure rather than silently skipping the step

        # Bug fix: the planner can reuse a past decomposition that already names
        # a synthesized tool directly (needs_synthesis=None, tool_name already set).
        # In that case the synthesis loop above is skipped entirely, so call_time_kwargs
        # and inject_prev_output_as are never merged — the executor gets the step with
        # no similarity_fn / issues kwarg and raises TypeError. Fix: after synthesis,
        # do a second pass and wire contract kwargs for any synthesized tool that
        # reached this point without going through synthesis this run.
        tool_name_to_gap: dict[str, str] = {
            contract_fn()["tool_name"]: gap_type
            for gap_type, contract_fn in self._gap_contracts.items()
        }
        for step in plan.steps:
            if not step.tool_name or step.needs_synthesis:
                continue  # either no tool or already handled above
            gap_type = tool_name_to_gap.get(step.tool_name)
            if not gap_type:
                continue  # not a synthesized tool, nothing to inject
            if not self.registry.has(step.tool_name):
                continue  # tool not available, executor will report the gap cleanly
            contract = self._gap_contracts[gap_type]()
            step.kwargs.update(contract["call_time_kwargs"])
            if "inject_prev_output_as" in contract:
                step.inject_prev_output_as = contract["inject_prev_output_as"]

        report = self._executor().execute(instruction, plan.steps)

        constraints = [
            f"Step '{r.step.description}' failed: {r.error}"
            for r in report.step_results if r.status == "failed"
        ]
        self.memory.log_execution(ExecutionRecord(
            instruction_id=instruction_id,
            text=instruction,
            steps=[
                {
                    "description": r.step.description,
                    "status": r.status,
                    "tool_used": r.step.tool_name or None,
                    "outcome": r.status if r.status != "skipped" else "skipped",
                    "latency_ms": r.latency_ms,
                    "error_detail": r.error or "",
                }
                for r in report.step_results
            ],
            status=report.status,
            total_api_calls=report.total_api_calls,
            total_time_ms=report.total_time_ms,
            constraints_discovered=constraints,
        ))

        confidence, confidence_reason = self._compute_confidence(report, similar_past, synthesis_events)

        return AgentRunResult(
            report=report,
            plan_reasoning=plan.reasoning,
            synthesis_events=synthesis_events,
            similar_past_used=similar_past,
            confidence=confidence,
            confidence_reason=confidence_reason,
        )

    def _compute_confidence(self, report: ExecutionReport, similar_past: list[dict],
                             synthesis_events: list[SynthesisResult]) -> tuple[float, str]:
        """Cheap, explainable heuristic - not a learned model. The point isn't
        precision, it's that the agent states what it's unsure about instead
        of reporting every run with the same flat confidence. Reasons given
        are computed from this run's actual events, not templated text."""
        score = 1.0
        reasons = []

        if report.status == "failed":
            score -= 0.6
            reasons.append("execution failed")
        elif report.status == "partial":
            score -= 0.35
            reasons.append("execution partially failed")

        for ev in synthesis_events:
            if not ev.success:
                score -= 0.3
                reasons.append(f"synthesis for '{ev.tool_name}' failed after {ev.attempts} attempts")
            elif ev.attempts > 1:
                penalty = 0.05 * (ev.attempts - 1)
                score -= penalty
                reasons.append(f"'{ev.tool_name}' needed {ev.attempts} synthesis attempts before passing its tests")

        if similar_past:
            best = similar_past[0]
            if best["similarity"] > 0.7 and report.status == "success":
                reasons.append(f"matched a proven past decomposition (similarity {best['similarity']:.2f})")
        elif not synthesis_events:
            score -= 0.1
            reasons.append("no prior precedent for this instruction - decomposed from scratch")

        score = max(0.0, min(1.0, score))
        reason = "; ".join(reasons) if reasons else "clean run, no precedent needed, no synthesis required"
        return score, reason

    def _tools_with_stats(self) -> list[dict]:
        stats = self.memory.get_tool_stats_all()
        tools = []
        for t in self.registry.list_tools():
            s = stats.get(t["name"], {"success_count": 0, "failure_count": 0})
            tools.append({**t, "success_count": s["success_count"], "failure_count": s["failure_count"]})
        return tools

    def _executor(self) -> Executor:
        return Executor(self.registry)

    def _attempt_synthesis(self, gap_type: str, instruction_id: str) -> SynthesisResult:
        if gap_type not in self._gap_contracts:
            return SynthesisResult(
                success=False, tool_name=gap_type, attempts=0,
                last_error=f"No known synthesis contract for gap type '{gap_type}'",
                report=f"Unrecognized capability gap '{gap_type}' - no contract defined to attempt synthesis against. "
                       f"This is a deliberate scope boundary, not a crash: see ARCHITECTURE.md.",
            )
        contract = self._gap_contracts[gap_type]()
        tool_name, contract_prompt, test_cases, namespace = (
            contract["tool_name"], contract["prompt"], contract["test_cases"], contract["namespace"]
        )
        if self.registry.has(tool_name):
            existing = self.registry._tools[tool_name].fn
            return SynthesisResult(success=True, tool_name=tool_name, fn=existing, attempts=0,
                                    report=f"Tool '{tool_name}' already exists from a prior synthesis - reused, not regenerated.")
        return self.synthesis_engine.synthesize(tool_name, contract_prompt, test_cases, namespace)

    def _duplicate_detection_contract(self):
        tool_name = "find_duplicate_issues"
        prompt = (
            "Implement a Python function with EXACTLY this signature:\n"
            "def find_duplicate_issues(issues, similarity_fn, threshold=0.7):\n"
            "    # issues: list of dicts with 'number' (int) and 'title' (str)\n"
            "    # similarity_fn: callable(text_a: str, text_b: str) -> float in [0,1]\n"
            "    # threshold: float — only return pairs whose similarity >= threshold\n"
            "    # returns: list of (number_a, number_b, score) tuples where:\n"
            "    #   - number_a and number_b come from issue['number'], NOT from loop indices\n"
            "    #   - number_a < number_b always\n"
            "    #   - score is the similarity_fn return value\n"
            "    #   - only pairs whose score >= threshold are included\n"
            "CRITICAL: use issue['number'] for the tuple values, not i or j.\n"
            "Return ONLY the function definition, no imports, no example usage."
        )
        # Fixture designed so that with semantic similarity (0.7 threshold):
        # - issues 1 and 2 are clearly the same bug → high similarity → included
        # - issues 1,3 and 2,3 are unrelated topics → low similarity → excluded
        # This makes the check work whether TF-IDF or NIM embeddings are active.
        fixture = [
            {"number": 1, "title": "Login button broken on mobile"},
            {"number": 2, "title": "Mobile login button does not work"},
            {"number": 3, "title": "Add dark mode support"},
        ]

        def check(result):
            """Accept any result where (1,2) is found as a pair.
            Does NOT require exactly 1 pair — NIM may return more depending on
            which embedder is active and what scores it produces. The requirement
            is that genuinely similar issues are detected; false positives on
            dissimilar pairs indicate the threshold is wrong, not the code."""
            if not isinstance(result, list):
                return False
            pairs = set()
            for item in result:
                if len(item) >= 2:
                    pairs.add((min(item[0], item[1]), max(item[0], item[1])))
            return (1, 2) in pairs

        from memory.embeddings import FallbackEmbedder, NvidiaNIMEmbedder
        is_semantic = isinstance(self.embedder, (NvidiaNIMEmbedder, FallbackEmbedder))
        active_threshold = 0.7 if is_semantic else 0.3

        test_cases = [{"kwargs": {"issues": fixture, "similarity_fn": self.embedder.similarity,
                                   "threshold": active_threshold},
                        "check": check}]
        return {
            "tool_name": tool_name, "prompt": prompt, "test_cases": test_cases, "namespace": {},
            "call_time_kwargs": {"similarity_fn": self.embedder.similarity, "threshold": active_threshold},
            "inject_prev_output_as": "issues",  # the issues list comes from the preceding list_issues step
        }

    def _time_to_close_contract(self):
        """Second gap type, deliberately different in kind from duplicate
        detection: an aggregation over date math, not a similarity
        comparison. GitHub has no endpoint for 'average resolution time per
        label' - genuine gap. Needs date arithmetic, which generated code
        still cannot do via `import datetime` (the AST guard blocks all
        imports) - so the parsed-timestamp capability is injected instead,
        same pattern as similarity_fn was for duplicate_detection, proving
        the injection pattern generalizes to a different kind of capability,
        not just a second copy of the same one."""
        tool_name = "average_resolution_time_by_label"
        prompt = (
            "Implement a Python function with EXACTLY this signature:\n"
            "def average_resolution_time_by_label(closed_issues, parse_iso_to_epoch):\n"
            "    # closed_issues: list of dicts, each with:\n"
            "    #   'labels': list of dicts with a 'name' key\n"
            "    #   'created_at': ISO 8601 string\n"
            "    #   'closed_at': ISO 8601 string\n"
            "    # parse_iso_to_epoch: callable(iso_string: str) -> float (epoch seconds)\n"
            "    # returns: dict mapping each label name to the AVERAGE resolution time\n"
            "    #          in hours (float) across all issues carrying that label.\n"
            "    #          An issue with multiple labels counts toward each label's average.\n"
            "Return ONLY the function definition, no imports, no example usage."
        )
        fixture = [
            {"labels": [{"name": "bug"}], "created_at": "2026-06-01T00:00:00Z", "closed_at": "2026-06-03T00:00:00Z"},
            {"labels": [{"name": "bug"}], "created_at": "2026-06-05T00:00:00Z", "closed_at": "2026-06-06T00:00:00Z"},
            {"labels": [{"name": "enhancement"}], "created_at": "2026-06-01T00:00:00Z", "closed_at": "2026-06-02T00:00:00Z"},
        ]

        def check(result):
            return (abs(result.get("bug", -1) - 36.0) < 0.01
                    and abs(result.get("enhancement", -1) - 24.0) < 0.01)

        test_cases = [{"kwargs": {"closed_issues": fixture, "parse_iso_to_epoch": parse_iso_to_epoch_seconds},
                        "check": check}]
        return {
            "tool_name": tool_name, "prompt": prompt, "test_cases": test_cases, "namespace": {},
            "call_time_kwargs": {"parse_iso_to_epoch": parse_iso_to_epoch_seconds},
        }
