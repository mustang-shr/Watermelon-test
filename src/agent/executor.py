"""
Executes a Plan (ordered list of Steps) against the ToolRegistry.

Partial-failure contract (this is what the brief means by "must not silently
produce a half-complete result"): if step N fails, steps N+1.. are NOT
executed (default: sequential dependency assumed unless a step is marked
independent=True) - and the returned ExecutionReport says explicitly which
steps ran, which one failed and why, and which were never attempted. A
caller that only looks at `.status` already gets "partial", not "success" -
but the per-step detail is there for anyone who wants the specifics, so
nothing is hidden, just summarized.
"""

from dataclasses import dataclass, field
import time


@dataclass
class Step:
    description: str
    tool_name: str
    kwargs: dict = field(default_factory=dict)
    independent: bool = False
    needs_synthesis: str | None = None
    inject_prev_output_as: str | None = None  # if set, the previous step's output is
                                               # injected into this step's kwargs under
                                               # this key name before execution.
                                               # Used when a synthesis step needs data
                                               # produced by the preceding tool call
                                               # (e.g. list_issues → find_duplicate_issues
                                               # needs the issue list as 'issues').


@dataclass
class StepResult:
    step: Step
    status: str          # "success" | "failed" | "skipped"
    output: object = None
    error: str | None = None
    latency_ms: int = 0


@dataclass
class ExecutionReport:
    instruction_text: str
    status: str           # "success" | "partial" | "failed"
    step_results: list[StepResult]
    total_api_calls: int
    total_time_ms: int

    def summary(self) -> str:
        lines = [f"Instruction: {self.instruction_text}", f"Overall status: {self.status}"]
        for i, r in enumerate(self.step_results, 1):
            lines.append(f"  Step {i} [{r.status}]: {r.step.description}")
            if r.status == "failed":
                lines.append(f"    -> reason: {r.error}")
            if r.status == "skipped":
                lines.append(f"    -> not attempted: blocked by an earlier failure")
        lines.append(f"Total tool calls: {self.total_api_calls}, total time: {self.total_time_ms}ms")
        return "\n".join(lines)


class Executor:
    def __init__(self, registry):
        self.registry = registry

    def execute(self, instruction_text: str, plan: list[Step]) -> ExecutionReport:
        results: list[StepResult] = []
        blocked = False
        total_calls = 0
        start = time.time()

        for i, step in enumerate(plan):
            if blocked and not step.independent:
                results.append(StepResult(step=step, status="skipped"))
                continue

            # Output chaining: if this step declares it needs the previous step's
            # output as a named argument, inject it now before calling the tool.
            # This is how list_issues → find_duplicate_issues works: the issues
            # list produced by step 1 is passed as the 'issues' kwarg to step 2.
            # The key is declared in inject_prev_output_as, set by agent.run()
            # after synthesis completes - the planner can't set it because the
            # synthesized tool doesn't exist at plan time.
            effective_kwargs = dict(step.kwargs)
            if step.inject_prev_output_as and i > 0 and results[i - 1].status == "success":
                effective_kwargs[step.inject_prev_output_as] = results[i - 1].output

            tool_result = self.registry.call(step.tool_name, **effective_kwargs)
            total_calls += 1

            if tool_result.success:
                results.append(StepResult(
                    step=step, status="success", output=tool_result.output,
                    latency_ms=tool_result.latency_ms,
                ))
            else:
                results.append(StepResult(
                    step=step, status="failed", error=tool_result.error,
                    latency_ms=tool_result.latency_ms,
                ))
                blocked = True

        total_time_ms = int((time.time() - start) * 1000)

        statuses = {r.status for r in results}
        if "failed" not in statuses and "skipped" not in statuses:
            overall = "success"
        elif all(r.status in ("failed", "skipped") for r in results):
            overall = "failed"
        else:
            overall = "partial"

        return ExecutionReport(
            instruction_text=instruction_text,
            status=overall,
            step_results=results,
            total_api_calls=total_calls,
            total_time_ms=total_time_ms,
        )
