# DEMO.md

Three instructions, increasing complexity, run live against a real (disposable) test
repo no mocks.

## 1. Simple single tool call
**"Create an issue titled 'Test issue from autonomous agent' with the body 'Created to
verify the agent core works end to end.'"**

Expected: planner emits a 1-step plan using the existing `create_issue` tool. Structured
execution report shows 1 successful step, 1 API call. Confirms the basic
instruction→execution path before anything more interesting happens.

## 2. Compound multi-step, exercises memory
**"Find all open issues with no assignee, group them by label, and post a summary
comment on the most recently created one."**

Run this twice, a few minutes apart, with slightly different phrasing the second time
(e.g. swap "no assignee" for "unassigned"). Expected: run 1 decomposes from scratch.
Run 2's planner call is given the run-1 decomposition from memory (similarity score
shown in the run log) and should reuse it rather than re-deriving show the real
before/after numbers from both `ExecutionReport`s, not a claim.

## 3. Novel forces capability synthesis
**"Find duplicate issues among the open issues based on title similarity and report
which ones look like duplicates."**

No existing tool does this GitHub has no "are these the same issue" endpoint.
Expected: planner flags the step `needs_synthesis: duplicate_detection`, the synthesis
engine generates a similarity-based detector, tests it against real fetched issues, and
(if it fails its self-test) retries with the error fed back before registering. Show the
generated code and the attempt count in the run log this is the one to slow down on
in the actual call, since it's the requirement most candidates do the cheap version of.

---

**Before the call**: run all three once locally to confirm current state of the real
repo (issue numbers will differ from any rehearsal run) and to confirm the synthesized
`find_duplicate_issues` tool persisted into memory restart the process and run
instruction 3 again to show capability memory survives a restart, not just an in-memory
cache.
