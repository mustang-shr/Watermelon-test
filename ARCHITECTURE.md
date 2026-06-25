# ARCHITECTURE.md

## 1. What does your memory system store, and why did you structure it that way?

A Kuzu embedded graph (`src/memory/schema.py`), not a vector store or flat log table.
Four node types — `Instruction`, `Step`, `Tool`, `Constraint` — map onto the brief's two
required layers: `Instruction`+`Step` is Execution Memory; `Tool`+`Constraint` is
Capability Memory (including real discovered constraints, e.g. GitHub's `/issues`
endpoint silently returning PRs too).

Why a graph: "which approaches worked for similar instruction patterns" is a traversal
(instruction → similar instructions → their decompositions → tools used → outcomes),
not a lookup — N hand-written SQL joins vs. one Cypher query per question. Real cost:
three Kuzu reserved words (`description`, `desc`, `order` — even as bound *parameter*
names) cost real debugging time, caught by running the schema, not assuming it.

Similarity is a swappable `EmbeddingProvider`: `LocalTFIDFEmbedder` (offline, used by
all 25 tests) or `NvidiaNIMEmbedder` (real semantic similarity, needed for the live
demo). Measured directly: TF-IDF scores genuine paraphrase at 0.16 vs. near-identical
phrasing at 0.77 — only catches "asked almost the same way." Documented, not hidden.

**Fixed this pass, caught by internal review**: `Tool.code` for a synthesized capability
was written but never read back — a fresh process could see the record but couldn't
call it, forcing full re-synthesis every restart, violating "memory must survive between
runs." `Agent.reload_capability_memory()` now reconstructs live callables from persisted
code on construction, through the same AST import-guard fresh synthesis uses — verified
by a test building a tool in one `Agent`, then constructing a second against the same DB
with no synthesis step, and confirming it's callable. Also fixed: `Tool.success_count`/
`failure_count` were written every run but read by nothing — the planner now receives
each tool's track record and is told to weigh it.

## 2. How does capability synthesis work in your implementation?

`src/tools/synthesis.py`. The planner emits a step with `needs_synthesis` instead of a
`tool_name`. The agent looks up a contract (signature, docstring, real test cases) and
runs `CapabilitySynthesisEngine.synthesize()`: generate → AST-check (no imports
allowed) → exec in a restricted namespace → run real tests → on failure, feed the actual
error back and retry (bounded, default 3) → register only on an actual pass.

Verified with a deliberately broken first attempt (wrong dict key, `KeyError`) the
harness caught, fed back, and corrected — attempt count asserted to be exactly 2, not
rigged to pass on attempt 1. Import-rejection separately verified against `import os`.
Two gap types are implemented: `duplicate_detection` (similarity comparison, injects
`similarity_fn`) and `time_to_close_by_label` (date-math aggregation, injects a date
parser instead — generated code still cannot `import datetime`, the guard is absolute,
so date math is pre-built and handed in). Two different kinds of gap, two different
injected capabilities — chosen deliberately to prove the pattern generalizes rather than
ship a second copy of the same trick.

**Bug the second gap type exposed, fixed this pass**: a synthesized tool's call
signature can't be known until synthesis finishes — a real planner can't pre-populate
`similarity_fn`/`parse_iso_to_epoch` into a step's kwargs in advance. The first gap type
only ever worked because test setup happened to pre-wire the right kwarg by hand; that
silently masked a missing mechanism. `Agent.run()` now merges each contract's declared
`call_time_kwargs` into the step automatically after a successful synthesis — verified
by rewriting the original test to NOT pre-wire anything and confirming it still passes.

**What is tested vs what requires a live key**: every test of this engine uses
`MockCodeGen` returning hand-written code. The harness — exec, AST guard, test runner,
repair loop — is fully verified. The live `NvidiaNIMCodeGen` path (model:
`meta/llama-3.1-70b-instruct`) requires a real `NVIDIA_API_KEY` to exercise; run
`python scripts/run_demo.py` with `.env` populated to confirm live synthesis. Test cases
are hand-written rather than LLM-generated — a deliberate scope call to keep the harness
verifiable offline.

## 3. What is your learning signal, and what does the agent do differently on run N vs run 1?

Before planning, the agent queries memory for similar past instructions; a
high-confidence match with a successful decomposition is passed to the planner with an
instruction to reuse that shape. Verified with asserted numbers: a first instruction
executed as 2 tool calls; a differently-phrased similar one matched it (similarity
0.847) and executed as 1.

Added this pass: a confidence score (`AgentRunResult.confidence`) from real run events
— status, synthesis attempt count, whether a proven precedent matched — with a reason
string built from what actually happened, not a template. Cheap given the structured
report already existed.

**What is verified vs live**: the measurement machinery is fully verified offline —
the test uses a scripted planner to prove the agent correctly measures and routes on
whatever plan it receives. The live NIM planner (`meta/llama-3.1-70b-instruct`) receives
a hardened `=== MEMORY DIRECTIVE ===` block when a past decomposition matches, explicitly
instructing it to reuse the exact step count. Run instruction 2 from DEMO.md twice to
confirm the call count drops live; the before/after numbers will be in the printed
`ExecutionReport`.
