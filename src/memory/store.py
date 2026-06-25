"""
Read/write interface over the memory graph. The agent/planner only ever talks
to this module - never to raw Cypher - so the schema can change without the
agent code changing.

The method that matters most for the "memory must change behaviour" requirement
is find_similar_past_instructions(): the planner calls this BEFORE decomposing
a new instruction. If a similar one exists with a recorded successful
decomposition, the planner reuses that decomposition instead of re-deriving it
from scratch - that's the actual behavioural change, not just a log entry.
"""

import time
import uuid
from dataclasses import dataclass, field

from .schema import init_schema
from .embeddings import EmbeddingProvider


@dataclass
class ExecutionRecord:
    instruction_id: str
    text: str
    steps: list[dict]              # [{description, status, tool_used, outcome, latency_ms, error_detail}]
    status: str                    # "success" | "partial" | "failed"
    total_api_calls: int
    total_time_ms: int
    constraints_discovered: list[str] = field(default_factory=list)


class MemoryStore:
    def __init__(self, db_path: str, embedder: EmbeddingProvider, similarity_threshold: float = 0.55):
        self.db, self.conn = init_schema(db_path)
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold

    def close(self) -> None:
        """Release the underlying Kuzu connection AND database file lock.

        History, because this took two attempts: the first fix only called
        conn.close() (MemoryStore never stored the Database object at all,
        only init_schema's local variable held it). That fix verified clean
        on Linux but the identical test still failed on Windows with the
        exact same lock error - meaning conn.close() alone does not reliably
        release the OS-level file lock on Windows, only on Linux. This
        version closes the Connection, then the Database, then forces a GC
        pass in case pybind11-style bindings hold an internal reference
        cycle that keeps the C++ object (and its OS file handle) alive past
        the logical close() call until Python's cyclic collector runs -
        refcounting alone wouldn't catch that, .close() being "called"
        doesn't guarantee the underlying resource was actually released.
        I cannot verify this resolves it on Windows from this environment -
        no Windows access here - this is the most defensible complete fix
        given the evidence, not a confirmed-working one yet."""
        import gc
        self.conn.close()
        self.db.close()
        gc.collect()

    # ---------- writes ----------

    def log_execution(self, record: ExecutionRecord) -> None:
        now = int(time.time())
        self.conn.execute(
            "CREATE (i:Instruction {id: $id, text: $text, timestamp: $ts, "
            "total_api_calls: $calls, total_time_ms: $time, status: $status})",
            {"id": record.instruction_id, "text": record.text, "ts": now,
             "calls": record.total_api_calls, "time": record.total_time_ms,
             "status": record.status},
        )

        for order, step in enumerate(record.steps):
            step_id = str(uuid.uuid4())
            self.conn.execute(
                "CREATE (s:Step {id: $id, step_text: $stext, status: $status, error_detail: $err})",
                {"id": step_id, "stext": step["description"], "status": step["status"],
                 "err": step.get("error_detail", "")},
            )
            self.conn.execute(
                "MATCH (i:Instruction {id: $iid}), (s:Step {id: $sid}) "
                "CREATE (i)-[:DECOMPOSED_INTO {step_order: $sorder}]->(s)",
                {"iid": record.instruction_id, "sid": step_id, "sorder": order},
            )
            tool_name = step.get("tool_used")
            if tool_name:
                self._ensure_tool_exists(tool_name, kind="builtin", source="unspecified")
                self._bump_tool_stat(tool_name, step["outcome"])
                self.conn.execute(
                    "MATCH (s:Step {id: $sid}), (t:Tool {name: $tname}) "
                    "CREATE (s)-[:USED_TOOL {outcome: $outcome, latency_ms: $latency}]->(t)",
                    {"sid": step_id, "tname": tool_name, "outcome": step["outcome"],
                     "latency": step.get("latency_ms", 0)},
                )

        for constraint_desc in record.constraints_discovered:
            cid = str(uuid.uuid4())
            self.conn.execute(
                "CREATE (c:Constraint {id: $id, constraint_text: $ctext, constraint_type: 'discovered', discovered_at: $ts})",
                {"id": cid, "ctext": constraint_desc, "ts": now},
            )

        # Link to similar past instructions (this is the edge find_similar reads back)
        self._link_similar_instructions(record.instruction_id, record.text)

    def get_synthesized_tools(self) -> list[dict]:
        """All tools with kind='synthesized' and their persisted source code -
        what a restarting process needs to reconstruct capability memory
        instead of re-synthesizing from scratch. This was previously dead
        data: written by register_tool, never read back by anything."""
        result = self.conn.execute(
            "MATCH (t:Tool {kind: 'synthesized'}) "
            "RETURN t.name, t.code, t.source, t.success_count, t.failure_count"
        )
        tools = []
        while result.has_next():
            row = result.get_next()
            tools.append({
                "name": row[0], "code": row[1], "source": row[2],
                "success_count": row[3], "failure_count": row[4],
            })
        return tools

    def get_tool_stats_all(self) -> dict[str, dict]:
        """All tool stats keyed by name - what the planner needs to actually
        weigh tool selection by track record, instead of that data sitting
        unused in the graph."""
        result = self.conn.execute(
            "MATCH (t:Tool) RETURN t.name, t.success_count, t.failure_count, t.kind"
        )
        stats = {}
        while result.has_next():
            row = result.get_next()
            stats[row[0]] = {"success_count": row[1], "failure_count": row[2], "kind": row[3]}
        return stats

    def register_tool(self, name: str, kind: str, source: str, code: str = "") -> None:
        exists = self.conn.execute(
            "MATCH (t:Tool {name: $name}) RETURN t.name", {"name": name}
        )
        if exists.has_next():
            return
        self.conn.execute(
            "CREATE (t:Tool {name: $name, kind: $kind, success_count: 0, "
            "failure_count: 0, source: $source, code: $code, created_at: $ts})",
            {"name": name, "kind": kind, "source": source, "code": code, "ts": int(time.time())},
        )

    def link_synthesis_trigger(self, instruction_id: str, tool_name: str) -> None:
        self.conn.execute(
            "MATCH (i:Instruction {id: $iid}), (t:Tool {name: $tname}) "
            "CREATE (i)-[:TRIGGERED_SYNTHESIS]->(t)",
            {"iid": instruction_id, "tname": tool_name},
        )

    # ---------- reads ----------

    def find_similar_past_instructions(self, text: str, top_k: int = 3) -> list[dict]:
        """Compare against all past instructions and return the most similar,
        with their recorded decomposition - this is what lets the planner skip
        re-deriving a plan it already knows works."""
        result = self.conn.execute("MATCH (i:Instruction) RETURN i.id, i.text, i.status")
        past = []
        while result.has_next():
            row = result.get_next()
            past.append({"id": row[0], "text": row[1], "status": row[2]})

        if not past:
            return []

        scored = []
        embedder_warned = False
        for p in past:
            try:
                sim = self.embedder.similarity(text, p["text"])
            except ValueError:
                # TF-IDF fails on a single-document vocabulary mismatch edge case
                sim = 0.0
            except Exception as e:
                # A broken/misconfigured embedder (wrong model ID, network issue,
                # auth failure) must not crash the whole run over a similarity
                # check - confirmed this was a real failure mode, not
                # hypothetical: a misconfigured NIM embedding model name took
                # down an entire agent.run() that had already completed its
                # actual task successfully, hiding a real success behind a
                # crash. Treat as "no match found" for this candidate, warn
                # once so the misconfiguration isn't silently invisible either.
                if not embedder_warned:
                    print(f"WARNING: similarity comparison failed ({type(e).__name__}: {e}). "
                          f"Memory-based plan reuse is not working until this is fixed. "
                          f"Continuing without it rather than crashing the run.")
                    embedder_warned = True
                sim = 0.0
            if sim >= self.similarity_threshold:
                scored.append({**p, "similarity": sim})

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        top = scored[:top_k]

        for match in top:
            match["decomposition"] = self._get_decomposition(match["id"])
        return top

    def get_tool_stats(self, name: str) -> dict | None:
        result = self.conn.execute(
            "MATCH (t:Tool {name: $name}) RETURN t.name, t.success_count, t.failure_count, t.kind",
            {"name": name},
        )
        if not result.has_next():
            return None
        row = result.get_next()
        return {"name": row[0], "success_count": row[1], "failure_count": row[2], "kind": row[3]}

    def get_instruction_history_stats(self, similar_instruction_ids: list[str]) -> dict:
        """Aggregate timing/call-count across a set of past similar runs - this
        is the raw material for the before/after learning-signal numbers."""
        if not similar_instruction_ids:
            return {"count": 0}
        result = self.conn.execute(
            "MATCH (i:Instruction) WHERE i.id IN $ids "
            "RETURN i.total_api_calls, i.total_time_ms",
            {"ids": similar_instruction_ids},
        )
        calls, times = [], []
        while result.has_next():
            row = result.get_next()
            calls.append(row[0])
            times.append(row[1])
        return {
            "count": len(calls),
            "avg_api_calls": sum(calls) / len(calls) if calls else 0,
            "avg_time_ms": sum(times) / len(times) if times else 0,
        }

    # ---------- internal ----------

    def _get_decomposition(self, instruction_id: str) -> list[dict]:
        result = self.conn.execute(
            "MATCH (i:Instruction {id: $iid})-[r:DECOMPOSED_INTO]->(s:Step) "
            "RETURN s.step_text, s.status, r.step_order ORDER BY r.step_order",
            {"iid": instruction_id},
        )
        steps = []
        while result.has_next():
            row = result.get_next()
            steps.append({"description": row[0], "status": row[1], "order": row[2]})
        return steps

    def _ensure_tool_exists(self, name: str, kind: str, source: str) -> None:
        self.register_tool(name, kind, source)

    def _bump_tool_stat(self, name: str, outcome: str) -> None:
        field_name = "success_count" if outcome == "success" else "failure_count"
        self.conn.execute(
            f"MATCH (t:Tool {{name: $name}}) SET t.{field_name} = t.{field_name} + 1",
            {"name": name},
        )

    def _link_similar_instructions(self, new_id: str, new_text: str) -> None:
        similar = self.find_similar_past_instructions(new_text, top_k=5)
        for match in similar:
            if match["id"] == new_id:
                continue
            self.conn.execute(
                "MATCH (a:Instruction {id: $a}), (b:Instruction {id: $b}) "
                "CREATE (a)-[:SIMILAR_TO {similarity: $sim}]->(b)",
                {"a": new_id, "b": match["id"], "sim": match["similarity"]},
            )
