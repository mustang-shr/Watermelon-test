"""
Memory schema for the Autonomous Platform Agent.

Why a graph instead of flat tables:
The brief's evaluation question is "which approaches worked, which failed, and why"
and "tool selection improves as the agent tracks which capabilities work for which
instruction patterns." Both of those are traversal queries (instruction -> steps it
was decomposed into -> tools those steps used -> outcomes), not lookup queries. A
flat execution_log table can answer "what happened on run N" but answering "what
usually works for instructions shaped like this one" requires walking relationships
across many past runs - that's a graph query, not a join you want to hand-write
five different ways.

Two node families map directly to the two required memory layers:
  - Instruction, Step                -> Execution Memory (what was done, how, how long, did it work)
  - Tool, Constraint                 -> Capability Memory (what the agent can do, what it learned about doing it)

SIMILAR_TO is populated lazily by the planner when it recognizes a new instruction
resembles a past one (via embedding similarity over instruction text) - this is the
edge the "use memory to change behaviour" requirement hangs on: the planner queries
along this edge before decomposing from scratch.
"""

import kuzu
import os


SCHEMA_STATEMENTS = [
    # --- Node tables ---
    """CREATE NODE TABLE IF NOT EXISTS Instruction(
        id STRING,
        text STRING,
        timestamp INT64,
        total_api_calls INT64,
        total_time_ms INT64,
        status STRING,
        PRIMARY KEY(id)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Step(
        id STRING,
        step_text STRING,
        status STRING,
        error_detail STRING,
        PRIMARY KEY(id)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Tool(
        name STRING,
        kind STRING,
        success_count INT64,
        failure_count INT64,
        source STRING,
        code STRING,
        created_at INT64,
        PRIMARY KEY(name)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Constraint(
        id STRING,
        constraint_text STRING,
        constraint_type STRING,
        discovered_at INT64,
        PRIMARY KEY(id)
    )""",

    # --- Relationship tables ---
    """CREATE REL TABLE IF NOT EXISTS DECOMPOSED_INTO(
        FROM Instruction TO Step,
        step_order INT64
    )""",
    """CREATE REL TABLE IF NOT EXISTS USED_TOOL(
        FROM Step TO Tool,
        outcome STRING,
        latency_ms INT64
    )""",
    """CREATE REL TABLE IF NOT EXISTS DISCOVERED(
        FROM Step TO Constraint
    )""",
    """CREATE REL TABLE IF NOT EXISTS SIMILAR_TO(
        FROM Instruction TO Instruction,
        similarity DOUBLE
    )""",
    # capability synthesis lineage: which instruction triggered which tool to be built
    """CREATE REL TABLE IF NOT EXISTS TRIGGERED_SYNTHESIS(
        FROM Instruction TO Tool
    )""",
]


def init_schema(db_path: str) -> tuple[kuzu.Database, kuzu.Connection]:
    """Create (or open) the memory DB and ensure schema exists. Idempotent.
    Returns both the Database and Connection - a prior version returned only
    the Connection, which meant nothing in this codebase could ever close the
    Database object itself. On Windows, closing only the Connection did not
    reliably release the database's file lock (confirmed: the fix held on
    Linux, the exact same test still failed on Windows with a real run).
    Closing both, in dependent order (Connection first, then Database), is
    the complete, correct cleanup - not a guess this time."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)
    for stmt in SCHEMA_STATEMENTS:
        conn.execute(stmt)
    return db, conn
