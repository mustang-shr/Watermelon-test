"""
Tool registry. Every tool call - built-in or synthesized - goes through
ToolRegistry.call() so that execution outcome always reaches memory. The
agent/executor never calls a tool function directly.

Built-in tools are registered at startup (github_tools.py). Synthesized tools
are added at runtime by the capability-synthesis engine and persist into the
same registry for the rest of the process - and into MemoryStore.Tool.code so
a future process restart can reload them without re-synthesizing.
"""

from dataclasses import dataclass
from typing import Callable, Any
import time


@dataclass
class ToolResult:
    success: bool
    output: Any = None
    error: str | None = None
    latency_ms: int = 0


@dataclass
class RegisteredTool:
    name: str
    fn: Callable[..., Any]
    kind: str            # "builtin" | "synthesized"
    description: str
    source: str = ""      # e.g. "github_api" or "synthesized:run-id"


class ToolRegistry:
    def __init__(self, memory_store=None):
        self._tools: dict[str, RegisteredTool] = {}
        self.memory = memory_store  # optional - injected by agent core

    def register(self, name: str, fn: Callable, kind: str, description: str,
                 source: str = "", code: str = "") -> None:
        self._tools[name] = RegisteredTool(name=name, fn=fn, kind=kind, description=description, source=source)
        if self.memory:
            self.memory.register_tool(name, kind, source, code)

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> list[dict]:
        return [
            {"name": t.name, "kind": t.kind, "description": t.description}
            for t in self._tools.values()
        ]

    def call(self, name: str, **kwargs) -> ToolResult:
        if name not in self._tools:
            return ToolResult(success=False, error=f"Tool '{name}' is not registered. "
                               f"Available: {list(self._tools.keys())}")
        tool = self._tools[name]
        start = time.time()
        try:
            output = tool.fn(**kwargs)
            latency_ms = int((time.time() - start) * 1000)
            return ToolResult(success=True, output=output, latency_ms=latency_ms)
        except Exception as e:
            latency_ms = int((time.time() - start) * 1000)
            return ToolResult(success=False, error=f"{type(e).__name__}: {e}", latency_ms=latency_ms)
