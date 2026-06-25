"""
Capability synthesis: turns "the agent needs to do X and has no tool for it"
into a tested, registered tool - at runtime, not design time.

This is deliberately NOT a lookup table of API endpoints (the cheap version
the brief explicitly calls out). The contract is:

  1. Describe the gap as a function signature + docstring + test cases
  2. Ask a CodeGenClient to implement it
  3. Execute the generated code in a restricted namespace (no arbitrary
     imports, no filesystem/network access from inside generated code -
     anything it needs, like an embedding function, is injected explicitly)
  4. Run the test cases against it for real
  5. If a test fails, feed the actual error back to the codegen client and
     retry (bounded by max_attempts) - this is what makes it synthesis
     rather than one-shot code generation
  6. Only persist/register the tool if it actually passes its tests

CodeGenClient is swappable: NvidiaNIMCodeGen for the real demo (untestable in
this sandbox - build.nvidia.com isn't reachable here), MockCodeGen for
proving the harness itself works without needing a live model.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import ast


@dataclass
class SynthesisResult:
    success: bool
    tool_name: str
    code: str = ""
    last_code: str = ""     # populated on failure — shows exactly what NIM wrote
    fn: object = None
    attempts: int = 0
    last_error: str = ""
    report: str = ""


class CodeGenClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return raw code text (may include markdown fences - caller strips them)."""
        raise NotImplementedError


class MockCodeGen(CodeGenClient):
    """Deterministic stand-in for proving the synthesis harness (exec, test,
    retry-on-failure) actually works, without a live LLM. Takes a fixed
    sequence of responses and returns them in order - lets a test simulate
    'first attempt is broken, second attempt (after seeing the error) fixes it',
    which is the real behaviour the harness needs to handle correctly."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        resp = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return resp


class NvidiaNIMCodeGen(CodeGenClient):
    """Production codegen via NIM chat completions. Endpoint shape (OpenAI-
    compatible /chat/completions on integrate.api.nvidia.com) confirmed by
    NVIDIA docs. Default model: meta/llama-3.1-70b-instruct — verified active
    on integrate.api.nvidia.com as of June 2026."""

    def __init__(self, api_key: str, model: str = "meta/llama-3.1-70b-instruct",
                 base_url: str = "https://integrate.api.nvidia.com/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str) -> str:
        import requests
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                },
                timeout=120,
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"NIM request timed out after 120s (model='{self.model}'). Run "
                f"`python scripts/check_nim_connection.py` to isolate whether this is the model ID, "
                f"a slow/cold model, or a network issue."
            ) from None
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Could not connect to NIM at all: {e}. Run `python scripts/check_nim_connection.py`."
            ) from None
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def _strip_code_fences(text: str) -> str:
    if "```" not in text:
        return text.strip()
    parts = text.split("```")
    # take the largest fenced block - the code, not surrounding prose
    blocks = [p for i, p in enumerate(parts) if i % 2 == 1]
    code = max(blocks, key=len) if blocks else text
    lines = code.split("\n")
    if lines and lines[0].strip().lower() in ("python", "py"):
        lines = lines[1:]
    return "\n".join(lines).strip()


def exec_and_extract(code: str, fn_name: str, injected_namespace: dict | None = None):
    """Load a function from source text under the same safety constraints as
    fresh synthesis: no imports, restricted builtins, must define `fn_name`.
    Used both by CapabilitySynthesisEngine (new code) and by the agent's
    startup reload path (previously-synthesized code pulled back from
    memory) - reload does NOT get a weaker check just because the code
    passed its tests once before in a different process."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise PermissionError(
                f"Code attempted an import ({ast.dump(node)[:60]}...). "
                f"Synthesized tools may only use what's in injected_namespace."
            )

    safe_builtins = {
        "len": len, "range": range, "enumerate": enumerate, "sorted": sorted,
        "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
        "list": list, "dict": dict, "tuple": tuple, "set": set, "str": str,
        "int": int, "float": float, "bool": bool, "zip": zip, "map": map,
        "filter": filter, "isinstance": isinstance, "Exception": Exception,
        "ValueError": ValueError, "TypeError": TypeError,
    }
    namespace = {"__builtins__": safe_builtins, **(injected_namespace or {})}
    exec(code, namespace)

    if fn_name not in namespace or not callable(namespace[fn_name]):
        raise NameError(f"Code did not define a callable named '{fn_name}'")
    return namespace[fn_name]


class CapabilitySynthesisEngine:
    def __init__(self, codegen: CodeGenClient, max_attempts: int = 3):
        self.codegen = codegen
        self.max_attempts = max_attempts

    def synthesize(self, tool_name: str, contract_prompt: str,
                    test_cases: list[dict], injected_namespace: dict | None = None) -> SynthesisResult:
        """
        contract_prompt: full description of what the function must do, its
            exact required name/signature, and what's available in scope.
        test_cases: list of {"kwargs": {...}, "check": callable(result) -> bool}
        injected_namespace: pre-built objects the generated code may call
            (e.g. an `embed_fn`) WITHOUT importing anything itself.
        """
        prompt = contract_prompt
        last_error = ""
        last_code = ""

        for attempt in range(1, self.max_attempts + 1):
            raw = self.codegen.generate(prompt)
            code = _strip_code_fences(raw)
            last_code = code

            try:
                fn = self._exec_and_extract(code, tool_name, injected_namespace or {})
            except Exception as e:
                last_error = f"[attempt {attempt}] code did not load: {type(e).__name__}: {e}"
                prompt = self._repair_prompt(contract_prompt, code, last_error)
                continue

            failure = self._run_tests(fn, test_cases)
            if failure is None:
                return SynthesisResult(success=True, tool_name=tool_name, code=code, fn=fn,
                                        attempts=attempt,
                                        report=f"Synthesis succeeded on attempt {attempt}/{self.max_attempts}")
            last_error = f"[attempt {attempt}] test failed: {failure}"
            prompt = self._repair_prompt(contract_prompt, code, last_error)

        # Include the last generated code in the failure report so the caller can
        # see exactly what NIM wrote — this is the single most useful debugging
        # signal when synthesis fails. The code is trimmed to 800 chars to avoid
        # flooding logs, but enough to see the loop structure and any obvious bugs.
        code_preview = last_code[:800] + ("..." if len(last_code) > 800 else "")
        return SynthesisResult(success=False, tool_name=tool_name, attempts=self.max_attempts,
                                last_error=last_error,
                                last_code=code_preview,
                                report=f"Synthesis failed after {self.max_attempts} attempts. "
                                       f"Last error: {last_error}\n"
                                       f"Last generated code:\n{code_preview}")

    def _exec_and_extract(self, code: str, fn_name: str, injected_namespace: dict):
        return exec_and_extract(code, fn_name, injected_namespace)

    def _run_tests(self, fn, test_cases: list[dict]) -> str | None:
        """Returns None if all tests pass, else a description of the first failure."""
        for i, case in enumerate(test_cases):
            try:
                result = fn(**case["kwargs"])
            except Exception as e:
                return f"test case {i} raised {type(e).__name__}: {e}"
            if not case["check"](result):
                return f"test case {i} returned {result!r}, which failed its check"
        return None

    def _repair_prompt(self, original_prompt: str, broken_code: str, error: str) -> str:
        return (
            f"{original_prompt}\n\n"
            f"--- Your previous attempt failed. Fix it. ---\n"
            f"Previous code:\n{broken_code}\n\n"
            f"Error:\n{error}\n\n"
            f"Return ONLY the corrected function, same name and contract."
        )
