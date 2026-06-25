"""
Small helpers injected into synthesized tools that need date arithmetic.
Generated code still cannot `import datetime` itself - the AST guard in
synthesis.py blocks all imports, on purpose. This is how a second capability
that genuinely needs date math gets built without weakening that guard:
the parsing logic lives here, pre-built and reviewed, and only the parsed
numeric result is handed to generated code via injected_namespace.
"""

from datetime import datetime


def parse_iso_to_epoch_seconds(iso_string: str) -> float:
    """'2026-06-01T00:00:00Z' -> epoch seconds. GitHub timestamps are always
    UTC with a trailing 'Z', which Python's fromisoformat doesn't accept
    directly before 3.11 - normalize it explicitly rather than assume."""
    return datetime.fromisoformat(iso_string.replace("Z", "+00:00")).timestamp()
