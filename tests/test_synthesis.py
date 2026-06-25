from tools.synthesis import CapabilitySynthesisEngine, MockCodeGen
from memory.embeddings import LocalTFIDFEmbedder

ISSUES_FIXTURE = [
    {"number": 1, "title": "Login button broken on mobile"},
    {"number": 2, "title": "Mobile login button does not work"},
    {"number": 3, "title": "Add dark mode support"},
]

BROKEN_CODE = '''
def find_duplicate_issues(issues, similarity_fn, threshold=0.3):
    pairs = []
    for i in range(len(issues)):
        for j in range(i+1, len(issues)):
            sim = similarity_fn(issues[i]["text"], issues[j]["text"])
            if sim >= threshold:
                pairs.append((issues[i]["number"], issues[j]["number"], sim))
    return pairs
'''

FIXED_CODE = '''
def find_duplicate_issues(issues, similarity_fn, threshold=0.3):
    pairs = []
    for i in range(len(issues)):
        for j in range(i+1, len(issues)):
            sim = similarity_fn(issues[i]["title"], issues[j]["title"])
            if sim >= threshold:
                pairs.append((issues[i]["number"], issues[j]["number"], sim))
    return pairs
'''


def _check(result):
    pairs = [(p[0], p[1]) for p in result]
    return (1, 2) in pairs and (1, 3) not in pairs and (2, 3) not in pairs


def test_synthesis_recovers_from_broken_first_attempt():
    embedder = LocalTFIDFEmbedder()
    codegen = MockCodeGen(responses=[BROKEN_CODE, FIXED_CODE])
    engine = CapabilitySynthesisEngine(codegen=codegen, max_attempts=3)

    result = engine.synthesize(
        tool_name="find_duplicate_issues",
        contract_prompt="implement find_duplicate_issues(issues, similarity_fn, threshold=0.3)",
        test_cases=[{"kwargs": {"issues": ISSUES_FIXTURE, "similarity_fn": embedder.similarity},
                     "check": _check}],
    )
    assert result.success
    assert result.attempts == 2  # must actually fail attempt 1, not pass trivially
    assert result.fn is not None


def test_synthesis_gives_up_after_max_attempts_with_persistently_broken_code():
    codegen = MockCodeGen(responses=[BROKEN_CODE, BROKEN_CODE, BROKEN_CODE])
    engine = CapabilitySynthesisEngine(codegen=codegen, max_attempts=3)
    embedder = LocalTFIDFEmbedder()

    result = engine.synthesize(
        tool_name="find_duplicate_issues",
        contract_prompt="...",
        test_cases=[{"kwargs": {"issues": ISSUES_FIXTURE, "similarity_fn": embedder.similarity},
                     "check": _check}],
    )
    assert not result.success
    assert result.attempts == 3
    assert "KeyError" in result.last_error or "test failed" in result.last_error


def test_import_guard_rejects_unscoped_code():
    malicious = "import os\ndef find_duplicate_issues(issues, similarity_fn, threshold=0.3):\n    return []"
    codegen = MockCodeGen(responses=[malicious])
    engine = CapabilitySynthesisEngine(codegen=codegen, max_attempts=1)

    result = engine.synthesize(
        tool_name="find_duplicate_issues",
        contract_prompt="...",
        test_cases=[{"kwargs": {"issues": [], "similarity_fn": lambda a, b: 0}, "check": lambda r: True}],
    )
    assert not result.success
    assert "import" in result.last_error.lower()
