# Watermelon Software Recruitment Assignment  Autonomous Platform Agent (GitHub)

Natural-language instructions executed against GitHub, with persistent graph memory,
runtime capability synthesis, and a measured self-learning signal. Built for the
Watermelon Software "Autonomous Platform Intelligence Agent" assignment.

See **ARCHITECTURE.md** for the three required design answers, **DEMO.md** for the
three live-call instructions.

## What's verified vs what isn't

Everything in `src/memory/`, `src/tools/`, `src/agent/executor.py` + `src/agent/agent.py`'s
orchestration logic, and `api.py` is verified by the test suite  35 tests, zero API
keys needed**, including a real GitHub API call against a public repo, mocked-request
tests for every write operation, two independently-tested capability-synthesis gap
types, and a simulated-restart test proving synthesized capabilities survive a process
restart (this was broken until an internal review caught it see ARCHITECTURE.md).

**Not verified** (this build environment can't reach `build.nvidia.com` or make
authenticated GitHub writes): the live `NvidiaNIMPlanner` and `NvidiaNIMCodeGen` calls
in `src/agent/planner.py` and `src/tools/synthesis.py`. This is the single largest open
risk in the submission see ARCHITECTURE.md section 2. The exact NIM model ID is a
placeholder  confirm it on your `build.nvidia.com` catalog page before running.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: fill in NVIDIA_API_KEY, GITHUB_TOKEN, GITHUB_REPO, NIM_MODEL_ID
```

`.env` is loaded automatically by `config.py` via `python-dotenv` - no manual export
step needed, just have a filled-in `.env` in the project root.

`GITHUB_REPO` should be a disposable test repo - the agent will create real issues,
comments, and labels on it.

## Run the test suite (no API keys required)

```bash
pytest tests/ -v
```

## Run the live demo (requires .env)

```bash
python scripts/run_demo.py                 # runs all three DEMO.md instructions
python scripts/run_demo.py "your own instruction here"
```

## Run the HTTP API (requires .env)

```bash
python api.py        # http://localhost:8000 - POST /run {"instruction": "..."}, GET /tools, GET /health
```
Not required by the brief added because FastAPI is a JD-listed required skill and
the `Agent` class was already there to wrap. Tested with a fake agent, zero live keys
(`tests/test_api.py`).

## Project layout

```
src/
  memory/
    schema.py        Kuzu graph schema (Instruction/Step/Tool/Constraint)
    embeddings.py     Swappable similarity backend (TF-IDF offline / NIM production)
    store.py          Read/write interface the agent actually talks to
  tools/
    registry.py       Single point every tool call routes through
    github_tools.py   Real GitHub REST wrappers
    synthesis.py       Runtime capability synthesis engine
  agent/
    planner.py         NL instruction -> Plan (mock + NIM implementations)
    executor.py         Runs a Plan, honest partial-failure handling
    agent.py            Orchestrates memory -> planner -> synthesis -> executor -> memory;
                         reloads synthesized capabilities on construction
  learning/
    metrics.py          Formats the before/after learning signal with real numbers
api.py                 Thin FastAPI surface over Agent (not required, added for JD fit)
tests/                 35 tests across 8 files, fully offline, zero API keys
config.py              Wires real components from .env for local runs
scripts/run_demo.py    CLI entry point for the three DEMO.md instructions
```
