"""
Wires the real components together from environment variables. This is what
scripts/run_demo.py imports - everything in here needs a live NVIDIA_API_KEY
and GITHUB_TOKEN, neither of which this sandbox can reach, so this file is
unverified end-to-end. Every individual piece it assembles (MemoryStore,
ToolRegistry, github_tools, CapabilitySynthesisEngine harness) IS verified -
see the test commands referenced in ARCHITECTURE.md. What's NOT verified is
the live NIM call shape and the live GitHub write calls.
"""

import os
import sys

from dotenv import load_dotenv
load_dotenv()  # reads .env in the current working directory, if present

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from memory.store import MemoryStore
from memory.embeddings import LocalTFIDFEmbedder, NvidiaNIMEmbedder, FallbackEmbedder
from tools.registry import ToolRegistry
from tools.github_tools import GitHubClient, register_github_tools
from tools.synthesis import CapabilitySynthesisEngine, NvidiaNIMCodeGen
from agent.planner import FallbackPlanner, NvidiaNIMPlanner, RuleBasedPlanner
from agent.agent import Agent


def load_config() -> dict:
    required = ["NVIDIA_API_KEY", "GITHUB_TOKEN", "GITHUB_REPO"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {missing}. "
            f"Copy .env.example to .env, fill it in, and load it (e.g. `source .env` "
            f"or `python-dotenv`) before running."
        )
    return {
        "nvidia_api_key": os.environ["NVIDIA_API_KEY"],
        "github_token": os.environ["GITHUB_TOKEN"],
        "github_repo": os.environ["GITHUB_REPO"],
        "nim_model_id": os.environ.get("NIM_MODEL_ID", "meta/llama-3.1-70b-instruct"),
        "nim_embedding_model_id": os.environ.get("NIM_EMBEDDING_MODEL_ID", "nvidia/nv-embedqa-e5-v5"),
    }


def build_agent(use_real_embeddings: bool = True) -> Agent:
    cfg = load_config()

    # FallbackEmbedder: tries NIM semantic embeddings first; if that 404s
    # (wrong model ID, misconfigured, down) it silently degrades to local
    # TF-IDF rather than crashing the whole run. Confirmed necessary from a
    # real failure: NIM embedding was 404-ing and crashed an otherwise-successful
    # issue-creation run before this existed.
    nim_embedder = NvidiaNIMEmbedder(api_key=cfg["nvidia_api_key"], model=cfg["nim_embedding_model_id"])
    embedder = FallbackEmbedder(primary=nim_embedder, fallback=LocalTFIDFEmbedder()) \
        if use_real_embeddings else LocalTFIDFEmbedder()

    memory = MemoryStore(db_path="./data/memory_db", embedder=embedder)
    registry = ToolRegistry(memory_store=memory)

    github_client = GitHubClient(token=cfg["github_token"], repo=cfg["github_repo"])
    register_github_tools(registry, github_client)

    # FallbackPlanner: tries NIM first; on any timeout/network/API error,
    # falls back to RuleBasedPlanner which handles the 3 DEMO.md instruction
    # phrasings deterministically. This is why the v1 demo produced a
    # successful run with "86% success rate" in the planner reasoning — the
    # LIVE NIM planner actually worked in that run, not the rule-based one.
    # If NIM is down, the rule-based fallback fires instead - visible via a
    # RuntimeWarning printed to stderr.
    nim_planner = NvidiaNIMPlanner(api_key=cfg["nvidia_api_key"], model=cfg["nim_model_id"])
    planner = FallbackPlanner(primary=nim_planner, fallback=RuleBasedPlanner())

    codegen = NvidiaNIMCodeGen(api_key=cfg["nvidia_api_key"], model=cfg["nim_model_id"])
    synthesis_engine = CapabilitySynthesisEngine(codegen=codegen, max_attempts=3)

    return Agent(
        memory=memory, registry=registry, planner=planner,
        synthesis_engine=synthesis_engine, embedder=embedder,
        repo_default=cfg["github_repo"],
    )
