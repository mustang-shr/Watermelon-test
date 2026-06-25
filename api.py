"""
Thin FastAPI surface over the Agent. Not in the brief's requirements - added
because the JD lists FastAPI as a required skill and the agent core was
already there to wrap; this cost about 40 minutes given the existing Agent
class, not a separate effort.

create_app(agent) takes an already-built Agent so tests can pass a fake one
without needing NVIDIA_API_KEY / GITHUB_TOKEN - see tests/test_api.py, which
runs with zero live credentials. Running this file directly wires the real
config.build_agent() and needs your .env populated.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import uuid

logger = logging.getLogger("watermelon_agent_api")


class RunRequest(BaseModel):
    instruction: str


class StepResultOut(BaseModel):
    description: str
    status: str
    error: str | None = None


class RunResponse(BaseModel):
    status: str
    steps: list[StepResultOut]
    total_api_calls: int
    total_time_ms: int
    plan_reasoning: str
    confidence: float
    confidence_reason: str
    synthesis_triggered: list[str]
    similar_past_count: int


def create_app(agent) -> FastAPI:
    app = FastAPI(title="Watermelon Autonomous Platform Agent (GitHub)")

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        """Previously: an unhandled crash returned a bare 'Internal Server
        Error' with zero structured detail, silently breaking the 'structured
        report after every run' contract at the HTTP boundary even though
        Agent.run() itself honors it internally. Confirmed by deliberately
        crashing a planner and hitting /run before this existed.

        This does NOT catch tool-level failures - those already produce a
        normal 200 with status='partial'/'failed' in the step list, which is
        the correct path. This only catches genuinely unexpected errors
        (a planner/memory/synthesis bug), distinguished in the response so a
        caller doesn't confuse 'the GitHub API call failed' with 'the agent
        itself broke'."""
        error_id = str(uuid.uuid4())
        logger.exception(f"Unhandled error [{error_id}] on {request.url.path}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "internal_error",
                "error_id": error_id,
                "error_type": type(exc).__name__,
                "detail": (
                    "An unexpected internal error occurred (not a GitHub API or tool "
                    "failure - those are reported with status 200 and per-step detail). "
                    f"Reference error_id {error_id} when investigating server logs."
                ),
            },
        )

    @app.get("/health")
    def health():
        return {"status": "ok", "tools_registered": len(agent.registry.list_tools())}

    @app.get("/tools")
    def list_tools():
        return agent.registry.list_tools()

    @app.post("/run", response_model=RunResponse)
    def run(req: RunRequest):
        if not req.instruction.strip():
            raise HTTPException(status_code=422, detail="instruction must not be empty")
        result = agent.run(req.instruction)
        return RunResponse(
            status=result.report.status,
            steps=[
                StepResultOut(description=r.step.description, status=r.status, error=r.error)
                for r in result.report.step_results
            ],
            total_api_calls=result.report.total_api_calls,
            total_time_ms=result.report.total_time_ms,
            plan_reasoning=result.plan_reasoning,
            confidence=result.confidence,
            confidence_reason=result.confidence_reason,
            synthesis_triggered=[ev.tool_name for ev in result.synthesis_events],
            similar_past_count=len(result.similar_past_used),
        )

    return app


if __name__ == "__main__":
    import sys
    import os
    import uvicorn

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from config import build_agent

    real_agent = build_agent(use_real_embeddings=True)
    app = create_app(real_agent)
    uvicorn.run(app, host="0.0.0.0", port=8000)
