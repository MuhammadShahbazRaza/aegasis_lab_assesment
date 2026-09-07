from typing import Any

from app.domain import PlannedCall
from app.graph.runtime import GraphRuntime
from app.observability.logging import get_logger

log = get_logger(__name__)


def build_retrieval_worker(runtime: GraphRuntime):
    """One planned call, one node execution. The worker fetches and nothing else - it
    does not parse the payload, score it, or decide what happens next."""

    def retrieval_worker(state: dict[str, Any]) -> dict[str, Any]:
        call: PlannedCall = state["call"]
        invocation = runtime.executor.invoke(call.tool, call.args, call_id=call.call_id)
        errors = []
        if not invocation.ok:
            # The provider error goes in first so the call's own identity always
            # wins; a circuit-breaker error carries no tool name of its own.
            errors.append(
                {
                    **(invocation.error or {}),
                    "call_id": call.call_id,
                    "tool": call.tool,
                    "query_text": call.query_text,
                }
            )
        return {"invocations": [invocation], "retrieval_errors": errors}

    return retrieval_worker


def build_retrieval_gate(runtime: GraphRuntime):
    """Join point for the fan-out. Its only job is to judge whether enough of the
    plan came back to be worth analysing; the routing decision itself lives on the
    conditional edge."""

    def retrieval_gate(state: dict[str, Any]) -> dict[str, Any]:
        invocations = state.get("invocations", [])
        planned = state.get("planned_calls", [])
        succeeded = [i for i in invocations if i.ok]
        ratio = len(succeeded) / len(planned) if planned else 0.0
        threshold = runtime.settings.retrieval_success_threshold

        log.info(
            "retrieval complete",
            extra={
                "planned": len(planned),
                "succeeded": len(succeeded),
                "failed": len(invocations) - len(succeeded),
                "success_ratio": round(ratio, 3),
                "threshold": threshold,
                "breakers": runtime.executor.breakers.snapshot(),
            },
        )
        return {
            "flags": {
                "retrieval_success_ratio": round(ratio, 4),
                "retrieval_succeeded": len(succeeded),
                "retrieval_failed": len(invocations) - len(succeeded),
            },
            "degraded": bool(state.get("degraded")) or ratio < 1.0,
        }

    return retrieval_gate
