from typing import Any

from app.domain import RunStatus
from app.graph.runtime import GraphRuntime
from app.observability.logging import get_logger

log = get_logger(__name__)


def _failures_by_tool(errors: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for err in errors:
        key = str(err.get("tool") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_degraded_fallback(runtime: GraphRuntime):
    """Partial-data path: some retrievals came back, but not enough to call the run
    clean. It marks the run degraded and hands what did arrive to extraction - throwing
    away a successful call because its siblings failed would be worse than useless,
    since the customer already paid for it."""

    def degraded_fallback(state: dict[str, Any]) -> dict[str, Any]:
        errors = state.get("retrieval_errors", [])
        by_tool = _failures_by_tool(errors)
        log.warning(
            "retrieval below threshold, continuing with partial data",
            extra={
                "failed_calls": len(errors),
                "failures_by_tool": by_tool,
                "breakers": runtime.executor.breakers.snapshot(),
            },
        )
        return {
            "degraded": True,
            "flags": {
                "degraded_reason": "retrieval_success_below_threshold",
                "failures_by_tool": by_tool,
            },
        }

    return degraded_fallback


def build_no_data_fallback(runtime: GraphRuntime):
    """Terminal salvage path: nothing usable came back at all. It does not retry -
    the executor already exhausted its budget - it records why the run is empty so the
    Report agent still emits a well-formed, honestly-labelled response instead of the
    caller getting a 500."""

    def no_data_fallback(state: dict[str, Any]) -> dict[str, Any]:
        errors = state.get("retrieval_errors", [])
        by_tool = _failures_by_tool(errors)
        breakers = runtime.executor.breakers.snapshot()
        open_circuits = [dep for dep, s in breakers.items() if s["state"] != "closed"]

        log.error(
            "no usable data retrieved, failing run open",
            extra={
                "failed_calls": len(errors),
                "failures_by_tool": by_tool,
                "open_circuits": open_circuits,
            },
        )
        return {
            "degraded": True,
            "status": RunStatus.FAILED,
            "flags": {
                "degraded_reason": "no_usable_records",
                "failures_by_tool": by_tool,
                "open_circuits": open_circuits,
            },
        }

    return no_data_fallback
