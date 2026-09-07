import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from app.observability import context
from app.observability.logging import get_logger, redact
from app.observability.metrics import RunMetrics

log = get_logger("app.graph.node")

NodeFn = Callable[[dict[str, Any]], dict[str, Any]]


def _input_digest(state: dict[str, Any]) -> dict[str, Any]:
    """Log shape, not payload. A full state dump per node would bury the signal and
    drag whole SERP response bodies into the log stream."""
    call = state.get("call")
    if call is not None:
        # Fan-out branches carry a single planned call rather than the whole state.
        return {
            "call_id": getattr(call, "call_id", None),
            "tool": getattr(call, "tool", None),
            "query_text": redact(getattr(call, "query_text", None)),
            "invocations": len(state.get("invocations") or []),
        }
    return {
        "planned_calls": len(state.get("planned_calls") or []),
        "invocations": len(state.get("invocations") or []),
        "records": len(state.get("records") or []),
        "insights": len(state.get("insights") or []),
        "degraded": bool(state.get("degraded")),
        "question": redact(state.get("question")),
    }


def instrumented(
    name: str, metrics: RunMetrics, sink: Callable[..., None] | None = None
) -> Callable[[NodeFn], NodeFn]:
    def decorate(fn: NodeFn) -> NodeFn:
        @wraps(fn)
        def wrapper(state: dict[str, Any]) -> dict[str, Any]:
            with context.bind(node=name):
                started = time.perf_counter()
                log.info("node started", extra={"input": _input_digest(state)})
                try:
                    result = fn(state) or {}
                except Exception as exc:
                    elapsed = (time.perf_counter() - started) * 1000
                    metrics.record_node(name, duration_ms=elapsed, ok=False)
                    log.exception(
                        "node raised",
                        extra={"duration_ms": round(elapsed, 2), "error": str(exc)},
                    )
                    if sink:
                        sink(node=name, status="error", duration_ms=elapsed, error=str(exc))
                    raise

                elapsed = (time.perf_counter() - started) * 1000
                retries = sum(
                    max(inv.attempts - 1, 0) for inv in result.get("invocations", []) or []
                )
                metrics.record_node(name, duration_ms=elapsed, ok=True, retries=retries)
                log.info(
                    "node finished",
                    extra={
                        "duration_ms": round(elapsed, 2),
                        "retries": retries,
                        "output": _input_digest({**state, **result}),
                    },
                )
                if sink:
                    sink(node=name, status="ok", duration_ms=elapsed, retries=retries)
                result.setdefault("node_path", [])
                result["node_path"] = [*result["node_path"], name]
                return result

        return wrapper

    return decorate
