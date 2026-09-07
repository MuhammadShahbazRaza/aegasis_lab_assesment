import time
from dataclasses import dataclass, field
from typing import Any

from app.clients.dataforseo import DataForSEOClient
from app.config import Settings
from app.observability.logging import get_logger, safe_extra
from app.observability.metrics import RunMetrics
from app.resilience.circuit_breaker import CircuitBreakerRegistry
from app.resilience.errors import ToolExecutionError, ToolValidationError
from app.resilience.retry import RetryPolicy, call_with_retry
from app.tools.registry import ValidatedToolCall, validate_tool_call

log = get_logger(__name__)


@dataclass
class ToolInvocation:
    tool: str
    endpoint: str
    args: dict[str, Any] = field(default_factory=dict)
    ok: bool = False
    raw: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    attempts: int = 0
    duration_ms: float = 0.0
    cost_usd: float = 0.0
    call_id: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "endpoint": self.endpoint,
            "ok": self.ok,
            "attempts": self.attempts,
            "duration_ms": round(self.duration_ms, 2),
            "cost_usd": self.cost_usd,
            **({"error": self.error} if self.error else {}),
        }


def _request_body(call: ValidatedToolCall) -> list[dict[str, Any]]:
    return [call.spec.to_payload(call.payload)]


class ToolExecutor:
    """Everything between a proposed tool call and a response: schema validation,
    circuit breaker, retry with backoff, timing and metrics. Nodes call this and
    never touch the HTTP client directly."""

    def __init__(
        self,
        client: DataForSEOClient,
        settings: Settings,
        metrics: RunMetrics,
        breakers: CircuitBreakerRegistry | None = None,
    ):
        self._client = client
        self._metrics = metrics
        self._policy = RetryPolicy(
            max_attempts=settings.retry_max_attempts,
            base_delay=settings.retry_base_delay,
            max_delay=settings.retry_max_delay,
        )
        self._breakers = breakers or CircuitBreakerRegistry(
            settings.circuit_failure_threshold, settings.circuit_reset_timeout
        )

    @property
    def breakers(self) -> CircuitBreakerRegistry:
        return self._breakers

    def invoke(
        self, name: str, raw_args: dict[str, Any] | None, *, call_id: str | None = None
    ) -> ToolInvocation:
        started = time.perf_counter()
        try:
            call = validate_tool_call(name, raw_args)
        except ToolValidationError as exc:
            log.warning("rejected tool call", extra={"tool": name, "rejection": exc.as_dict()})
            self._metrics.record_api_call(name, ok=False)
            return ToolInvocation(
                tool=name,
                endpoint=exc.context.endpoint or "unknown",
                args=raw_args or {},
                error=exc.as_dict(),
                duration_ms=(time.perf_counter() - started) * 1000,
                call_id=call_id,
            )

        endpoint = call.spec.endpoint
        invocation = ToolInvocation(
            tool=name, endpoint=endpoint, args=call.payload, call_id=call_id
        )

        def _do() -> dict[str, Any]:
            self._breakers.before_call(endpoint)
            try:
                result = self._client.execute(endpoint, _request_body(call), tool=name)
            except ToolExecutionError as exc:
                if exc.retryable:
                    self._breakers.record_failure(endpoint)
                raise
            self._breakers.record_success(endpoint)
            return result

        try:
            raw, outcome = call_with_retry(_do, self._policy, label=f"{name}:{endpoint}")
            invocation.ok = True
            invocation.raw = raw
            invocation.attempts = outcome.attempts
        except ToolExecutionError as exc:
            invocation.error = exc.as_dict()
            invocation.attempts = exc.context.attempts or self._policy.max_attempts
        finally:
            invocation.duration_ms = (time.perf_counter() - started) * 1000

        # DataForSEO reports the billed amount in every envelope; taking it from the
        # response rather than from the static table keeps the run total honest when
        # their pricing changes.
        billed = float((invocation.raw or {}).get("cost") or 0.0)
        invocation.cost_usd = billed
        self._metrics.record_api_call(name, ok=invocation.ok, cost_usd=billed)
        log.info(
            "tool call finished" if invocation.ok else "tool call failed",
            extra=safe_extra(invocation.summary()),
        )
        return invocation
