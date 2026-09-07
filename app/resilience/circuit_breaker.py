import threading
import time
from dataclasses import dataclass, field
from typing import Literal

from app.observability.logging import get_logger
from app.resilience.errors import CircuitOpenError, ErrorContext

log = get_logger(__name__)

State = Literal["closed", "open", "half_open"]


@dataclass
class _Breaker:
    failure_threshold: int
    reset_timeout: float
    failures: int = 0
    state: State = "closed"
    opened_at: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


class CircuitBreakerRegistry:
    """One breaker per dependency key so a dead endpoint cannot starve the others.

    Half-open admits a single probe; success closes the breaker, failure re-opens it
    and restarts the cooldown.
    """

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 30.0):
        self._threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._breakers: dict[str, _Breaker] = {}
        self._registry_lock = threading.Lock()

    def _get(self, key: str) -> _Breaker:
        with self._registry_lock:
            if key not in self._breakers:
                self._breakers[key] = _Breaker(self._threshold, self._reset_timeout)
            return self._breakers[key]

    def before_call(self, key: str) -> None:
        b = self._get(key)
        with b.lock:
            if b.state == "open":
                if time.monotonic() - b.opened_at < b.reset_timeout:
                    raise CircuitOpenError(
                        f"circuit open for {key}",
                        ErrorContext(endpoint=key, detail={"failures": b.failures}),
                        retry_after=b.reset_timeout,
                    )
                b.state = "half_open"
                log.info("circuit half-open, admitting probe", extra={"dependency": key})

    def record_success(self, key: str) -> None:
        b = self._get(key)
        with b.lock:
            if b.state != "closed":
                log.info("circuit closed after successful probe", extra={"dependency": key})
            b.failures = 0
            b.state = "closed"

    def record_failure(self, key: str) -> None:
        b = self._get(key)
        with b.lock:
            b.failures += 1
            if b.state == "half_open" or b.failures >= b.failure_threshold:
                if b.state != "open":
                    log.error(
                        "circuit opened",
                        extra={"dependency": key, "failures": b.failures},
                    )
                b.state = "open"
                b.opened_at = time.monotonic()

    def snapshot(self) -> dict[str, dict[str, object]]:
        return {
            key: {"state": b.state, "failures": b.failures}
            for key, b in self._breakers.items()
        }

    def reset(self) -> None:
        with self._registry_lock:
            self._breakers.clear()
