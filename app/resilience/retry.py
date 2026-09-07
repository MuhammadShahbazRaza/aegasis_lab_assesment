import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from app.observability.logging import get_logger
from app.resilience.errors import CircuitOpenError, ToolExecutionError, TransientToolError

T = TypeVar("T")
log = get_logger(__name__)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay: float = 0.4
    max_delay: float = 8.0
    jitter: bool = True

    def delay_for(self, attempt: int, retry_after: float | None = None) -> float:
        if retry_after is not None:
            return min(retry_after, self.max_delay)
        backoff = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
        # Full jitter. Equal jitter keeps a floor under the delay, but full jitter
        # de-correlates concurrent retrievers better, and here every planned call
        # hits the same upstream at the same instant.
        return random.uniform(0, backoff) if self.jitter else backoff


@dataclass
class RetryOutcome:
    attempts: int = 0
    total_delay: float = 0.0
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


def call_with_retry(
    fn: Callable[[], T],
    policy: RetryPolicy,
    *,
    label: str,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[T, RetryOutcome]:
    outcome = RetryOutcome()
    last: ToolExecutionError | None = None

    for attempt in range(1, policy.max_attempts + 1):
        outcome.attempts = attempt
        try:
            return fn(), outcome
        except CircuitOpenError as exc:
            # The breaker has already concluded this dependency is down. Sleeping and
            # trying again inside the same call would spend the caller's latency budget
            # to reach the same verdict, so this one short-circuits out of the loop.
            exc.context.attempts = attempt
            log.warning(
                "aborting, circuit is open",
                extra={"label": label, "attempt": attempt, "error": exc.message},
            )
            raise
        except TransientToolError as exc:
            last = exc
            outcome.errors.append(exc.message)
            if attempt == policy.max_attempts:
                break
            delay = policy.delay_for(attempt, exc.retry_after)
            outcome.total_delay += delay
            log.warning(
                "retrying after transient failure",
                extra={
                    "label": label,
                    "attempt": attempt,
                    "max_attempts": policy.max_attempts,
                    "sleep_seconds": round(delay, 3),
                    "error": exc.message,
                },
            )
            sleep(delay)
        except ToolExecutionError as exc:
            exc.context.attempts = attempt
            log.warning(
                "aborting on non-retryable failure",
                extra={"label": label, "attempt": attempt, "error": exc.message},
            )
            raise

    assert last is not None
    last.context.attempts = outcome.attempts
    raise last
