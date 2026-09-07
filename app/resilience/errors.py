from dataclasses import dataclass, field
from typing import Any


class PipelineError(Exception):
    pass


@dataclass
class ErrorContext:
    tool: str | None = None
    endpoint: str | None = None
    status_code: int | None = None
    provider_code: int | None = None
    attempts: int = 0
    detail: dict[str, Any] = field(default_factory=dict)


class ToolExecutionError(PipelineError):
    retryable = False

    def __init__(self, message: str, context: ErrorContext | None = None):
        super().__init__(message)
        self.message = message
        self.context = context or ErrorContext()

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "message": self.message,
            "retryable": self.retryable,
            "tool": self.context.tool,
            "endpoint": self.context.endpoint,
            "status_code": self.context.status_code,
            "provider_code": self.context.provider_code,
            "attempts": self.context.attempts,
            **({"detail": self.context.detail} if self.context.detail else {}),
        }


class TransientToolError(ToolExecutionError):
    """Worth retrying: timeouts, connection resets, 429, 5xx, provider 5xxxx codes."""

    retryable = True

    def __init__(
        self,
        message: str,
        context: ErrorContext | None = None,
        retry_after: float | None = None,
    ):
        super().__init__(message, context)
        self.retry_after = retry_after


class PermanentToolError(ToolExecutionError):
    """Retrying cannot help: malformed request, auth failure, unknown endpoint."""

    retryable = False


class ToolValidationError(PermanentToolError):
    """The LLM produced tool arguments that do not satisfy the tool's schema."""


class CircuitOpenError(TransientToolError):
    """The breaker rejected the call before it reached the network."""


# DataForSEO returns HTTP 200 with a status_code in the body; the numeric ranges
# below come from their published status code table.
def classify_provider_code(code: int) -> type[ToolExecutionError] | None:
    if 20000 <= code < 30000:
        return None
    if code in (40100, 40101, 40200, 40201, 40202):  # auth / payment
        return PermanentToolError
    if code == 40429:
        return TransientToolError
    if 40000 <= code < 50000:
        return PermanentToolError
    return TransientToolError


def classify_http_status(status: int) -> type[ToolExecutionError] | None:
    if 200 <= status < 300:
        return None
    if status in (408, 425, 429):
        return TransientToolError
    if status >= 500:
        return TransientToolError
    return PermanentToolError
