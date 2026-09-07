import json
import logging
import sys
from typing import Any

from app.observability.context import current_context

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}

_SENSITIVE_KEYS = {
    "password",
    "api_key",
    "apikey",
    "authorization",
    "token",
    "secret",
    "groq_api_key",
    "openai_api_key",
    "anthropic_api_key",
    "dataforseo_password",
}

_MAX_STRING = 400
_MAX_ITEMS = 25


def safe_extra(payload: dict[str, Any]) -> dict[str, Any]:
    """logging.makeRecord raises if `extra` shadows a LogRecord attribute, and
    `message` is both a reserved name and an obvious key for an error dict. Anything
    that would collide gets prefixed instead of taking the process down."""
    return {(f"ctx_{k}" if k in _RESERVED else k): v for k, v in payload.items()}


def redact(value: Any, _depth: int = 0) -> Any:
    """Strip credentials and cap payload size so node inputs are loggable verbatim."""
    if _depth > 6:
        return "<max-depth>"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if str(k).lower() in _SENSITIVE_KEYS:
                out[k] = "***"
            else:
                out[k] = redact(v, _depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        items = [redact(v, _depth + 1) for v in list(value)[:_MAX_ITEMS]]
        if len(value) > _MAX_ITEMS:
            items.append(f"<+{len(value) - _MAX_ITEMS} more>")
        return items
    if isinstance(value, str) and len(value) > _MAX_STRING:
        return value[:_MAX_STRING] + f"<+{len(value) - _MAX_STRING} chars>"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:_MAX_STRING]


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            **current_context(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = redact(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ctx = current_context()
        prefix = " ".join(f"{k}={v}" for k, v in ctx.items())
        extras = {
            k: redact(v)
            for k, v in record.__dict__.items()
            if k not in _RESERVED and not k.startswith("_")
        }
        tail = " ".join(f"{k}={v}" for k, v in extras.items())
        parts = [f"{record.levelname:<7}", record.getMessage()]
        if prefix:
            parts.append(f"[{prefix}]")
        if tail:
            parts.append(tail)
        return " ".join(parts)


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else ConsoleFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
