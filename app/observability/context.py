from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any

_run_id: ContextVar[str | None] = ContextVar("run_id", default=None)
_node: ContextVar[str | None] = ContextVar("node", default=None)
_profile: ContextVar[str | None] = ContextVar("profile_uuid", default=None)

_FIELDS: dict[str, ContextVar[str | None]] = {
    "run_id": _run_id,
    "node": _node,
    "profile_uuid": _profile,
}


def current_context() -> dict[str, Any]:
    return {name: var.get() for name, var in _FIELDS.items() if var.get() is not None}


def current_run_id() -> str | None:
    return _run_id.get()


@contextmanager
def bind(**values: str | None) -> Iterator[None]:
    tokens: list[tuple[ContextVar[str | None], Token]] = []
    for name, value in values.items():
        var = _FIELDS.get(name)
        if var is not None:
            tokens.append((var, var.set(value)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)
