import threading
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import median
from typing import Any


@dataclass
class NodeStat:
    calls: int = 0
    successes: int = 0
    failures: int = 0
    retries: int = 0
    durations_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        d = sorted(self.durations_ms)
        return {
            "calls": self.calls,
            "successes": self.successes,
            "failures": self.failures,
            "retries": self.retries,
            "p50_ms": round(median(d), 2) if d else 0.0,
            "max_ms": round(d[-1], 2) if d else 0.0,
            "total_ms": round(sum(d), 2),
        }


class RunMetrics:
    """Per-run counters. Scoped to one DAG execution, so it is a plain in-process
    object rather than a global registry; a production build would push the same
    fields to a Prometheus/OTel collector instead."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self._nodes: dict[str, NodeStat] = defaultdict(NodeStat)
        self._api_calls: dict[str, int] = defaultdict(int)
        self._api_failures: dict[str, int] = defaultdict(int)
        self._tokens: dict[str, int] = defaultdict(int)
        self._cost_usd: float = 0.0
        self._lock = threading.Lock()

    def record_node(
        self, node: str, *, duration_ms: float, ok: bool, retries: int = 0
    ) -> None:
        with self._lock:
            stat = self._nodes[node]
            stat.calls += 1
            stat.retries += retries
            stat.durations_ms.append(duration_ms)
            if ok:
                stat.successes += 1
            else:
                stat.failures += 1

    def record_api_call(self, tool: str, *, ok: bool, cost_usd: float = 0.0) -> None:
        with self._lock:
            self._api_calls[tool] += 1
            self._cost_usd += cost_usd
            if not ok:
                self._api_failures[tool] += 1

    def record_tokens(self, usage: dict[str, int] | None) -> None:
        if not usage:
            return
        with self._lock:
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                if usage.get(key):
                    self._tokens[key] += int(usage[key])

    @property
    def token_usage(self) -> dict[str, int]:
        with self._lock:
            return dict(self._tokens)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            total_api = sum(self._api_calls.values())
            failed_api = sum(self._api_failures.values())
            node_calls = sum(s.calls for s in self._nodes.values())
            node_fail = sum(s.failures for s in self._nodes.values())
            return {
                "run_id": self.run_id,
                "nodes": {name: stat.summary() for name, stat in self._nodes.items()},
                "node_success_rate": round(
                    (node_calls - node_fail) / node_calls, 4
                ) if node_calls else None,
                "api_calls": dict(self._api_calls),
                "api_failures": dict(self._api_failures),
                "api_call_total": total_api,
                "api_success_rate": round((total_api - failed_api) / total_api, 4)
                if total_api
                else None,
                "tokens": dict(self._tokens),
                "dataforseo_cost_usd": round(self._cost_usd, 4),
            }
