import json
import logging

import pytest

from app.observability import context
from app.observability.logging import JsonFormatter, redact, safe_extra
from app.observability.metrics import RunMetrics


def _format(record_kwargs: dict, **extra) -> dict:
    logger = logging.getLogger("test.obs")
    record = logger.makeRecord(
        "test.obs", logging.INFO, "f.py", 1, record_kwargs["msg"], (), None, extra=extra
    )
    return json.loads(JsonFormatter().format(record))


def test_log_lines_are_single_line_json_with_the_event_name():
    payload = _format({"msg": "node finished"}, duration_ms=12.5)
    assert payload["event"] == "node finished"
    assert payload["level"] == "INFO"
    assert payload["duration_ms"] == 12.5
    assert payload["ts"]


def test_correlation_ids_are_attached_without_being_passed_down():
    with context.bind(run_id="run-abc", node="query_planner", profile_uuid="p1"):
        payload = _format({"msg": "node started"})
    assert payload["run_id"] == "run-abc"
    assert payload["node"] == "query_planner"
    assert payload["profile_uuid"] == "p1"

    # Bindings must not leak past their scope, or a later run inherits them.
    assert "run_id" not in _format({"msg": "after"})


def test_nested_context_restores_the_outer_value():
    with context.bind(node="a"):
        with context.bind(node="b"):
            assert context.current_context()["node"] == "b"
        assert context.current_context()["node"] == "a"


@pytest.mark.parametrize(
    "key", ["password", "api_key", "Authorization", "GROQ_API_KEY", "dataforseo_password"]
)
def test_credentials_are_redacted_at_any_depth(key):
    assert redact({key: "hunter2"})[key] == "***"
    assert redact({"outer": {"inner": {key: "hunter2"}}})["outer"]["inner"][key] == "***"
    assert redact([{key: "hunter2"}])[0][key] == "***"


def test_large_payloads_are_truncated_rather_than_dumped():
    long_text = redact("x" * 5000)
    assert len(long_text) < 500 and long_text.endswith("<+4600 chars>")

    big_list = redact(list(range(200)))
    assert len(big_list) == 26 and "more" in big_list[-1]


def test_recursive_structures_do_not_hang_the_logger():
    deep: dict = {}
    cursor = deep
    for _ in range(20):
        cursor["next"] = {}
        cursor = cursor["next"]
    assert "max-depth" in json.dumps(redact(deep))


def test_reserved_logrecord_keys_are_renamed_instead_of_raising():
    cleaned = safe_extra({"message": "boom", "tool": "serp_organic_results"})
    assert cleaned["ctx_message"] == "boom"
    assert cleaned["tool"] == "serp_organic_results"
    # The whole point: this must not raise.
    _format({"msg": "rejected tool call"}, **cleaned)


def test_metrics_track_latency_success_and_retries():
    metrics = RunMetrics("run-1")
    metrics.record_node("retrieval_worker", duration_ms=10.0, ok=True, retries=1)
    metrics.record_node("retrieval_worker", duration_ms=30.0, ok=False)
    metrics.record_node("report_assembler", duration_ms=5.0, ok=True)

    snapshot = metrics.snapshot()
    worker = snapshot["nodes"]["retrieval_worker"]
    assert worker == {
        "calls": 2, "successes": 1, "failures": 1, "retries": 1,
        "p50_ms": 20.0, "max_ms": 30.0, "total_ms": 40.0,
    }
    assert snapshot["node_success_rate"] == pytest.approx(2 / 3, abs=1e-4)


def test_metrics_track_api_calls_per_tool():
    metrics = RunMetrics("run-1")
    metrics.record_api_call("serp_organic_results", ok=True)
    metrics.record_api_call("serp_organic_results", ok=False)
    metrics.record_api_call("ai_overview_snapshot", ok=True)

    snapshot = metrics.snapshot()
    assert snapshot["api_calls"] == {"serp_organic_results": 2, "ai_overview_snapshot": 1}
    assert snapshot["api_failures"] == {"serp_organic_results": 1}
    assert snapshot["api_call_total"] == 3
    assert snapshot["api_success_rate"] == pytest.approx(2 / 3, abs=1e-4)


def test_token_usage_accumulates_across_agents():
    metrics = RunMetrics("run-1")
    metrics.record_tokens({"input_tokens": 100, "output_tokens": 50, "total_tokens": 150})
    metrics.record_tokens({"input_tokens": 20, "output_tokens": 10, "total_tokens": 30})
    metrics.record_tokens(None)
    assert metrics.token_usage == {
        "input_tokens": 120, "output_tokens": 60, "total_tokens": 180
    }


def test_a_run_emits_a_trace_that_covers_every_executed_node(settings, session, profile, caplog):
    from app.service import PipelineService

    with caplog.at_level(logging.INFO, logger="app.graph.node"):
        run = PipelineService(session, settings).run_profile(profile)

    started = [r for r in caplog.records if r.getMessage() == "node started"]
    finished = [r for r in caplog.records if r.getMessage() == "node finished"]
    assert len(started) == len(finished) == len(run.node_path)
    assert all(hasattr(r, "duration_ms") for r in finished)


def test_metrics_accumulate_the_billed_api_cost():
    metrics = RunMetrics("run-1")
    metrics.record_api_call("serp_organic_results", ok=True, cost_usd=0.002)
    metrics.record_api_call("keyword_search_volume", ok=True, cost_usd=0.09)
    metrics.record_api_call("ai_overview_snapshot", ok=False)
    assert metrics.snapshot()["dataforseo_cost_usd"] == 0.092
