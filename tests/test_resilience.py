import pytest

from app.clients.dataforseo import MockDataForSEOClient
from app.resilience.circuit_breaker import CircuitBreakerRegistry
from app.resilience.errors import (
    CircuitOpenError,
    ErrorContext,
    PermanentToolError,
    TransientToolError,
    classify_http_status,
    classify_provider_code,
)
from app.resilience.retry import RetryPolicy, call_with_retry
from app.tools.executor import ToolExecutor


@pytest.mark.parametrize(
    ("status", "expected"),
    [(200, None), (429, TransientToolError), (503, TransientToolError),
     (408, TransientToolError), (400, PermanentToolError), (401, PermanentToolError)],
)
def test_http_status_classification(status, expected):
    assert classify_http_status(status) is expected


@pytest.mark.parametrize(
    ("code", "expected"),
    [(20000, None), (40100, PermanentToolError), (40501, PermanentToolError),
     (40429, TransientToolError), (50000, TransientToolError)],
)
def test_provider_code_classification(code, expected):
    assert classify_provider_code(code) is expected


def test_retry_succeeds_after_transient_failures():
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise TransientToolError("429 slow down")
        return "ok"

    result, outcome = call_with_retry(
        flaky, RetryPolicy(max_attempts=4, base_delay=0.001), label="t", sleep=lambda _: None
    )
    assert result == "ok"
    assert outcome.attempts == 3


def test_permanent_error_is_not_retried():
    attempts = {"n": 0}

    def bad_request():
        attempts["n"] += 1
        raise PermanentToolError("400 bad request")

    with pytest.raises(PermanentToolError):
        call_with_retry(
            bad_request, RetryPolicy(max_attempts=4), label="t", sleep=lambda _: None
        )
    assert attempts["n"] == 1


def test_retry_gives_up_and_reports_attempt_count():
    def always_down():
        raise TransientToolError("upstream down")

    with pytest.raises(TransientToolError) as excinfo:
        call_with_retry(
            always_down,
            RetryPolicy(max_attempts=3, base_delay=0.001),
            label="t",
            sleep=lambda _: None,
        )
    assert excinfo.value.context.attempts == 3


def test_backoff_grows_and_is_bounded():
    policy = RetryPolicy(base_delay=1.0, max_delay=8.0, jitter=False)
    assert [policy.delay_for(n) for n in range(1, 6)] == [1.0, 2.0, 4.0, 8.0, 8.0]


def test_jitter_stays_within_the_backoff_envelope():
    policy = RetryPolicy(base_delay=1.0, max_delay=8.0, jitter=True)
    samples = [policy.delay_for(3) for _ in range(200)]
    assert all(0 <= s <= 4.0 for s in samples)
    assert len(set(samples)) > 1


def test_retry_after_header_overrides_computed_backoff():
    policy = RetryPolicy(base_delay=1.0, max_delay=8.0, jitter=False)
    assert policy.delay_for(4, retry_after=0.5) == 0.5


def test_circuit_opens_then_half_opens_then_closes():
    breakers = CircuitBreakerRegistry(failure_threshold=3, reset_timeout=0.05)
    key = "/v3/serp"

    for _ in range(3):
        breakers.before_call(key)
        breakers.record_failure(key)

    with pytest.raises(CircuitOpenError):
        breakers.before_call(key)

    import time

    time.sleep(0.06)
    breakers.before_call(key)  # admitted as the half-open probe
    assert breakers.snapshot()[key]["state"] == "half_open"
    breakers.record_success(key)
    assert breakers.snapshot()[key]["state"] == "closed"


def test_a_failed_probe_reopens_the_circuit():
    breakers = CircuitBreakerRegistry(failure_threshold=2, reset_timeout=0.01)
    key = "/v3/labs"
    for _ in range(2):
        breakers.record_failure(key)

    import time

    time.sleep(0.02)
    breakers.before_call(key)
    breakers.record_failure(key)
    assert breakers.snapshot()[key]["state"] == "open"


def test_breakers_are_isolated_per_dependency():
    breakers = CircuitBreakerRegistry(failure_threshold=1, reset_timeout=10)
    breakers.record_failure("/v3/serp")
    with pytest.raises(CircuitOpenError):
        breakers.before_call("/v3/serp")
    breakers.before_call("/v3/labs")


def test_open_circuit_fails_fast_instead_of_burning_the_retry_budget():
    breakers = CircuitBreakerRegistry(failure_threshold=1, reset_timeout=10)
    breakers.record_failure("/v3/serp")
    slept: list[float] = []

    def blocked():
        breakers.before_call("/v3/serp")
        return "unreachable"

    with pytest.raises(CircuitOpenError):
        call_with_retry(
            blocked,
            RetryPolicy(max_attempts=4, base_delay=1.0),
            label="t",
            sleep=slept.append,
        )
    assert slept == []


def test_executor_surfaces_exhausted_retries_as_a_failed_invocation(settings, metrics, monkeypatch):
    monkeypatch.setenv("MOCK_ALWAYS_FAIL_TOOLS", "serp_organic_results")
    from app.config import reload_settings

    reloaded = reload_settings()
    executor = ToolExecutor(MockDataForSEOClient(reloaded), reloaded, metrics)
    invocation = executor.invoke("serp_organic_results", {"keyword": "best crm"})

    assert invocation.ok is False
    assert invocation.attempts == reloaded.retry_max_attempts
    assert invocation.error["retryable"] is True


def test_executor_recovers_when_the_fault_is_transient(settings, metrics):
    calls = {"n": 0}

    class FlakyOnce:
        mode = "mock"

        def execute(self, endpoint, payload, *, tool=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TransientToolError("timeout", ErrorContext(endpoint=endpoint))
            return MockDataForSEOClient(settings).execute(endpoint, payload, tool=tool)

        def close(self):
            pass

    executor = ToolExecutor(FlakyOnce(), settings, metrics)
    invocation = executor.invoke("serp_organic_results", {"keyword": "best crm"})

    assert invocation.ok is True
    assert invocation.attempts == 2
    assert metrics.snapshot()["api_success_rate"] == 1.0


def test_http_errors_carry_the_provider_message(settings, monkeypatch):
    # DataForSEO returns the actionable reason in the body even on a non-2xx.
    import httpx

    from app.clients.dataforseo import LiveDataForSEOClient

    monkeypatch.setenv("DATAFORSEO_LOGIN", "user")
    monkeypatch.setenv("DATAFORSEO_PASSWORD", "pass")
    from app.config import reload_settings

    reloaded = reload_settings()
    client = LiveDataForSEOClient(reloaded)

    def fake_post(*_args, **_kwargs):
        return httpx.Response(
            403,
            json={
                "status_code": 40104,
                "status_message": "Please verify your account before using the API.",
            },
            request=httpx.Request("POST", "https://api.dataforseo.com/v3/x"),
        )

    monkeypatch.setattr(client._client, "post", fake_post)
    with pytest.raises(PermanentToolError) as excinfo:
        client.execute("/v3/x", [{}])

    assert "40104" in excinfo.value.message
    assert "verify your account" in excinfo.value.message
    assert excinfo.value.context.provider_code == 40104
    assert excinfo.value.retryable is False
