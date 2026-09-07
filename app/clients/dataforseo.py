import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote_plus

import httpx

from app.config import Settings
from app.observability.logging import get_logger
from app.resilience.errors import (
    ErrorContext,
    PermanentToolError,
    TransientToolError,
    classify_http_status,
    classify_provider_code,
)

log = get_logger(__name__)
FIXTURES = Path(__file__).parent / "fixtures"


class DataForSEOClient(Protocol):
    mode: str

    def execute(
        self, endpoint: str, payload: list[dict[str, Any]], *, tool: str | None = None
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...


def _unwrap(endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
    """DataForSEO answers 200 OK even when the task itself failed, and nests the real
    status two levels down. Collapsing both levels here means every caller upstream
    sees one error taxonomy instead of three."""
    top = int(body.get("status_code", 0))
    failure = classify_provider_code(top)
    if failure is not None:
        raise failure(
            f"DataForSEO rejected the request: {body.get('status_message', top)}",
            ErrorContext(endpoint=endpoint, provider_code=top),
        )

    tasks = body.get("tasks") or []
    if not tasks:
        raise PermanentToolError(
            "DataForSEO returned no tasks",
            ErrorContext(endpoint=endpoint, provider_code=top),
        )

    task = tasks[0]
    task_code = int(task.get("status_code", 0))
    failure = classify_provider_code(task_code)
    if failure is not None:
        raise failure(
            f"DataForSEO task failed: {task.get('status_message', task_code)}",
            ErrorContext(endpoint=endpoint, provider_code=task_code),
        )
    return body


class LiveDataForSEOClient:
    mode = "live"

    def __init__(self, settings: Settings):
        creds = settings.dataforseo_credentials
        if creds is None:
            raise PermanentToolError(
                "DATAFORSEO_MODE=live requires DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD"
            )
        self._client = httpx.Client(
            base_url=settings.dataforseo_base_url,
            auth=creds,
            timeout=httpx.Timeout(
                connect=settings.dataforseo_connect_timeout,
                read=settings.dataforseo_read_timeout,
                write=settings.dataforseo_read_timeout,
                pool=settings.dataforseo_connect_timeout,
            ),
            headers={"Content-Type": "application/json"},
        )

    def execute(
        self, endpoint: str, payload: list[dict[str, Any]], *, tool: str | None = None
    ) -> dict[str, Any]:
        # `tool` is routing metadata for the mock's fault injection; nothing about it
        # belongs on the wire, and DataForSEO rejects unknown fields outright.
        try:
            response = self._client.post(endpoint, json=payload)
        except httpx.TimeoutException as exc:
            raise TransientToolError(
                f"timeout calling {endpoint}", ErrorContext(endpoint=endpoint)
            ) from exc
        except httpx.TransportError as exc:
            raise TransientToolError(
                f"transport error calling {endpoint}: {exc}", ErrorContext(endpoint=endpoint)
            ) from exc

        failure = classify_http_status(response.status_code)
        if failure is not None:
            retry_after = _parse_retry_after(response.headers.get("Retry-After"))
            # DataForSEO puts the actionable reason in the body even on a non-2xx -
            # "verify your account", "not authorized" - so surfacing only the HTTP
            # status throws away the one line that tells you how to fix it.
            provider_code, provider_message = _provider_error(response)
            ctx = ErrorContext(
                endpoint=endpoint,
                status_code=response.status_code,
                provider_code=provider_code,
            )
            detail = f"HTTP {response.status_code} from {endpoint}"
            if provider_message:
                detail = f"{detail}: [{provider_code}] {provider_message}"
            if failure is TransientToolError:
                raise TransientToolError(detail, ctx, retry_after=retry_after)
            raise failure(detail, ctx)

        try:
            body = response.json()
        except ValueError as exc:
            raise TransientToolError(
                f"non-JSON body from {endpoint}", ErrorContext(endpoint=endpoint)
            ) from exc
        return _unwrap(endpoint, body)

    def close(self) -> None:
        self._client.close()


def _provider_error(response: httpx.Response) -> tuple[int | None, str | None]:
    try:
        body = response.json()
    except ValueError:
        return None, None
    if not isinstance(body, dict):
        return None, None
    code = body.get("status_code")
    return (int(code) if isinstance(code, int) else None), body.get("status_message")


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


_FIXTURE_BY_ENDPOINT = {
    "/v3/serp/google/organic/live/advanced": "serp_organic.json",
    "/v3/serp/google/ai_mode/live/advanced": "ai_mode.json",
    "/v3/ai_optimization/chat_gpt/llm_responses/live": "llm_responses.json",
    "/v3/keywords_data/google_ads/search_volume/live": "search_volume.json",
    "/v3/dataforseo_labs/google/keyword_ideas/live": "keyword_ideas.json",
}

_FILLER_DOMAINS = [
    "g2.com",
    "capterra.com",
    "reddit.com",
    "forbes.com",
    "techradar.com",
    "zapier.com",
    "pcmag.com",
    "trustradius.com",
    "softwareadvice.com",
    "getapp.com",
]

_MODIFIERS = [
    "alternatives",
    "pricing",
    "review",
    "vs competitors",
    "for small business",
    "free trial",
]


class MockDataForSEOClient:
    """Replays the real response envelopes from ./fixtures with values derived
    deterministically from (endpoint, keyword), so a given profile produces the same
    report on every run while still varying across queries. Fault injection lives here
    too - it is the only honest way to exercise the retry and fallback edges offline."""

    mode = "mock"

    def __init__(
        self,
        settings: Settings,
        *,
        target_domain: str | None = None,
        competitor_domains: list[str] | None = None,
        latency: float = 0.02,
    ):
        self._settings = settings
        self._target = (target_domain or "example.com").lower()
        self._competitors = [d.lower() for d in (competitor_domains or [])]
        self._latency = latency
        self._always_fail = settings.always_fail_tools
        self._failure_rate = settings.mock_failure_rate
        self._attempts: dict[str, int] = {}
        self._templates: dict[str, dict[str, Any]] = {}

    def _template(self, endpoint: str) -> dict[str, Any]:
        name = _FIXTURE_BY_ENDPOINT.get(endpoint)
        if name is None:
            raise PermanentToolError(
                f"no fixture registered for {endpoint}", ErrorContext(endpoint=endpoint)
            )
        if name not in self._templates:
            self._templates[name] = json.loads((FIXTURES / name).read_text())
        return json.loads(json.dumps(self._templates[name]))

    def _maybe_fail(self, endpoint: str, tool_name: str | None) -> None:
        if tool_name and tool_name in self._always_fail:
            raise TransientToolError(
                f"injected upstream outage for {tool_name}",
                ErrorContext(endpoint=endpoint, tool=tool_name, status_code=503),
            )
        if self._failure_rate <= 0:
            return
        # Deterministic per (endpoint, attempt) so a flaky call recovers on retry
        # instead of failing forever, which is what a real transient fault looks like.
        seen = self._attempts.get(endpoint, 0)
        self._attempts[endpoint] = seen + 1
        rng = random.Random(f"{endpoint}:{seen}")
        if rng.random() < self._failure_rate:
            raise TransientToolError(
                f"injected transient failure on {endpoint} (attempt {seen + 1})",
                ErrorContext(endpoint=endpoint, status_code=429),
                retry_after=0.05,
            )

    def execute(
        self, endpoint: str, payload: list[dict[str, Any]], *, tool: str | None = None
    ) -> dict[str, Any]:
        params = payload[0] if payload else {}
        self._maybe_fail(endpoint, tool)
        if self._latency:
            time.sleep(self._latency)

        body = self._template(endpoint)
        task = body["tasks"][0]
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S +00:00")

        if endpoint.endswith("/organic/live/advanced"):
            self._fill_serp(task, params, now)
        elif endpoint.endswith("/ai_mode/live/advanced"):
            self._fill_ai_mode(task, params, now)
        elif endpoint.endswith("/llm_responses/live"):
            self._fill_llm(task, params, now)
        elif endpoint.endswith("/search_volume/live"):
            self._fill_volume(task, params)
        elif endpoint.endswith("/keyword_ideas/live"):
            self._fill_ideas(task, params)

        return _unwrap(endpoint, body)

    def _rng(self, *parts: str) -> random.Random:
        return random.Random("|".join(parts))

    def _ranked_domains(self, keyword: str, depth: int) -> list[str]:
        rng = self._rng("serp", keyword, self._target)
        pool = list(dict.fromkeys(self._competitors + _FILLER_DOMAINS))
        rng.shuffle(pool)
        ranked = pool[:depth]
        # The target ranks for roughly half of the queries, and when it does it is
        # usually mid-page. A profile that ranked everywhere would make the whole
        # opportunity score meaningless.
        if rng.random() < 0.55:
            position = rng.randint(1, min(depth, 18)) - 1
            ranked.insert(position, self._target)
        return ranked[:depth]

    def _fill_serp(self, task: dict[str, Any], params: dict[str, Any], now: str) -> None:
        keyword = str(params.get("keyword", "")).strip()
        depth = int(params.get("depth", 20))
        result = task["result"][0]
        result["keyword"] = keyword
        result["datetime"] = now
        result["check_url"] = f"https://www.google.com/search?q={quote_plus(keyword)}"
        task["data"]["keyword"] = keyword

        rng = self._rng("serp-meta", keyword)
        items = []
        for idx, domain in enumerate(self._ranked_domains(keyword, depth), start=1):
            items.append(
                {
                    "type": "organic",
                    "rank_group": idx,
                    "rank_absolute": idx,
                    "domain": domain,
                    "title": f"{domain.split('.')[0].title()} - {keyword}",
                    "url": f"https://{domain}/{keyword.replace(' ', '-')}",
                    "description": f"Guide to {keyword} from {domain}.",
                    "breadcrumb": f"https://{domain} › blog",
                    "is_featured_snippet": idx == 1 and rng.random() < 0.3,
                }
            )
        result["items"] = items
        result["items_count"] = len(items)
        result["se_results_count"] = rng.randint(1_000_000, 90_000_000)

    def _fill_ai_mode(self, task: dict[str, Any], params: dict[str, Any], now: str) -> None:
        keyword = str(params.get("keyword", "")).strip()
        rng = self._rng("ai", keyword, self._target)
        cited = self._competitors[:2] + _FILLER_DOMAINS[:2]
        if rng.random() < 0.35:
            cited.insert(rng.randint(0, len(cited)), self._target)

        named = ", ".join(d.split(".")[0].title() for d in cited)
        result = task["result"][0]
        result["keyword"] = keyword
        result["datetime"] = now
        task["data"]["keyword"] = keyword
        result["items"] = [
            {
                "type": "ai_mode_message",
                "text": (
                    f"For {keyword}, the options most often recommended are {named}. "
                    "Choice usually comes down to team size, integrations and budget."
                ),
                "references": [
                    {
                        "type": "ai_mode_reference",
                        "source": domain,
                        "domain": domain,
                        "url": f"https://{domain}/{keyword.replace(' ', '-')}",
                        "title": f"{domain.split('.')[0].title()} on {keyword}",
                    }
                    for domain in cited
                ],
            }
        ]
        result["items_count"] = 1

    def _fill_llm(self, task: dict[str, Any], params: dict[str, Any], now: str) -> None:
        prompt = str(params.get("user_prompt", "")).strip()
        model = str(params.get("model_name", "gpt-4o-mini"))
        rng = self._rng("llm", prompt, self._target)
        mentioned = self._competitors[:3]
        if rng.random() < 0.4:
            mentioned.insert(rng.randint(0, len(mentioned)), self._target)
        named = ", ".join(d.split(".")[0].title() for d in mentioned) or "several vendors"

        result = task["result"][0]
        result["model_name"] = model
        result["datetime"] = now
        result["output_tokens"] = rng.randint(180, 420)
        task["data"]["model_name"] = model
        task["data"]["user_prompt"] = prompt
        result["items"] = [
            {
                "type": "message",
                "sections": [
                    {
                        "type": "text",
                        "text": (
                            f"In response to \"{prompt}\", the tools most commonly named are "
                            f"{named}. Each is positioned differently on price and depth."
                        ),
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": f"https://{d}/",
                                "title": d.split(".")[0].title(),
                            }
                            for d in mentioned
                        ],
                    }
                ],
            }
        ]

    def _fill_volume(self, task: dict[str, Any], params: dict[str, Any]) -> None:
        keywords = params.get("keywords") or []
        items = []
        for kw in keywords:
            rng = self._rng("vol", str(kw))
            volume = int(rng.lognormvariate(6.2, 1.35))
            items.append(
                {
                    "keyword": kw,
                    "location_code": 2840,
                    "language_code": params.get("language_code", "en"),
                    "search_partners": False,
                    "competition": rng.choice(["LOW", "MEDIUM", "HIGH"]),
                    "competition_index": rng.randint(5, 98),
                    "search_volume": max(20, min(volume, 240_000)),
                    "low_top_of_page_bid": round(rng.uniform(0.4, 6.0), 2),
                    "high_top_of_page_bid": round(rng.uniform(6.0, 24.0), 2),
                    "cpc": round(rng.uniform(1.0, 18.0), 2),
                }
            )
        task["result"] = items
        task["result_count"] = len(items)

    def _fill_ideas(self, task: dict[str, Any], params: dict[str, Any]) -> None:
        seeds = params.get("keywords") or params.get("seed_keywords") or []
        limit = int(params.get("limit", 15))
        items: list[dict[str, Any]] = []
        for seed in seeds:
            for modifier in _MODIFIERS:
                if len(items) >= limit:
                    break
                phrase = f"{seed} {modifier}"
                rng = self._rng("idea", phrase)
                items.append(
                    {
                        "se_type": "google",
                        "keyword": phrase,
                        "location_code": 2840,
                        "language_code": params.get("language_code", "en"),
                        "keyword_info": {
                            "search_volume": max(30, int(rng.lognormvariate(5.6, 1.2))),
                            "competition_level": rng.choice(["LOW", "MEDIUM", "HIGH"]),
                            "competition": round(rng.random(), 2),
                            "cpc": round(rng.uniform(0.8, 15.0), 2),
                        },
                        "keyword_properties": {"keyword_difficulty": rng.randint(4, 92)},
                    }
                )
        result = task["result"][0]
        result["items"] = items
        result["items_count"] = len(items)
        result["total_count"] = len(items)

    def close(self) -> None:
        return None


def build_client(
    settings: Settings,
    *,
    target_domain: str | None = None,
    competitor_domains: list[str] | None = None,
) -> DataForSEOClient:
    if settings.dataforseo_mode == "live":
        log.info("using live DataForSEO client", extra={"base_url": settings.dataforseo_base_url})
        return LiveDataForSEOClient(settings)
    return MockDataForSEOClient(
        settings, target_domain=target_domain, competitor_domains=competitor_domains
    )
