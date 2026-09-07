import pytest

from app.clients.dataforseo import MockDataForSEOClient
from app.domain import PlannedCall, ProfileContext, SurfaceType
from app.graph.nodes.extraction import _matches, build_extraction
from app.graph.runtime import GraphRuntime
from app.llm.fake import ScriptedChatModel
from app.observability.metrics import RunMetrics
from app.tools.executor import ToolExecutor

PROFILE = ProfileContext(
    profile_uuid="p1",
    name="Surfer SEO",
    domain="surferseo.com",
    industry="SEO Software",
    competitors=["clearscope.io", "marketmuse.com"],
)


@pytest.mark.parametrize(
    ("candidate", "domain", "expected"),
    [
        ("surferseo.com", "surferseo.com", True),
        ("www.surferseo.com", "surferseo.com", True),
        ("https://blog.surferseo.com/post", "surferseo.com", True),
        ("notsurferseo.com", "surferseo.com", False),
        ("surferseo.com.evil.net", "surferseo.com", False),
        (None, "surferseo.com", False),
    ],
)
def test_domain_matching_rejects_lookalikes(candidate, domain, expected):
    assert _matches(candidate, domain) is expected


def _run_extraction(settings, tool, args, surface):
    metrics = RunMetrics("r")
    client = MockDataForSEOClient(
        settings, target_domain=PROFILE.domain, competitor_domains=PROFILE.competitors
    )
    executor = ToolExecutor(client, settings, metrics)
    call = PlannedCall(
        call_id="c1", tool=tool, args=args, query_text=args.get("keyword", "best seo software"),
        surface=surface,
    )
    invocation = executor.invoke(tool, args, call_id="c1")
    assert invocation.ok

    runtime = GraphRuntime(
        settings=settings, llm=ScriptedChatModel(), executor=executor, metrics=metrics
    )
    node = build_extraction(runtime)
    return node({"profile": PROFILE, "planned_calls": [call], "invocations": [invocation]})


def test_serp_payload_yields_position_and_competitors(settings):
    out = _run_extraction(
        settings, "serp_organic_results", {"keyword": "best seo software"}, SurfaceType.ORGANIC
    )
    record = out["records"][0]

    assert record.surface is SurfaceType.ORGANIC
    assert record.query_text == "best seo software"
    assert record.domain_visible is (record.visibility_position is not None)
    assert set(record.competitors_present) <= set(PROFILE.competitors)
    assert record.cited_sources


def test_ai_overview_payload_yields_citations_and_an_excerpt(settings):
    out = _run_extraction(
        settings, "ai_overview_snapshot", {"keyword": "best seo software"}, SurfaceType.AI_OVERVIEW
    )
    record = out["records"][0]

    assert record.surface is SurfaceType.AI_OVERVIEW
    assert record.answer_excerpt
    assert record.cited_sources
    assert record.domain_visible in (True, False)


def test_llm_answer_payload_yields_an_excerpt(settings):
    out = _run_extraction(
        settings,
        "llm_answer_visibility",
        {"user_prompt": "which seo tool should I buy"},
        SurfaceType.LLM_ANSWER,
    )
    record = out["records"][0]
    assert record.surface is SurfaceType.LLM_ANSWER
    assert record.answer_excerpt


def test_volume_payload_expands_to_one_record_per_keyword(settings):
    keywords = ["best seo software", "seo software pricing", "surfer seo alternatives"]
    out = _run_extraction(
        settings, "keyword_search_volume", {"keywords": keywords}, SurfaceType.KEYWORD_METRICS
    )
    assert len(out["records"]) == len(keywords)
    assert {r.query_text for r in out["records"]} == set(keywords)
    assert all(r.search_volume and r.competition_index is not None for r in out["records"])


def test_failed_invocations_are_skipped_not_fabricated(settings, monkeypatch):
    monkeypatch.setenv("MOCK_ALWAYS_FAIL_TOOLS", "serp_organic_results")
    from app.config import reload_settings

    reloaded = reload_settings()
    metrics = RunMetrics("r")
    client = MockDataForSEOClient(reloaded, target_domain=PROFILE.domain)
    executor = ToolExecutor(client, reloaded, metrics)
    call = PlannedCall(
        call_id="c1", tool="serp_organic_results", args={"keyword": "best seo software"},
        query_text="best seo software", surface=SurfaceType.ORGANIC,
    )
    invocation = executor.invoke("serp_organic_results", call.args, call_id="c1")
    assert invocation.ok is False

    runtime = GraphRuntime(
        settings=reloaded, llm=ScriptedChatModel(), executor=executor, metrics=metrics
    )
    out = build_extraction(runtime)(
        {"profile": PROFILE, "planned_calls": [call], "invocations": [invocation]}
    )
    assert out["records"] == []


def test_a_truncated_payload_costs_one_record_not_the_run(settings):
    metrics = RunMetrics("r")
    client = MockDataForSEOClient(settings, target_domain=PROFILE.domain)
    executor = ToolExecutor(client, settings, metrics)
    good = executor.invoke("serp_organic_results", {"keyword": "best seo software"}, call_id="c1")
    broken = executor.invoke("serp_organic_results", {"keyword": "seo pricing"}, call_id="c2")
    broken.raw = {"tasks": [{"result": [{"items": "not-a-list"}]}]}

    calls = [
        PlannedCall(call_id="c1", tool="serp_organic_results", args={}, query_text="a",
                    surface=SurfaceType.ORGANIC),
        PlannedCall(call_id="c2", tool="serp_organic_results", args={}, query_text="b",
                    surface=SurfaceType.ORGANIC),
    ]
    runtime = GraphRuntime(
        settings=settings, llm=ScriptedChatModel(), executor=executor, metrics=metrics
    )
    out = build_extraction(runtime)(
        {"profile": PROFILE, "planned_calls": calls, "invocations": [good, broken]}
    )
    assert len(out["records"]) == 1
