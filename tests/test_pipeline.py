import pytest

from app.clients.dataforseo import MockDataForSEOClient
from app.config import reload_settings
from app.db.repository import QueryRepository, RecommendationRepository
from app.domain import RunStatus, SurfaceType, VisibilityStatus
from app.graph.builder import build_graph
from app.graph.runtime import GraphRuntime
from app.graph.state import initial_state
from app.llm.fake import ScriptedChatModel
from app.observability.metrics import RunMetrics
from app.service import PipelineService
from app.tools.executor import ToolExecutor


def make_runtime(settings, profile_ctx, scripted=None, metrics=None):
    metrics = metrics or RunMetrics("test-run")
    client = MockDataForSEOClient(
        settings,
        target_domain=profile_ctx.domain,
        competitor_domains=profile_ctx.competitors,
    )
    return GraphRuntime(
        settings=settings,
        llm=ScriptedChatModel(responses=scripted or {}),
        executor=ToolExecutor(client, settings, metrics),
        metrics=metrics,
        client=client,
    )


def run_graph(runtime, profile_ctx, question="how visible are we?"):
    return build_graph(runtime).invoke(initial_state("run-1", profile_ctx, question))


def test_happy_path_walks_every_agent_in_order(settings, profile_context):
    metrics = RunMetrics("run-1")
    runtime = make_runtime(settings, profile_context, metrics=metrics)
    final = run_graph(runtime, profile_context)

    path = final["node_path"]
    assert path[0] == "query_planner"
    assert path[-1] == "report_assembler"
    assert path.index("retrieval_gate") < path.index("extraction_normalizer")
    assert path.index("extraction_normalizer") < path.index("analysis_synthesizer")
    assert "no_data_fallback" not in path
    assert "plan_fallback" not in path

    assert final["status"] is RunStatus.COMPLETED
    assert final["degraded"] is False
    assert len(final["planned_calls"]) > 0
    assert len(final["records"]) > 0
    assert len(final["insights"]) > 0
    assert final["report"] is not None
    assert final["report"].summary_markdown
    assert metrics.snapshot()["tokens"]["total_tokens"] > 0


def test_retrieval_fans_out_to_one_node_execution_per_planned_call(settings, profile_context):
    metrics = RunMetrics("run-1")
    runtime = make_runtime(settings, profile_context, metrics=metrics)
    final = run_graph(runtime, profile_context)

    workers = metrics.snapshot()["nodes"]["retrieval_worker"]["calls"]
    assert workers == len(final["planned_calls"])
    assert len(final["invocations"]) == len(final["planned_calls"])


def test_insights_are_sorted_by_opportunity_score(settings, profile_context):
    final = run_graph(make_runtime(settings, profile_context), profile_context)
    scores = [i.opportunity_score for i in final["insights"]]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_empty_plan_routes_to_the_heuristic_planner(settings, profile_context):
    runtime = make_runtime(settings, profile_context, scripted={"planner": []})
    final = run_graph(runtime, profile_context)

    assert "plan_fallback" in final["node_path"]
    assert final["plan_source"] == "heuristic"
    assert final["degraded"] is True
    assert final["report"] is not None
    assert len(final["planned_calls"]) > 0


def test_malformed_tool_arguments_are_recorded_and_the_rest_of_the_plan_runs(
    settings, profile_context
):
    plan = [
        {"name": "serp_organic_results", "args": {"keyword": "best seo software"}, "id": "1"},
        {"name": "serp_organic_results", "args": {"depth": 20}, "id": "2"},
        {"name": "nonexistent_tool", "args": {}, "id": "3"},
        {"name": "keyword_search_volume", "args": {"keywords": ["best seo software"]}, "id": "4"},
    ]
    runtime = make_runtime(settings, profile_context, scripted={"planner": plan})
    final = run_graph(runtime, profile_context)

    assert len(final["rejected_tool_calls"]) == 2
    assert {r["tool"] for r in final["rejected_tool_calls"]} == {
        "serp_organic_results",
        "nonexistent_tool",
    }
    # The two valid calls survive; coverage top-up may add more on top of them.
    accepted = {(c.tool, c.query_text) for c in final["planned_calls"]}
    assert ("serp_organic_results", "best seo software") in accepted
    assert ("keyword_search_volume", "best seo software") in accepted
    assert "plan_fallback" not in final["node_path"]
    assert final["status"] is RunStatus.COMPLETED


def test_partial_upstream_outage_degrades_but_still_reports(
    settings, profile_context, monkeypatch
):
    monkeypatch.setenv(
        "MOCK_ALWAYS_FAIL_TOOLS", "serp_organic_results,ai_overview_snapshot,llm_answer_visibility"
    )
    reloaded = reload_settings()
    runtime = make_runtime(reloaded, profile_context)
    final = build_graph(runtime).invoke(initial_state("run-1", profile_context, "q"))

    assert "partial_data_fallback" in final["node_path"]
    assert final["degraded"] is True
    assert final["status"] is RunStatus.PARTIAL
    # The one endpoint that stayed up is still parsed rather than thrown away.
    assert len(final["records"]) > 0
    assert final["report"] is not None
    assert any("degraded" in c.lower() for c in final["report"].caveats)
    assert all(
        i.visibility_status is VisibilityStatus.UNKNOWN for i in final["insights"]
    )


def test_total_outage_returns_an_empty_but_well_formed_report(
    settings, profile_context, monkeypatch
):
    monkeypatch.setenv(
        "MOCK_ALWAYS_FAIL_TOOLS",
        "serp_organic_results,ai_overview_snapshot,llm_answer_visibility,"
        "keyword_search_volume,related_keyword_ideas",
    )
    reloaded = reload_settings()
    runtime = make_runtime(reloaded, profile_context)
    final = build_graph(runtime).invoke(initial_state("run-1", profile_context, "q"))

    assert "no_data_fallback" in final["node_path"]
    assert final["status"] is RunStatus.FAILED
    assert final["records"] == []
    assert final["report"] is not None
    assert final["flags"]["degraded_reason"] == "no_usable_records"
    assert final["flags"]["failures_by_tool"]


def test_a_crashing_analysis_model_does_not_lose_the_scored_table(
    settings, profile_context
):
    def explode(_messages):
        raise RuntimeError("provider 500")

    runtime = make_runtime(settings, profile_context, scripted={"analyst": explode})
    final = run_graph(runtime, profile_context)

    assert final["status"] is RunStatus.COMPLETED
    assert len(final["insights"]) > 0
    assert final["report"].key_findings  # fell back to mechanical findings


def test_service_persists_run_queries_and_recommendations(settings, session, profile):
    run = PipelineService(session, settings).run_profile(profile)
    session.commit()

    assert run.status == RunStatus.COMPLETED.value
    assert run.planned_call_count > 0
    assert run.normalized_record_count > 0
    assert run.node_path[0] == "query_planner"
    assert run.token_usage["total_tokens"] > 0
    assert run.metrics["api_call_total"] == run.planned_call_count

    rows, total = QueryRepository(session).list_for_run(run.run_uuid)
    assert total == len(rows) > 0
    recs = RecommendationRepository(session).list_for_run(run.run_uuid)
    assert len(recs) > 0
    assert {r.target_query_uuid for r in recs} <= {r.query_uuid for r in rows}

    assert [n.node for n in run.node_executions] == run.node_path


def test_recheck_reuses_the_query_uuid_and_refreshes_metrics(settings, session, profile):
    service = PipelineService(session, settings)
    run = service.run_profile(profile)
    session.commit()

    rows, _ = QueryRepository(session).list_for_run(run.run_uuid)
    target = next(r for r in rows if r.visibility_status != "unknown")
    before = target.discovered_at

    result = service.recheck_query(target, profile)
    session.commit()

    assert result["changed"] is True
    assert result["run"].trigger == "recheck"
    assert target.query_uuid == rows[rows.index(target)].query_uuid
    assert target.discovered_at > before
    # A recheck must not re-plan: it goes straight to retrieval.
    assert "query_planner" not in (result["run"].node_path or [])


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_success_threshold_controls_the_degraded_route(
    settings, profile_context, monkeypatch, threshold
):
    monkeypatch.setenv("MOCK_ALWAYS_FAIL_TOOLS", "ai_overview_snapshot")
    monkeypatch.setenv("RETRIEVAL_SUCCESS_THRESHOLD", str(threshold))
    reloaded = reload_settings()
    runtime = make_runtime(reloaded, profile_context)
    final = build_graph(runtime).invoke(initial_state("run-1", profile_context, "q"))

    took_partial_route = "partial_data_fallback" in final["node_path"]
    assert took_partial_route is (threshold == 1.0)


def test_a_single_call_plan_is_topped_up_to_cover_both_surfaces(settings, profile_context):
    # Some models return one tool call however the prompt is worded. A plan that only
    # knows search volume cannot answer a visibility question.
    runtime = make_runtime(
        settings,
        profile_context,
        scripted={
            "planner": [
                {
                    "name": "keyword_search_volume",
                    "args": {"keywords": ["best seo software", "seo software pricing"]},
                    "id": "1",
                }
            ]
        },
    )
    final = run_graph(runtime, profile_context)

    surfaces = {c.surface for c in final["planned_calls"]}
    assert SurfaceType.ORGANIC in surfaces
    assert SurfaceType.AI_OVERVIEW in surfaces
    assert final["plan_source"] == "model+topup"
    assert any("topped up" in note for note in final["plan_notes"])
    # Topping up is not degrading: the model's plan was valid, just thin.
    assert final["degraded"] is False
    assert final["status"] is RunStatus.COMPLETED
    assert any(i.visibility_status is not VisibilityStatus.UNKNOWN for i in final["insights"])


def test_a_complete_plan_is_left_alone(settings, profile_context):
    plan = [
        {"name": "keyword_search_volume", "args": {"keywords": ["best seo software"]}, "id": "1"},
        {"name": "serp_organic_results", "args": {"keyword": "best seo software"}, "id": "2"},
        {"name": "ai_overview_snapshot", "args": {"keyword": "best seo software"}, "id": "3"},
    ]
    runtime = make_runtime(settings, profile_context, scripted={"planner": plan})
    final = run_graph(runtime, profile_context)

    assert len(final["planned_calls"]) == 3
    assert final["plan_source"] == "model"


def test_top_up_respects_the_call_budget(settings, profile_context, monkeypatch):
    monkeypatch.setenv("MAX_PLANNED_CALLS", "2")
    reloaded = reload_settings()
    runtime = make_runtime(
        reloaded,
        profile_context,
        scripted={
            "planner": [
                {
                    "name": "keyword_search_volume",
                    "args": {"keywords": ["a crm", "b crm"]},
                    "id": "1",
                }
            ]
        },
    )
    final = build_graph(runtime).invoke(initial_state("run-1", profile_context, "q"))
    assert len(final["planned_calls"]) <= 2


def test_an_empty_plan_still_falls_back_rather_than_being_topped_up(settings, profile_context):
    # Top-up extends a thin plan; it must never manufacture one out of nothing, or the
    # heuristic fallback edge would become unreachable.
    runtime = make_runtime(settings, profile_context, scripted={"planner": []})
    final = run_graph(runtime, profile_context)
    assert final["plan_source"] == "heuristic"
    assert "plan_fallback" in final["node_path"]
