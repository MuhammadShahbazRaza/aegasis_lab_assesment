import threading
import time
import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.clients.dataforseo import build_client
from app.config import Settings, get_settings
from app.db.models import DiscoveredQuery, PipelineRun, Profile
from app.db.repository import (
    QueryRepository,
    RecommendationRepository,
    RunRepository,
)
from app.domain import (
    PlannedCall,
    ProfileContext,
    QueryInsight,
    RunStatus,
    SurfaceType,
)
from app.graph.builder import build_graph, build_recheck_graph
from app.graph.runtime import GraphRuntime
from app.graph.state import initial_state
from app.llm.provider import build_chat_model
from app.observability import context
from app.observability.logging import get_logger
from app.observability.metrics import RunMetrics

log = get_logger(__name__)

DEFAULT_QUESTION = (
    "How does {name} ({domain}) show up in AI-generated answers and organic search "
    "results for the queries its buyers actually use, and where are the biggest gaps "
    "against {competitors}?"
)


class _NodeTraceSink:
    """Collects node events during the run and writes them once afterwards. Nodes fan
    out across threads, so touching the ORM session from inside them would mean
    sharing a Session across threads."""

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def __call__(self, **event: Any) -> None:
        with self._lock:
            self._events.append(event)

    @property
    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)


def default_question(profile: ProfileContext) -> str:
    return DEFAULT_QUESTION.format(
        name=profile.name,
        domain=profile.domain,
        competitors=", ".join(profile.competitors) or "its category rivals",
    )


class PipelineService:
    def __init__(self, session: Session, settings: Settings | None = None):
        self.session = session
        self.settings = settings or get_settings()
        self.runs = RunRepository(session)
        self.queries = QueryRepository(session)
        self.recommendations = RecommendationRepository(session)

    def _runtime(self, profile: ProfileContext, metrics: RunMetrics) -> GraphRuntime:
        from app.tools.executor import ToolExecutor

        client = build_client(
            self.settings,
            target_domain=profile.domain,
            competitor_domains=profile.competitors,
        )
        return GraphRuntime(
            settings=self.settings,
            llm=build_chat_model(self.settings),
            executor=ToolExecutor(client, self.settings, metrics),
            metrics=metrics,
            client=client,
        )

    def _persist_trace(self, run_uuid: str, sink: _NodeTraceSink) -> list[str]:
        path: list[str] = []
        for index, event in enumerate(sink.events, start=1):
            self.runs.add_node_execution(
                run_uuid,
                sequence=index,
                node=str(event.get("node")),
                status=str(event.get("status", "ok")),
                duration_ms=float(event.get("duration_ms", 0.0)),
                retries=int(event.get("retries", 0)),
                error=event.get("error"),
            )
            path.append(str(event.get("node")))
        return path

    def run_profile(self, profile: Profile, question: str | None = None) -> PipelineRun:
        ctx = ProfileContext(
            profile_uuid=profile.profile_uuid,
            name=profile.name,
            domain=profile.domain,
            industry=profile.industry,
            description=profile.description,
            competitors=list(profile.competitors or []),
        )
        resolved = question or default_question(ctx)
        run = self.runs.create(profile_uuid=profile.profile_uuid, question=resolved)
        run_uuid = run.run_uuid

        metrics = RunMetrics(run_uuid)
        sink = _NodeTraceSink()
        started = time.perf_counter()

        with context.bind(run_id=run_uuid, profile_uuid=profile.profile_uuid):
            log.info(
                "pipeline run started",
                extra={
                    "question": resolved,
                    "dataforseo_mode": self.settings.dataforseo_mode,
                    "llm_provider": self.settings.llm_provider,
                },
            )
            runtime: GraphRuntime | None = None
            try:
                runtime = self._runtime(ctx, metrics)
                final = build_graph(runtime, sink).invoke(
                    initial_state(run_uuid, ctx, resolved)
                )
                error: str | None = None
            except Exception as exc:
                log.exception("pipeline run crashed", extra={"error": str(exc)})
                final = {}
                error = str(exc)
            finally:
                if runtime is not None:
                    runtime.close()

            duration_ms = (time.perf_counter() - started) * 1000
            path = self._persist_trace(run_uuid, sink)
            snapshot = metrics.snapshot()

            insights: list[QueryInsight] = final.get("insights", []) or []
            report = final.get("report")
            status = final.get("status", RunStatus.FAILED)
            if error:
                status = RunStatus.FAILED

            if insights:
                self.queries.bulk_create(
                    run_uuid=run_uuid, profile_uuid=profile.profile_uuid, insights=insights
                )
            if report is not None and report.recommendations:
                self.recommendations.bulk_create(
                    run_uuid=run_uuid,
                    profile_uuid=profile.profile_uuid,
                    recommendations=report.recommendations,
                )

            self.runs.finish(
                run,
                status=status.value if isinstance(status, RunStatus) else str(status),
                degraded=bool(final.get("degraded")),
                planned_call_count=len(final.get("planned_calls", []) or []),
                normalized_record_count=len(final.get("records", []) or []),
                rejected_tool_calls=final.get("rejected_tool_calls", []) or [],
                retrieval_errors=final.get("retrieval_errors", []) or [],
                node_path=path,
                metrics=snapshot,
                token_usage=metrics.token_usage,
                report=report.model_dump(mode="json") if report is not None else None,
                error=error,
                duration_ms=duration_ms,
            )
            log.info(
                "pipeline run finished",
                extra={
                    "status": run.status,
                    "duration_ms": round(duration_ms, 2),
                    "metrics": snapshot,
                },
            )
        return run

    def recheck_query(self, row: DiscoveredQuery, profile: Profile) -> dict[str, Any]:
        ctx = ProfileContext(
            profile_uuid=profile.profile_uuid,
            name=profile.name,
            domain=profile.domain,
            industry=profile.industry,
            description=profile.description,
            competitors=list(profile.competitors or []),
        )
        run = self.runs.create(
            profile_uuid=profile.profile_uuid,
            question=f"Recheck visibility for '{row.query_text}'",
            trigger="recheck",
        )
        metrics = RunMetrics(run.run_uuid)
        sink = _NodeTraceSink()
        started = time.perf_counter()

        # A recheck deliberately skips the planner: the query is already known, so
        # re-planning would burn a model call and could return a different query set.
        calls = _recheck_plan(row.query_text)
        state = initial_state(run.run_uuid, ctx, run.question)
        state["planned_calls"] = calls

        with context.bind(run_id=run.run_uuid, profile_uuid=profile.profile_uuid):
            log.info("recheck started", extra={"query_uuid": row.query_uuid, "calls": len(calls)})
            runtime = None
            try:
                runtime = self._runtime(ctx, metrics)
                final = build_recheck_graph(runtime, sink).invoke(state)
                error = None
            except Exception as exc:
                log.exception("recheck crashed", extra={"error": str(exc)})
                final, error = {}, str(exc)
            finally:
                if runtime is not None:
                    runtime.close()

            duration_ms = (time.perf_counter() - started) * 1000
            path = self._persist_trace(run.run_uuid, sink)
            insights: list[QueryInsight] = final.get("insights", []) or []
            match = next(
                (i for i in insights if i.query_text.lower() == row.query_text.lower()),
                insights[0] if insights else None,
            )
            if match is not None:
                # Keep the original query_uuid so recommendations pointing at it stay valid.
                match.query_uuid = row.query_uuid
                self.queries.update_from_insight(row, match)

            status = (
                RunStatus.FAILED
                if error or match is None
                else (RunStatus.PARTIAL if final.get("degraded") else RunStatus.COMPLETED)
            )
            self.runs.finish(
                run,
                status=status.value,
                degraded=bool(final.get("degraded")),
                planned_call_count=len(calls),
                normalized_record_count=len(final.get("records", []) or []),
                retrieval_errors=final.get("retrieval_errors", []) or [],
                node_path=path,
                metrics=metrics.snapshot(),
                token_usage=metrics.token_usage,
                error=error,
                duration_ms=duration_ms,
            )
        return {
            "run": run,
            "query": row,
            "changed": match is not None,
            "duration_ms": duration_ms,
        }


def _recheck_plan(query_text: str) -> list[PlannedCall]:
    return [
        PlannedCall(
            call_id=str(uuid.uuid4()),
            tool="serp_organic_results",
            args={"keyword": query_text, "location_name": "United States",
                  "language_code": "en", "depth": 20},
            query_text=query_text,
            rationale="recheck organic position",
            surface=SurfaceType.ORGANIC,
        ),
        PlannedCall(
            call_id=str(uuid.uuid4()),
            tool="ai_overview_snapshot",
            args={"keyword": query_text, "location_name": "United States",
                  "language_code": "en"},
            query_text=query_text,
            rationale="recheck AI Overview citation",
            surface=SurfaceType.AI_OVERVIEW,
        ),
        PlannedCall(
            call_id=str(uuid.uuid4()),
            tool="keyword_search_volume",
            args={"keywords": [query_text], "location_name": "United States",
                  "language_code": "en"},
            query_text=query_text,
            rationale="refresh demand metrics",
            surface=SurfaceType.KEYWORD_METRICS,
        ),
    ]
