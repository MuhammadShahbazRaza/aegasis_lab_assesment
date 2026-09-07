from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_session
from app.api.errors import NotFound
from app.api.schemas import (
    InsightOut,
    NodeExecutionOut,
    RunRequest,
    RunResponse,
    RunTrace,
)
from app.db.models import DiscoveredQuery
from app.db.repository import ProfileRepository, RunRepository
from app.domain import VisibilityStatus
from app.service import PipelineService

router = APIRouter(prefix="/api/v1", tags=["pipeline"])


@router.post("/profiles/{profile_uuid}/run", response_model=RunResponse)
def trigger_run(
    profile_uuid: str,
    payload: RunRequest | None = None,
    session: Session = Depends(get_session),
):
    profile = ProfileRepository(session).get(profile_uuid)
    if profile is None:
        raise NotFound("profile not found", {"profile_uuid": profile_uuid})

    run = PipelineService(session).run_profile(
        profile, question=payload.question if payload else None
    )
    session.flush()

    rows = (
        session.query(DiscoveredQuery)
        .filter(DiscoveredQuery.run_uuid == run.run_uuid)
        .order_by(DiscoveredQuery.opportunity_score.desc())
        .limit(5)
        .all()
    )
    return RunResponse(
        run_uuid=run.run_uuid,
        profile_uuid=profile_uuid,
        status=run.status,
        degraded=run.degraded,
        question=run.question,
        retrieval_calls_planned=run.planned_call_count,
        records_normalized=run.normalized_record_count,
        top_insights=[
            InsightOut(
                query_uuid=r.query_uuid,
                query_text=r.query_text,
                opportunity_score=r.opportunity_score,
                visibility_status=VisibilityStatus(r.visibility_status),
                estimated_search_volume=r.estimated_search_volume,
                competitive_difficulty=r.competitive_difficulty,
                competitors_present=list(r.competitors_present or []),
            )
            for r in rows
        ],
        report=run.report,
        token_usage=run.token_usage or {},
        duration_ms=round(run.duration_ms, 2),
        node_path=list(run.node_path or []),
        rejected_tool_calls=list(run.rejected_tool_calls or []),
        retrieval_errors=list(run.retrieval_errors or []),
        metrics=run.metrics or {},
    )


@router.get("/runs/{run_uuid}", response_model=RunTrace)
def get_run_trace(run_uuid: str, session: Session = Depends(get_session)):
    """Not in the brief's endpoint list, but section 4 asks for a way to inspect DAG
    runs; this is the node-by-node view that pairs with the correlation-ID logs."""
    run = RunRepository(session).get(run_uuid)
    if run is None:
        raise NotFound("run not found", {"run_uuid": run_uuid})
    return RunTrace(
        run_uuid=run.run_uuid,
        profile_uuid=run.profile_uuid,
        status=run.status,
        degraded=run.degraded,
        started_at=run.started_at,
        finished_at=run.finished_at,
        duration_ms=round(run.duration_ms, 2),
        node_path=list(run.node_path or []),
        nodes=[
            NodeExecutionOut(
                sequence=n.sequence,
                node=n.node,
                status=n.status,
                duration_ms=round(n.duration_ms, 2),
                retries=n.retries,
                error=n.error,
            )
            for n in run.node_executions
        ],
        metrics=run.metrics or {},
        retrieval_errors=list(run.retrieval_errors or []),
    )
