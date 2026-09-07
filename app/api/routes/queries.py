from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import get_session
from app.api.errors import NotFound
from app.api.schemas import (
    PageMeta,
    QueryOut,
    QueryPage,
    RecheckResponse,
    RecommendationList,
    RecommendationOut,
)
from app.db.repository import (
    ProfileRepository,
    QueryRepository,
    RecommendationRepository,
    RunRepository,
)
from app.domain import VisibilityStatus
from app.service import PipelineService

router = APIRouter(prefix="/api/v1", tags=["queries"])


def _to_out(row) -> QueryOut:
    return QueryOut(
        query_uuid=row.query_uuid,
        query_text=row.query_text,
        estimated_search_volume=row.estimated_search_volume,
        competitive_difficulty=row.competitive_difficulty,
        opportunity_score=row.opportunity_score,
        domain_visible=row.domain_visible,
        visibility_position=row.visibility_position,
        visibility_status=VisibilityStatus(row.visibility_status),
        surfaces_checked=list(row.surfaces_checked or []),
        competitors_present=list(row.competitors_present or []),
        ai_surface_present=row.ai_surface_present,
        evidence=row.evidence,
        discovered_at=row.discovered_at,
    )


def _latest_run_or_404(session: Session, profile_uuid: str):
    if ProfileRepository(session).get(profile_uuid) is None:
        raise NotFound("profile not found", {"profile_uuid": profile_uuid})
    run = RunRepository(session).latest_for_profile(profile_uuid)
    if run is None:
        raise NotFound(
            "no pipeline run for this profile yet",
            {"profile_uuid": profile_uuid, "hint": "POST /api/v1/profiles/{uuid}/run first"},
        )
    return run


@router.get("/profiles/{profile_uuid}/queries", response_model=QueryPage)
def list_queries(
    profile_uuid: str,
    min_score: float | None = Query(default=None, ge=0.0, le=1.0),
    status: VisibilityStatus | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_session),
):
    run = _latest_run_or_404(session, profile_uuid)
    rows, total = QueryRepository(session).list_for_run(
        run.run_uuid,
        min_score=min_score,
        status=status.value if status else None,
        page=page,
        per_page=per_page,
    )
    return QueryPage(
        profile_uuid=profile_uuid,
        run_uuid=run.run_uuid,
        pagination=PageMeta(
            page=page,
            per_page=per_page,
            total=total,
            total_pages=(total + per_page - 1) // per_page,
        ),
        queries=[_to_out(r) for r in rows],
    )


@router.get("/profiles/{profile_uuid}/recommendations", response_model=RecommendationList)
def list_recommendations(profile_uuid: str, session: Session = Depends(get_session)):
    run = _latest_run_or_404(session, profile_uuid)
    rows = RecommendationRepository(session).list_for_run(run.run_uuid)
    return RecommendationList(
        profile_uuid=profile_uuid,
        run_uuid=run.run_uuid,
        recommendations=[
            RecommendationOut(
                recommendation_uuid=r.recommendation_uuid,
                target_query_uuid=r.target_query_uuid,
                content_type=r.content_type,
                title=r.title,
                rationale=r.rationale,
                target_keywords=list(r.target_keywords or []),
                priority=r.priority,
            )
            for r in rows
        ],
    )


@router.post("/queries/{query_uuid}/recheck", response_model=RecheckResponse)
def recheck_query(query_uuid: str, session: Session = Depends(get_session)):
    row = QueryRepository(session).get(query_uuid)
    if row is None:
        raise NotFound("query not found", {"query_uuid": query_uuid})
    profile = ProfileRepository(session).get(row.profile_uuid)
    if profile is None:
        raise NotFound("owning profile not found", {"profile_uuid": row.profile_uuid})

    result = PipelineService(session).recheck_query(row, profile)
    session.flush()
    run = result["run"]
    return RecheckResponse(
        query_uuid=row.query_uuid,
        run_uuid=run.run_uuid,
        status=run.status,
        changed=bool(result["changed"]),
        duration_ms=round(float(result["duration_ms"]), 2),
        query=_to_out(row),
    )
