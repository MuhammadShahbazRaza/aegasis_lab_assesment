from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from app.db.models import DiscoveredQuery, NodeExecution, PipelineRun, Profile, Recommendation
from app.domain import ProfileContext


class ProfileRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        *,
        name: str,
        domain: str,
        industry: str | None,
        description: str | None,
        competitors: list[str],
    ) -> Profile:
        profile = Profile(
            name=name,
            domain=domain,
            industry=industry,
            description=description,
            competitors=competitors,
        )
        self.session.add(profile)
        self.session.flush()
        return profile

    def get(self, profile_uuid: str) -> Profile | None:
        return self.session.get(Profile, profile_uuid)

    def find_by_domain(self, domain: str) -> Profile | None:
        return self.session.scalar(select(Profile).where(Profile.domain == domain))

    def context(self, profile: Profile) -> ProfileContext:
        return ProfileContext(
            profile_uuid=profile.profile_uuid,
            name=profile.name,
            domain=profile.domain,
            industry=profile.industry,
            description=profile.description,
            competitors=list(profile.competitors or []),
        )

    def summary_stats(self, profile_uuid: str) -> dict[str, Any]:
        def _count(trigger: str) -> int:
            return int(
                self.session.scalar(
                    select(func.count(PipelineRun.run_uuid)).where(
                        PipelineRun.profile_uuid == profile_uuid,
                        PipelineRun.trigger == trigger,
                    )
                )
                or 0
            )

        total_runs = _count("full_run")
        total_rechecks = _count("recheck")
        latest = self.session.scalar(
            select(PipelineRun)
            .where(
                PipelineRun.profile_uuid == profile_uuid,
                PipelineRun.trigger == "full_run",
            )
            .order_by(PipelineRun.started_at.desc())
            .limit(1)
        )
        avg_score = None
        query_count = 0
        if latest is not None:
            avg_score = self.session.scalar(
                select(func.avg(DiscoveredQuery.opportunity_score)).where(
                    DiscoveredQuery.run_uuid == latest.run_uuid
                )
            )
            query_count = (
                self.session.scalar(
                    select(func.count(DiscoveredQuery.query_uuid)).where(
                        DiscoveredQuery.run_uuid == latest.run_uuid
                    )
                )
                or 0
            )
        return {
            "total_runs": total_runs,
            "total_rechecks": total_rechecks,
            "last_run_uuid": latest.run_uuid if latest else None,
            "last_run_status": latest.status if latest else None,
            "last_run_at": latest.started_at if latest else None,
            "queries_in_last_run": query_count,
            "average_opportunity_score": round(float(avg_score), 4)
            if avg_score is not None
            else None,
        }


class RunRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, *, profile_uuid: str, question: str, trigger: str = "full_run") -> PipelineRun:
        run = PipelineRun(profile_uuid=profile_uuid, question=question, trigger=trigger)
        self.session.add(run)
        self.session.flush()
        return run

    def get(self, run_uuid: str) -> PipelineRun | None:
        return self.session.get(PipelineRun, run_uuid)

    def latest_for_profile(
        self, profile_uuid: str, trigger: str | None = "full_run"
    ) -> PipelineRun | None:
        """A recheck is a sub-run: it refreshes one query in place and does not own a
        query set of its own. Letting it win "most recent run" would empty the queries
        and recommendations endpoints the moment anyone rechecked anything."""
        stmt = select(PipelineRun).where(PipelineRun.profile_uuid == profile_uuid)
        if trigger is not None:
            stmt = stmt.where(PipelineRun.trigger == trigger)
        return self.session.scalar(stmt.order_by(PipelineRun.started_at.desc()).limit(1))

    def finish(self, run: PipelineRun, **fields: Any) -> PipelineRun:
        for key, value in fields.items():
            setattr(run, key, value)
        run.finished_at = datetime.now(UTC)
        self.session.flush()
        return run

    def add_node_execution(
        self,
        run_uuid: str,
        *,
        sequence: int,
        node: str,
        status: str,
        duration_ms: float,
        retries: int = 0,
        error: str | None = None,
    ) -> None:
        self.session.add(
            NodeExecution(
                run_uuid=run_uuid,
                sequence=sequence,
                node=node,
                status=status,
                duration_ms=duration_ms,
                retries=retries,
                error=error,
            )
        )


class QueryRepository:
    def __init__(self, session: Session):
        self.session = session

    def bulk_create(
        self, *, run_uuid: str, profile_uuid: str, insights: list[Any]
    ) -> list[DiscoveredQuery]:
        rows = [
            DiscoveredQuery(
                query_uuid=i.query_uuid,
                run_uuid=run_uuid,
                profile_uuid=profile_uuid,
                query_text=i.query_text,
                estimated_search_volume=i.estimated_search_volume,
                competitive_difficulty=i.competitive_difficulty,
                opportunity_score=i.opportunity_score,
                domain_visible=i.domain_visible,
                visibility_position=i.visibility_position,
                visibility_status=i.visibility_status.value,
                surfaces_checked=[s.value for s in i.surfaces_checked],
                competitors_present=i.competitors_present,
                ai_surface_present=i.ai_surface_present,
                evidence=i.evidence or None,
                discovered_at=i.discovered_at,
            )
            for i in insights
        ]
        self.session.add_all(rows)
        self.session.flush()
        return rows

    def get(self, query_uuid: str) -> DiscoveredQuery | None:
        return self.session.get(DiscoveredQuery, query_uuid)

    def _base(self, run_uuid: str) -> Select[tuple[DiscoveredQuery]]:
        return select(DiscoveredQuery).where(DiscoveredQuery.run_uuid == run_uuid)

    def list_for_run(
        self,
        run_uuid: str,
        *,
        min_score: float | None = None,
        status: str | None = None,
        page: int = 1,
        per_page: int = 20,
    ) -> tuple[list[DiscoveredQuery], int]:
        stmt = self._base(run_uuid)
        if min_score is not None:
            stmt = stmt.where(DiscoveredQuery.opportunity_score >= min_score)
        if status is not None:
            stmt = stmt.where(DiscoveredQuery.visibility_status == status)

        total = self.session.scalar(
            select(func.count()).select_from(stmt.subquery())
        ) or 0
        rows = list(
            self.session.scalars(
                stmt.order_by(DiscoveredQuery.opportunity_score.desc())
                .offset((page - 1) * per_page)
                .limit(per_page)
            )
        )
        return rows, int(total)

    def update_from_insight(self, row: DiscoveredQuery, insight: Any) -> DiscoveredQuery:
        row.estimated_search_volume = insight.estimated_search_volume
        row.competitive_difficulty = insight.competitive_difficulty
        row.opportunity_score = insight.opportunity_score
        row.domain_visible = insight.domain_visible
        row.visibility_position = insight.visibility_position
        row.visibility_status = insight.visibility_status.value
        row.surfaces_checked = [s.value for s in insight.surfaces_checked]
        row.competitors_present = insight.competitors_present
        row.ai_surface_present = insight.ai_surface_present
        row.evidence = insight.evidence or row.evidence
        row.discovered_at = datetime.now(UTC)
        self.session.flush()
        return row


class RecommendationRepository:
    def __init__(self, session: Session):
        self.session = session

    def bulk_create(
        self, *, run_uuid: str, profile_uuid: str, recommendations: list[Any]
    ) -> list[Recommendation]:
        rows = [
            Recommendation(
                recommendation_uuid=r.recommendation_uuid,
                run_uuid=run_uuid,
                profile_uuid=profile_uuid,
                target_query_uuid=r.target_query_uuid or None,
                content_type=r.content_type,
                title=r.title,
                rationale=r.rationale,
                target_keywords=r.target_keywords,
                priority=r.priority.value,
            )
            for r in recommendations
        ]
        self.session.add_all(rows)
        self.session.flush()
        return rows

    def list_for_run(self, run_uuid: str) -> list[Recommendation]:
        return list(
            self.session.scalars(
                select(Recommendation)
                .where(Recommendation.run_uuid == run_uuid)
                .order_by(Recommendation.created_at)
            )
        )
