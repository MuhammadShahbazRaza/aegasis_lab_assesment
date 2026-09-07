import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.types import UtcDateTime


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


class Profile(Base):
    __tablename__ = "profiles"

    profile_uuid: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    industry: Mapped[str | None] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)
    competitors: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=_now)

    runs: Mapped[list["PipelineRun"]] = relationship(
        back_populates="profile", cascade="all, delete-orphan", order_by="PipelineRun.started_at"
    )


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    run_uuid: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    profile_uuid: Mapped[str] = mapped_column(
        ForeignKey("profiles.profile_uuid", ondelete="CASCADE"), index=True
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="completed")
    trigger: Mapped[str] = mapped_column(String(20), nullable=False, default="full_run")
    degraded: Mapped[bool] = mapped_column(Boolean, default=False)
    planned_call_count: Mapped[int] = mapped_column(Integer, default=0)
    normalized_record_count: Mapped[int] = mapped_column(Integer, default=0)
    rejected_tool_calls: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    retrieval_errors: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    node_path: Mapped[list[str]] = mapped_column(JSON, default=list)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    token_usage: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    report: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    started_at: Mapped[datetime] = mapped_column(UtcDateTime, default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    profile: Mapped[Profile] = relationship(back_populates="runs")
    queries: Mapped[list["DiscoveredQuery"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    recommendations: Mapped[list["Recommendation"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    node_executions: Mapped[list["NodeExecution"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="NodeExecution.sequence"
    )


class DiscoveredQuery(Base):
    __tablename__ = "discovered_queries"

    query_uuid: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_uuid: Mapped[str] = mapped_column(
        ForeignKey("pipeline_runs.run_uuid", ondelete="CASCADE"), index=True
    )
    profile_uuid: Mapped[str] = mapped_column(
        ForeignKey("profiles.profile_uuid", ondelete="CASCADE"), index=True
    )
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    estimated_search_volume: Mapped[int] = mapped_column(Integer, default=0)
    competitive_difficulty: Mapped[int] = mapped_column(Integer, default=50)
    opportunity_score: Mapped[float] = mapped_column(Float, default=0.0)
    domain_visible: Mapped[bool] = mapped_column(Boolean, default=False)
    visibility_position: Mapped[int | None] = mapped_column(Integer)
    visibility_status: Mapped[str] = mapped_column(String(20), default="unknown")
    surfaces_checked: Mapped[list[str]] = mapped_column(JSON, default=list)
    competitors_present: Mapped[list[str]] = mapped_column(JSON, default=list)
    ai_surface_present: Mapped[bool] = mapped_column(Boolean, default=False)
    evidence: Mapped[str | None] = mapped_column(Text)
    discovered_at: Mapped[datetime] = mapped_column(UtcDateTime, default=_now)

    run: Mapped[PipelineRun] = relationship(back_populates="queries")

    # Both list endpoints filter on score and sort descending; SQLite will not use an
    # index for the sort otherwise.
    __table_args__ = (
        Index("ix_queries_profile_score", "profile_uuid", "opportunity_score"),
        Index("ix_queries_run_status", "run_uuid", "visibility_status"),
    )


class Recommendation(Base):
    __tablename__ = "recommendations"

    recommendation_uuid: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_uuid: Mapped[str] = mapped_column(
        ForeignKey("pipeline_runs.run_uuid", ondelete="CASCADE"), index=True
    )
    profile_uuid: Mapped[str] = mapped_column(
        ForeignKey("profiles.profile_uuid", ondelete="CASCADE"), index=True
    )
    target_query_uuid: Mapped[str | None] = mapped_column(
        ForeignKey("discovered_queries.query_uuid", ondelete="SET NULL")
    )
    content_type: Mapped[str] = mapped_column(String(50), default="blog_post")
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    rationale: Mapped[str] = mapped_column(Text, default="")
    target_keywords: Mapped[list[str]] = mapped_column(JSON, default=list)
    priority: Mapped[str] = mapped_column(String(10), default="medium")
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=_now)

    run: Mapped[PipelineRun] = relationship(back_populates="recommendations")


class NodeExecution(Base):
    """One row per node run. This is the persisted half of the trace: the JSON logs
    carry the detail, this carries the shape of the run so it can be queried later."""

    __tablename__ = "node_executions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_uuid: Mapped[str] = mapped_column(
        ForeignKey("pipeline_runs.run_uuid", ondelete="CASCADE"), index=True
    )
    sequence: Mapped[int] = mapped_column(Integer, default=0)
    node: Mapped[str] = mapped_column(String(60), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="ok")
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    retries: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    recorded_at: Mapped[datetime] = mapped_column(UtcDateTime, default=_now)

    run: Mapped[PipelineRun] = relationship(back_populates="node_executions")
