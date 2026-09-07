from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


class VisibilityStatus(StrEnum):
    VISIBLE = "visible"
    NOT_VISIBLE = "not_visible"
    UNKNOWN = "unknown"


class RunStatus(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class Priority(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SurfaceType(StrEnum):
    ORGANIC = "organic"
    AI_OVERVIEW = "ai_overview"
    LLM_ANSWER = "llm_answer"
    KEYWORD_METRICS = "keyword_metrics"
    KEYWORD_IDEAS = "keyword_ideas"


class PlannedCall(BaseModel):
    """One retrieval the planner committed to, before any HTTP happens."""

    model_config = ConfigDict(frozen=False)

    call_id: str
    tool: str
    args: dict[str, Any]
    query_text: str
    rationale: str = ""
    surface: SurfaceType


class NormalizedRecord(BaseModel):
    """Provider-shaped payloads collapse into this one schema so the analysis agent
    never has to know which endpoint a fact came from."""

    call_id: str
    tool: str
    surface: SurfaceType
    query_text: str
    domain_visible: bool | None = None
    visibility_position: int | None = None
    competitors_present: list[str] = Field(default_factory=list)
    search_volume: int | None = None
    competition_index: int | None = None
    cpc: float | None = None
    cited_sources: list[str] = Field(default_factory=list)
    answer_excerpt: str | None = None
    related_queries: list[dict[str, Any]] = Field(default_factory=list)
    collected_at: datetime = Field(default_factory=utcnow)


class QueryInsight(BaseModel):
    query_uuid: str
    query_text: str
    estimated_search_volume: int = 0
    competitive_difficulty: int = Field(default=50, ge=0, le=100)
    opportunity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    domain_visible: bool = False
    visibility_position: int | None = None
    visibility_status: VisibilityStatus = VisibilityStatus.UNKNOWN
    surfaces_checked: list[SurfaceType] = Field(default_factory=list)
    competitors_present: list[str] = Field(default_factory=list)
    ai_surface_present: bool = False
    evidence: str = ""
    discovered_at: datetime = Field(default_factory=utcnow)


class Recommendation(BaseModel):
    recommendation_uuid: str
    target_query_uuid: str
    content_type: str
    title: str
    rationale: str
    target_keywords: list[str] = Field(default_factory=list)
    priority: Priority = Priority.MEDIUM


class FinalReport(BaseModel):
    headline: str
    summary_markdown: str
    visibility_score: float = Field(ge=0.0, le=1.0)
    queries_analyzed: int
    queries_visible: int
    top_competitors: list[str] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class ProfileContext(BaseModel):
    profile_uuid: str
    name: str
    domain: str
    industry: str | None = None
    description: str | None = None
    competitors: list[str] = Field(default_factory=list)
