from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, field_validator

from app.domain import VisibilityStatus


def _iso_z(value: datetime) -> str:
    aware = value if value.tzinfo else value.replace(tzinfo=UTC)
    return aware.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


# The spec's examples use trailing-Z timestamps; Pydantic would otherwise emit +00:00.
Timestamp = Annotated[datetime, PlainSerializer(_iso_z, return_type=str)]

_DOMAIN_STRIP = ("https://", "http://", "www.")


class ProfileCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    domain: str = Field(min_length=3, max_length=255)
    industry: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    competitors: list[str] = Field(default_factory=list, max_length=25)

    @field_validator("domain", "competitors", mode="before")
    @classmethod
    def _normalize(cls, value: Any) -> Any:
        if isinstance(value, str):
            return _clean_domain(value)
        if isinstance(value, list):
            return [_clean_domain(v) if isinstance(v, str) else v for v in value]
        return value

    @field_validator("domain")
    @classmethod
    def _has_tld(cls, value: str) -> str:
        if "." not in value:
            raise ValueError("domain must include a TLD, e.g. 'surferseo.com'")
        return value


def _clean_domain(value: str) -> str:
    cleaned = value.strip().lower()
    for prefix in _DOMAIN_STRIP:
        cleaned = cleaned.removeprefix(prefix)
    return cleaned.rstrip("/").split("/")[0]


class ProfileCreated(BaseModel):
    profile_uuid: str
    name: str
    domain: str
    status: Literal["created"] = "created"
    created_at: Timestamp


class ProfileStats(BaseModel):
    total_runs: int
    total_rechecks: int
    last_run_uuid: str | None
    last_run_status: str | None
    last_run_at: Timestamp | None
    queries_in_last_run: int
    average_opportunity_score: float | None


class ProfileDetail(BaseModel):
    profile_uuid: str
    name: str
    domain: str
    industry: str | None
    description: str | None
    competitors: list[str]
    created_at: Timestamp
    stats: ProfileStats


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str | None = Field(
        default=None,
        max_length=1000,
        description="Overrides the default research question for this profile.",
    )


class QueryOut(BaseModel):
    query_uuid: str
    query_text: str
    estimated_search_volume: int
    competitive_difficulty: int
    opportunity_score: float
    domain_visible: bool
    visibility_position: int | None
    visibility_status: VisibilityStatus
    surfaces_checked: list[str]
    competitors_present: list[str]
    ai_surface_present: bool
    evidence: str | None
    discovered_at: Timestamp


class PageMeta(BaseModel):
    page: int
    per_page: int
    total: int
    total_pages: int


class QueryPage(BaseModel):
    profile_uuid: str
    run_uuid: str
    pagination: PageMeta
    queries: list[QueryOut]


class RecommendationOut(BaseModel):
    recommendation_uuid: str
    target_query_uuid: str | None
    content_type: str
    title: str
    rationale: str
    target_keywords: list[str]
    priority: str


class RecommendationList(BaseModel):
    profile_uuid: str
    run_uuid: str
    recommendations: list[RecommendationOut]


class InsightOut(BaseModel):
    query_uuid: str
    query_text: str
    opportunity_score: float
    visibility_status: VisibilityStatus
    estimated_search_volume: int
    competitive_difficulty: int
    competitors_present: list[str]


class RunResponse(BaseModel):
    run_uuid: str
    profile_uuid: str
    status: str
    degraded: bool
    question: str
    retrieval_calls_planned: int
    records_normalized: int
    top_insights: list[InsightOut]
    report: dict[str, Any] | None
    token_usage: dict[str, int]
    duration_ms: float
    node_path: list[str]
    rejected_tool_calls: list[dict[str, Any]]
    retrieval_errors: list[dict[str, Any]]
    metrics: dict[str, Any]


class NodeExecutionOut(BaseModel):
    sequence: int
    node: str
    status: str
    duration_ms: float
    retries: int
    error: str | None


class RunTrace(BaseModel):
    run_uuid: str
    profile_uuid: str
    status: str
    degraded: bool
    started_at: Timestamp
    finished_at: Timestamp | None
    duration_ms: float
    node_path: list[str]
    nodes: list[NodeExecutionOut]
    metrics: dict[str, Any]
    retrieval_errors: list[dict[str, Any]]


class RecheckResponse(BaseModel):
    query_uuid: str
    run_uuid: str
    status: str
    changed: bool
    duration_ms: float
    query: QueryOut


class ErrorResponse(BaseModel):
    error: str
    detail: Any = None
