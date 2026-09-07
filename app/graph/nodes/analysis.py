import json
import uuid
from collections import Counter, defaultdict
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.domain import (
    NormalizedRecord,
    ProfileContext,
    QueryInsight,
    SurfaceType,
    VisibilityStatus,
)
from app.graph.runtime import GraphRuntime
from app.llm.provider import usage_from
from app.observability.logging import get_logger
from app.scoring import difficulty_from, opportunity_score

log = get_logger(__name__)

_SYSTEM = """[role:analyst] You interpret already-normalized search-visibility data. \
You never invent numbers: search volume, difficulty and ranking positions are given \
to you and must be quoted as-is.

Return JSON only, matching:
{"findings": ["..."], "query_notes": [{"query_text": "...", "evidence": "..."}]}

findings: 3-6 short, specific observations about where the brand is and is not \
visible, naming competitors where the data shows them. \
query_notes: one line per query explaining what the numbers show. \
No preamble, no markdown fences."""


def _aggregate(records: list[NormalizedRecord], profile: ProfileContext) -> list[QueryInsight]:
    """Fold the per-surface records into one row per query.

    A query can be checked on three surfaces and each returns a different kind of
    evidence, so visibility is resolved as: seen anywhere = visible; checked but
    absent everywhere = not visible; only metric rows = unknown.
    """
    grouped: dict[str, list[NormalizedRecord]] = defaultdict(list)
    for record in records:
        key = record.query_text.strip().lower()
        if key:
            grouped[key].append(record)

    insights: list[QueryInsight] = []
    for group in grouped.values():
        visibility_records = [r for r in group if r.domain_visible is not None]
        volumes = [r.search_volume for r in group if r.search_volume]
        positions = [r.visibility_position for r in group if r.visibility_position]

        if not visibility_records:
            status = VisibilityStatus.UNKNOWN
        elif any(r.domain_visible for r in visibility_records):
            status = VisibilityStatus.VISIBLE
        else:
            status = VisibilityStatus.NOT_VISIBLE

        competitors = sorted({c for r in group for c in r.competitors_present})
        insight = QueryInsight(
            query_uuid=str(uuid.uuid4()),
            query_text=group[0].query_text,
            estimated_search_volume=max(volumes) if volumes else 0,
            competitive_difficulty=difficulty_from(group),
            domain_visible=status is VisibilityStatus.VISIBLE,
            visibility_position=min(positions) if positions else None,
            visibility_status=status,
            surfaces_checked=sorted({r.surface for r in group}),
            competitors_present=competitors,
            ai_surface_present=any(
                r.surface in (SurfaceType.AI_OVERVIEW, SurfaceType.LLM_ANSWER) for r in group
            ),
        )
        insights.append(insight)

    return insights


def _expand_from_ideas(records: list[NormalizedRecord]) -> list[dict[str, Any]]:
    ideas: list[dict[str, Any]] = []
    for record in records:
        if record.surface is SurfaceType.KEYWORD_IDEAS:
            ideas.extend(record.related_queries)
    return ideas


def _parse_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text.strip("`")
        text = text.removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start : end + 1])


def build_analysis(runtime: GraphRuntime):
    settings = runtime.settings

    def analysis(state: dict[str, Any]) -> dict[str, Any]:
        profile: ProfileContext = state["profile"]
        records: list[NormalizedRecord] = state.get("records", [])
        insights = _aggregate(records, profile)

        # Scores come from the formula, never from the model. The LLM is used for
        # the qualitative read; ranking has to be reproducible and auditable.
        for insight in insights:
            insight.opportunity_score = opportunity_score(insight, settings)
        insights.sort(key=lambda i: i.opportunity_score, reverse=True)

        findings: list[str] = []
        if insights:
            payload = {
                "brand": profile.name,
                "domain": profile.domain,
                "competitors": profile.competitors,
                "queries": [
                    {
                        "query_text": i.query_text,
                        "search_volume": i.estimated_search_volume,
                        "competitive_difficulty": i.competitive_difficulty,
                        "opportunity_score": i.opportunity_score,
                        "visibility_status": i.visibility_status.value,
                        "visibility_position": i.visibility_position,
                        "surfaces_checked": [s.value for s in i.surfaces_checked],
                        "competitors_present": i.competitors_present,
                    }
                    for i in insights[:12]
                ],
                "related_query_ideas": _expand_from_ideas(records)[:10],
            }
            try:
                response = runtime.llm.invoke(
                    [
                        SystemMessage(_SYSTEM),
                        HumanMessage(json.dumps(payload, default=str)),
                    ]
                )
                runtime.metrics.record_tokens(usage_from(response))
                parsed = _parse_json(str(response.content))
                findings = [str(f) for f in parsed.get("findings", [])][:6]
                notes = {
                    str(n.get("query_text", "")).lower(): str(n.get("evidence", ""))
                    for n in parsed.get("query_notes", [])
                    if isinstance(n, dict)
                }
                for insight in insights:
                    insight.evidence = notes.get(insight.query_text.lower(), "")
            except Exception as exc:
                # Losing the narrative is survivable; the scored table is the part
                # customers act on, so the node degrades instead of failing.
                log.warning("analysis narrative unavailable", extra={"error": str(exc)})
                findings = _mechanical_findings(insights, profile)

        visible = sum(1 for i in insights if i.domain_visible)
        log.info(
            "analysis complete",
            extra={
                "queries": len(insights),
                "visible": visible,
                "findings": len(findings),
                "top_score": insights[0].opportunity_score if insights else None,
            },
        )
        return {
            "insights": insights,
            "flags": {
                "analysis_findings": findings,
                "queries_visible": visible,
                "top_competitors": [
                    d for d, _ in Counter(
                        c for i in insights for c in i.competitors_present
                    ).most_common(5)
                ],
            },
        }

    return analysis


def _mechanical_findings(insights: list[QueryInsight], profile: ProfileContext) -> list[str]:
    out: list[str] = []
    gaps = [i for i in insights if not i.domain_visible][:3]
    for gap in gaps:
        rivals = ", ".join(gap.competitors_present[:3]) or "no tracked competitor"
        out.append(
            f"{profile.domain} is absent for '{gap.query_text}' "
            f"({gap.estimated_search_volume:,} monthly searches); {rivals} present."
        )
    wins = [i for i in insights if i.domain_visible and i.visibility_position]
    if wins:
        best = min(wins, key=lambda i: i.visibility_position or 999)
        out.append(
            f"Best existing position is #{best.visibility_position} for '{best.query_text}'."
        )
    return out
