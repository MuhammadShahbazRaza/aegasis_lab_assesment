import json
import uuid
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.domain import (
    FinalReport,
    Priority,
    ProfileContext,
    QueryInsight,
    Recommendation,
    RunStatus,
)
from app.graph.runtime import GraphRuntime
from app.llm.provider import usage_from
from app.observability.logging import get_logger
from app.scoring import priority_for

log = get_logger(__name__)

_SYSTEM = """[role:reporter] You assemble the final deliverable from analysis that is \
already complete. You do not re-score anything and you do not introduce new data.

Return JSON only:
{"headline": "...", "summary_markdown": "...", "recommendations": [
  {"target_query_uuid": "...", "content_type": "blog_post|landing_page|faq|comparison_page",
   "title": "...", "rationale": "...", "target_keywords": ["..."], "priority": "high|medium|low"}
]}

summary_markdown: 120-200 words a marketing lead could act on, in markdown. \
One recommendation per high-opportunity query, using the exact target_query_uuid given. \
No preamble, no markdown fences around the JSON."""


def _parse_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text.strip("`")
        text = text.removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start : end + 1])


def _fallback_recommendations(insights: list[QueryInsight]) -> list[Recommendation]:
    out: list[Recommendation] = []
    for insight in insights[:5]:
        if insight.domain_visible and (insight.visibility_position or 99) <= 3:
            continue
        out.append(
            Recommendation(
                recommendation_uuid=str(uuid.uuid4()),
                target_query_uuid=insight.query_uuid,
                content_type="comparison_page" if insight.competitors_present else "blog_post",
                title=f"{insight.query_text.title()}: buyer's guide",
                rationale=(
                    f"No owned coverage ranking for this query at "
                    f"{insight.estimated_search_volume:,} monthly searches "
                    f"(difficulty {insight.competitive_difficulty}/100)."
                ),
                target_keywords=[insight.query_text],
                priority=priority_for(insight.opportunity_score),
            )
        )
    return out


def _summary_markdown(
    profile: ProfileContext, insights: list[QueryInsight], findings: list[str], visible: int
) -> str:
    lines = [
        f"## AI & search visibility - {profile.name}",
        "",
        f"Analysed **{len(insights)}** queries; `{profile.domain}` appears in "
        f"**{visible}** of them.",
        "",
    ]
    if findings:
        lines += ["### What the data shows", *[f"- {f}" for f in findings], ""]
    if insights:
        lines += [
            "### Highest-opportunity gaps",
            "| Query | Volume | Difficulty | Score | Visible |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
        for i in insights[:5]:
            lines.append(
                f"| {i.query_text} | {i.estimated_search_volume:,} | "
                f"{i.competitive_difficulty} | {i.opportunity_score:.2f} | "
                f"{'yes' if i.domain_visible else 'no'} |"
            )
    return "\n".join(lines)


def build_report(runtime: GraphRuntime):
    """Assembly only. Every number it prints was computed upstream; if the model is
    unreachable the deterministic branch below still produces a complete report."""

    def report(state: dict[str, Any]) -> dict[str, Any]:
        profile: ProfileContext = state["profile"]
        insights: list[QueryInsight] = state.get("insights", [])
        flags = state.get("flags", {})
        findings = list(flags.get("analysis_findings", []))
        visible = int(flags.get("queries_visible", 0))
        degraded = bool(state.get("degraded"))

        headline = f"{profile.name} visibility across search and AI answers"
        summary = _summary_markdown(profile, insights, findings, visible)
        recommendations = _fallback_recommendations(insights)

        if insights:
            payload = {
                "brand": profile.name,
                "domain": profile.domain,
                "queries_analyzed": len(insights),
                "queries_visible": visible,
                "findings": findings,
                "queries": [
                    {
                        "target_query_uuid": i.query_uuid,
                        "query_text": i.query_text,
                        "opportunity_score": i.opportunity_score,
                        "search_volume": i.estimated_search_volume,
                        "competitive_difficulty": i.competitive_difficulty,
                        "visibility_status": i.visibility_status.value,
                        "competitors_present": i.competitors_present,
                    }
                    for i in insights[:8]
                ],
            }
            try:
                response = runtime.llm.invoke(
                    [SystemMessage(_SYSTEM), HumanMessage(json.dumps(payload, default=str))]
                )
                runtime.metrics.record_tokens(usage_from(response))
                parsed = _parse_json(str(response.content))
                headline = str(parsed.get("headline") or headline)
                summary = str(parsed.get("summary_markdown") or summary)
                model_recs = _coerce_recommendations(parsed.get("recommendations"), insights)
                if model_recs:
                    recommendations = model_recs
            except Exception as exc:
                log.warning("report narrative unavailable", extra={"error": str(exc)})

        visibility_score = round(visible / len(insights), 4) if insights else 0.0
        caveats: list[str] = []
        if degraded:
            caveats.append(
                "Run degraded: some retrievals failed or a fallback path was taken; "
                "coverage is partial."
            )
        if state.get("rejected_tool_calls"):
            caveats.append(
                f"{len(state['rejected_tool_calls'])} planned tool call(s) were rejected "
                "by argument validation and never executed."
            )
        if runtime.settings.dataforseo_mode == "mock":
            caveats.append("DataForSEO ran in mock mode; figures are synthetic.")

        final = FinalReport(
            headline=headline,
            summary_markdown=summary,
            visibility_score=visibility_score,
            queries_analyzed=len(insights),
            queries_visible=visible,
            top_competitors=list(flags.get("top_competitors", [])),
            key_findings=findings,
            recommendations=recommendations,
            caveats=caveats,
        )
        status = _resolve_status(state, insights)
        log.info(
            "report assembled",
            extra={
                "status": status.value,
                "recommendations": len(recommendations),
                "visibility_score": visibility_score,
            },
        )
        return {"report": final, "status": status}

    return report


def _resolve_status(state: dict[str, Any], insights: list[QueryInsight]) -> RunStatus:
    if not insights:
        return RunStatus.FAILED
    if state.get("degraded"):
        return RunStatus.PARTIAL
    return RunStatus.COMPLETED


def _coerce_recommendations(
    raw: Any, insights: list[QueryInsight]
) -> list[Recommendation]:
    if not isinstance(raw, list):
        return []
    valid_ids = {i.query_uuid for i in insights}
    fallback_id = insights[0].query_uuid if insights else ""
    out: list[Recommendation] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        target = str(item.get("target_query_uuid", ""))
        # The model occasionally echoes a query string where a uuid belongs; pin it
        # to a real row rather than emitting a dangling foreign key.
        if target not in valid_ids:
            match = next(
                (i.query_uuid for i in insights if i.query_text.lower() == target.lower()),
                fallback_id,
            )
            target = match
        try:
            priority = Priority(str(item.get("priority", "medium")).lower())
        except ValueError:
            priority = Priority.MEDIUM
        keywords = item.get("target_keywords")
        out.append(
            Recommendation(
                recommendation_uuid=str(uuid.uuid4()),
                target_query_uuid=target,
                content_type=str(item.get("content_type") or "blog_post"),
                title=str(item.get("title") or "Untitled recommendation")[:200],
                rationale=str(item.get("rationale") or "")[:1000],
                target_keywords=[str(k) for k in keywords] if isinstance(keywords, list) else [],
                priority=priority,
            )
        )
    return out
