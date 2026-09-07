from typing import Any

from app.domain import NormalizedRecord, PlannedCall, ProfileContext, SurfaceType
from app.graph.runtime import GraphRuntime
from app.observability.logging import get_logger
from app.tools.executor import ToolInvocation

log = get_logger(__name__)


def _tasks(raw: dict[str, Any] | None) -> list[dict[str, Any]]:
    return (raw or {}).get("tasks") or []


def _first_result(raw: dict[str, Any] | None) -> dict[str, Any]:
    tasks = _tasks(raw)
    if not tasks:
        return {}
    results = tasks[0].get("result") or []
    return results[0] if results and isinstance(results[0], dict) else {}


def _all_results(raw: dict[str, Any] | None) -> list[dict[str, Any]]:
    tasks = _tasks(raw)
    return [r for r in (tasks[0].get("result") or []) if isinstance(r, dict)] if tasks else []


def _matches(candidate: str | None, domain: str) -> bool:
    if not candidate:
        return False
    host = candidate.lower().split("//")[-1].split("/")[0].removeprefix("www.")
    target = domain.lower().removeprefix("www.")
    return host == target or host.endswith("." + target)


def _extract_serp(
    invocation: ToolInvocation, call: PlannedCall, profile: ProfileContext
) -> NormalizedRecord:
    result = _first_result(invocation.raw)
    items = [i for i in (result.get("items") or []) if i.get("type") == "organic"]

    position: int | None = None
    competitors: list[str] = []
    for item in items:
        domain = item.get("domain") or ""
        if position is None and _matches(domain, profile.domain):
            position = item.get("rank_absolute") or item.get("rank_group")
        if any(_matches(domain, c) for c in profile.competitors):
            competitors.append(domain)

    return NormalizedRecord(
        call_id=call.call_id,
        tool=call.tool,
        surface=SurfaceType.ORGANIC,
        query_text=result.get("keyword") or call.query_text,
        domain_visible=position is not None,
        visibility_position=position,
        competitors_present=sorted(set(competitors)),
        cited_sources=[i.get("domain") for i in items[:10] if i.get("domain")],
    )


def _extract_ai_overview(
    invocation: ToolInvocation, call: PlannedCall, profile: ProfileContext
) -> NormalizedRecord:
    result = _first_result(invocation.raw)
    text_parts: list[str] = []
    sources: list[str] = []
    for item in result.get("items") or []:
        if item.get("text"):
            text_parts.append(str(item["text"]))
        for ref in item.get("references") or []:
            domain = ref.get("domain") or ref.get("source")
            if domain:
                sources.append(str(domain))

    answer = " ".join(text_parts).strip()
    visible = any(_matches(s, profile.domain) for s in sources) or _mentions(answer, profile)
    return NormalizedRecord(
        call_id=call.call_id,
        tool=call.tool,
        surface=SurfaceType.AI_OVERVIEW,
        query_text=result.get("keyword") or call.query_text,
        domain_visible=visible,
        # Citation order is the only position signal an AI answer gives; there is no
        # rank_absolute to read here, so cite-order stands in for it.
        visibility_position=next(
            (i for i, s in enumerate(sources, 1) if _matches(s, profile.domain)), None
        ),
        competitors_present=sorted(
            {s for s in sources if any(_matches(s, c) for c in profile.competitors)}
        ),
        cited_sources=sources,
        answer_excerpt=answer[:600] or None,
    )


def _extract_llm_answer(
    invocation: ToolInvocation, call: PlannedCall, profile: ProfileContext
) -> NormalizedRecord:
    result = _first_result(invocation.raw)
    text_parts: list[str] = []
    sources: list[str] = []
    for item in result.get("items") or []:
        for section in item.get("sections") or []:
            if section.get("text"):
                text_parts.append(str(section["text"]))
            for note in section.get("annotations") or []:
                if note.get("url"):
                    sources.append(str(note["url"]))

    answer = " ".join(text_parts).strip()
    visible = _mentions(answer, profile) or any(_matches(s, profile.domain) for s in sources)
    return NormalizedRecord(
        call_id=call.call_id,
        tool=call.tool,
        surface=SurfaceType.LLM_ANSWER,
        query_text=call.query_text,
        domain_visible=visible,
        visibility_position=next(
            (i for i, s in enumerate(sources, 1) if _matches(s, profile.domain)), None
        ),
        competitors_present=sorted(
            {c for c in profile.competitors if _mentions_domain(answer, sources, c)}
        ),
        cited_sources=sources,
        answer_excerpt=answer[:600] or None,
    )


def _mentions(text: str, profile: ProfileContext) -> bool:
    lowered = text.lower()
    brand = profile.name.lower()
    bare = profile.domain.lower().removeprefix("www.").split(".")[0]
    return brand in lowered or bare in lowered


def _mentions_domain(text: str, sources: list[str], domain: str) -> bool:
    bare = domain.lower().removeprefix("www.").split(".")[0]
    return bare in text.lower() or any(_matches(s, domain) for s in sources)


def _extract_volume(invocation: ToolInvocation, call: PlannedCall) -> list[NormalizedRecord]:
    records = []
    for row in _all_results(invocation.raw):
        if not row.get("keyword"):
            continue
        records.append(
            NormalizedRecord(
                call_id=call.call_id,
                tool=call.tool,
                surface=SurfaceType.KEYWORD_METRICS,
                query_text=str(row["keyword"]),
                search_volume=row.get("search_volume"),
                competition_index=row.get("competition_index"),
                cpc=row.get("cpc"),
            )
        )
    return records


def _extract_ideas(invocation: ToolInvocation, call: PlannedCall) -> NormalizedRecord:
    result = _first_result(invocation.raw)
    related = []
    for item in result.get("items") or []:
        info = item.get("keyword_info") or {}
        props = item.get("keyword_properties") or {}
        related.append(
            {
                "keyword": item.get("keyword"),
                "search_volume": info.get("search_volume"),
                "difficulty": props.get("keyword_difficulty"),
                "cpc": info.get("cpc"),
            }
        )
    return NormalizedRecord(
        call_id=call.call_id,
        tool=call.tool,
        surface=SurfaceType.KEYWORD_IDEAS,
        query_text=call.query_text,
        related_queries=related,
    )


def build_extraction(runtime: GraphRuntime):
    """Pure transformation: provider payloads in, one flat schema out. It makes no
    judgements about what the numbers mean and calls nothing external, which is why
    it is the one node that can be tested against recorded payloads alone."""

    def extraction(state: dict[str, Any]) -> dict[str, Any]:
        profile: ProfileContext = state["profile"]
        by_id = {c.call_id: c for c in state.get("planned_calls", [])}
        records: list[NormalizedRecord] = []
        skipped = 0

        for invocation in state.get("invocations", []):
            call = by_id.get(invocation.call_id or "")
            if call is None or not invocation.ok:
                skipped += 1
                continue
            try:
                if invocation.tool == "serp_organic_results":
                    records.append(_extract_serp(invocation, call, profile))
                elif invocation.tool == "ai_overview_snapshot":
                    records.append(_extract_ai_overview(invocation, call, profile))
                elif invocation.tool == "llm_answer_visibility":
                    records.append(_extract_llm_answer(invocation, call, profile))
                elif invocation.tool == "keyword_search_volume":
                    records.extend(_extract_volume(invocation, call))
                elif invocation.tool == "related_keyword_ideas":
                    records.append(_extract_ideas(invocation, call))
                else:
                    skipped += 1
            except (KeyError, TypeError, AttributeError) as exc:
                # A schema drift on one endpoint should cost that record, not the run.
                skipped += 1
                log.warning(
                    "could not normalize payload",
                    extra={
                        "tool": invocation.tool,
                        "call_id": invocation.call_id,
                        "error": str(exc),
                    },
                )

        log.info(
            "normalization complete",
            extra={"records": len(records), "skipped_invocations": skipped},
        )
        return {"records": records, "flags": {"normalized_records": len(records)}}

    return extraction
