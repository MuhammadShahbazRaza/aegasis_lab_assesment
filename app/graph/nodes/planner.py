import uuid
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.domain import PlannedCall, ProfileContext, SurfaceType
from app.graph.runtime import GraphRuntime
from app.llm.provider import usage_from
from app.observability.logging import get_logger
from app.resilience.errors import ToolValidationError
from app.tools.registry import TOOLS, validate_tool_call

log = get_logger(__name__)

SURFACE_BY_TOOL = {
    "serp_organic_results": SurfaceType.ORGANIC,
    "ai_overview_snapshot": SurfaceType.AI_OVERVIEW,
    "llm_answer_visibility": SurfaceType.LLM_ANSWER,
    "keyword_search_volume": SurfaceType.KEYWORD_METRICS,
    "related_keyword_ideas": SurfaceType.KEYWORD_IDEAS,
}

_SYSTEM = """[role:planner] You plan data retrieval for a search-visibility analysis. \
You do not analyse anything and you do not write prose; you only decide which API \
calls to make.

Emit SEVERAL tool calls in this one response - a single call is never a complete plan. \
Aim for four to {budget}.

Rules:
- Emit tool calls only. Any explanation belongs in the tool arguments, not in text.
- Pick two or three high-intent queries a buyer would actually search.
- For EACH of those queries emit both serp_organic_results AND ai_overview_snapshot. \
Visibility on one surface says nothing about the other.
- Emit exactly one keyword_search_volume call listing every keyword you chose.
- Emit one llm_answer_visibility call phrased the way a buyer would ask an assistant, \
not as a bare keyword.
- Stay within {budget} tool calls total."""

_HUMAN = """Brand: {name}
Domain: {domain}
Industry: {industry}
What they do: {description}
Known competitors: {competitors}

Research question: {question}

Plan the retrieval calls needed to answer this."""


def _query_text_for(tool: str, args: dict[str, Any]) -> str:
    if "keyword" in args:
        return str(args["keyword"])
    if "user_prompt" in args:
        return str(args["user_prompt"])
    for key in ("keywords", "seed_keywords"):
        if args.get(key):
            return str(args[key][0])
    return ""


def heuristic_plan(profile: ProfileContext, question: str, budget: int) -> list[PlannedCall]:
    """Deterministic plan used when the model is unavailable or produced nothing
    usable. Deliberately narrow: enough coverage to return a real report, not an
    attempt to imitate the model's judgement."""
    topic = (profile.industry or profile.name).lower()
    seeds = [
        f"best {topic}",
        f"{profile.name} alternatives",
        f"{topic} for teams",
    ]
    calls: list[PlannedCall] = []

    def add(tool: str, args: dict[str, Any], rationale: str) -> None:
        if len(calls) >= budget:
            return
        calls.append(
            PlannedCall(
                call_id=str(uuid.uuid4()),
                tool=tool,
                args=args,
                query_text=_query_text_for(tool, args),
                rationale=rationale,
                surface=SURFACE_BY_TOOL[tool],
            )
        )

    add(
        "keyword_search_volume",
        {"keywords": seeds},
        "baseline demand for the seed set",
    )
    for seed in seeds[:2]:
        add("serp_organic_results", {"keyword": seed, "depth": 20}, "organic visibility")
        add("ai_overview_snapshot", {"keyword": seed}, "AI Overview visibility")
    add(
        "llm_answer_visibility",
        {"user_prompt": f"What is the best {topic} available right now?"},
        "assistant-answer visibility",
    )
    return calls


_KEYWORD_SURFACES = (SurfaceType.ORGANIC, SurfaceType.AI_OVERVIEW)


def _make_call(tool: str, args: dict[str, Any], rationale: str) -> PlannedCall:
    return PlannedCall(
        call_id=str(uuid.uuid4()),
        tool=tool,
        args=args,
        query_text=_query_text_for(tool, args),
        rationale=rationale,
        surface=SURFACE_BY_TOOL[tool],
    )


def _keyword_candidates(planned: list[PlannedCall]) -> list[str]:
    """Queries the plan is really about. LLM prompts are excluded: they are sentences,
    not search terms, and feeding one to a SERP endpoint returns noise."""
    seen: dict[str, None] = {}
    for call in planned:
        if call.surface in _KEYWORD_SURFACES:
            seen.setdefault(str(call.args.get("keyword", "")).strip().lower(), None)
        elif call.surface is SurfaceType.KEYWORD_METRICS:
            for keyword in call.args.get("keywords", []):
                seen.setdefault(str(keyword).strip().lower(), None)
        elif call.surface is SurfaceType.KEYWORD_IDEAS:
            for seed in call.args.get("seed_keywords", []):
                seen.setdefault(str(seed).strip().lower(), None)
    return [k for k in seen if k]


def _ensure_coverage(
    planned: list[PlannedCall], budget: int, depth: int = 2
) -> tuple[list[PlannedCall], list[str]]:
    """Top up a thin plan so it can actually answer the question.

    Models vary a lot in how many tool calls they will emit in one turn - some return a
    single call no matter how the prompt is worded. A plan is only useful if it checks
    both the organic and the AI surface for the queries it names and knows their demand,
    so any missing leg is added deterministically and recorded. This validates the plan
    for *coverage*, where validate_tool_call validates it for *correctness*.
    """
    notes: list[str] = []
    if not planned:
        return planned, notes

    candidates = _keyword_candidates(planned)[:depth]
    if not candidates:
        return planned, notes

    covered = {
        surface: {
            str(c.args.get("keyword", "")).strip().lower()
            for c in planned
            if c.surface is surface
        }
        for surface in _KEYWORD_SURFACES
    }
    calls = list(planned)
    added = 0

    for tool, surface in (
        ("serp_organic_results", SurfaceType.ORGANIC),
        ("ai_overview_snapshot", SurfaceType.AI_OVERVIEW),
    ):
        for keyword in candidates:
            if len(calls) >= budget:
                break
            if keyword in covered[surface]:
                continue
            calls.append(_make_call(tool, {"keyword": keyword}, "coverage top-up"))
            added += 1

    if not any(c.surface is SurfaceType.KEYWORD_METRICS for c in calls) and len(calls) < budget:
        calls.append(
            _make_call("keyword_search_volume", {"keywords": candidates}, "coverage top-up")
        )
        added += 1

    if added:
        notes.append(f"topped up {added} call(s) to cover organic, AI and demand data")
    return calls, notes


def build_planner(runtime: GraphRuntime):
    budget = runtime.settings.max_planned_calls
    model = runtime.llm.bind_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.args_model.model_json_schema(),
                },
            }
            for spec in TOOLS.values()
        ]
    )

    def planner(state: dict[str, Any]) -> dict[str, Any]:
        profile: ProfileContext = state["profile"]
        messages = [
            SystemMessage(_SYSTEM.format(budget=budget)),
            HumanMessage(
                _HUMAN.format(
                    name=profile.name,
                    domain=profile.domain,
                    industry=profile.industry or "unspecified",
                    description=profile.description or "unspecified",
                    competitors=", ".join(profile.competitors) or "none supplied",
                    question=state["question"],
                )
            ),
        ]

        notes: list[str] = []
        rejected: list[dict[str, Any]] = []
        planned: list[PlannedCall] = []

        try:
            response = model.invoke(messages)
        except Exception as exc:
            log.warning("planner model call failed", extra={"error": str(exc)})
            return {
                "planned_calls": [],
                "plan_notes": [f"planner model unavailable: {exc}"],
                "rejected_tool_calls": [],
                "plan_source": "model_error",
            }

        runtime.metrics.record_tokens(usage_from(response))
        raw_calls = list(getattr(response, "tool_calls", []) or [])

        for call in raw_calls:
            if len(planned) >= budget:
                notes.append(f"dropped '{call.get('name')}': plan budget of {budget} reached")
                continue
            name = call.get("name", "")
            args = call.get("args") or {}
            try:
                validated = validate_tool_call(name, args)
            except ToolValidationError as exc:
                # A rejected call is data, not a crash: it is recorded, surfaced in the
                # run response, and the rest of the plan proceeds.
                rejected.append(exc.as_dict())
                notes.append(f"rejected '{name}': {exc.message}")
                continue
            planned.append(
                PlannedCall(
                    call_id=str(uuid.uuid4()),
                    tool=validated.name,
                    args=validated.payload,
                    query_text=_query_text_for(validated.name, validated.payload),
                    rationale=str(response.content or "")[:300],
                    surface=SURFACE_BY_TOOL[validated.name],
                )
            )

        from_model = len(planned)
        planned, coverage_notes = _ensure_coverage(planned, budget)
        notes.extend(coverage_notes)

        log.info(
            "plan built",
            extra={
                "proposed": len(raw_calls),
                "accepted": from_model,
                "rejected": len(rejected),
                "after_coverage": len(planned),
            },
        )
        return {
            "planned_calls": planned,
            "plan_notes": notes,
            "rejected_tool_calls": rejected,
            "plan_source": "model+topup" if coverage_notes else "model",
        }

    return planner


def build_plan_fallback(runtime: GraphRuntime):
    def plan_fallback(state: dict[str, Any]) -> dict[str, Any]:
        profile: ProfileContext = state["profile"]
        calls = heuristic_plan(profile, state["question"], runtime.settings.max_planned_calls)
        log.warning("falling back to heuristic plan", extra={"planned_calls": len(calls)})
        return {
            "planned_calls": calls,
            "plan_notes": [*state.get("plan_notes", []), "used deterministic fallback plan"],
            "plan_source": "heuristic",
            "degraded": True,
            "flags": {"plan_fallback": True},
        }

    return plan_fallback
