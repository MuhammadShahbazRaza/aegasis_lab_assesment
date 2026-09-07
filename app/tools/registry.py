from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from app.resilience.errors import ErrorContext, ToolValidationError
from app.tools.schemas import (
    AiOverviewArgs,
    KeywordIdeasArgs,
    LlmVisibilityArgs,
    SearchVolumeArgs,
    SerpOrganicArgs,
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    endpoint: str
    args_model: type[BaseModel]
    description: str
    # Measured against the live API on 2026-09-06, in USD per task. These are not
    # guesses: keyword_search_volume costs 45x a SERP call, which is the opposite of
    # what the endpoint names suggest and is why its description tells the planner to
    # batch. Used for budgeting and for the cost line in the run metrics.
    cost_usd: float = 0.0
    # Our argument name -> the provider's field name, where the two differ. The tool
    # schema is the model-facing contract and is worded for the model's benefit; the
    # wire format is DataForSEO's and is not negotiable. Keeping them separate means a
    # provider rename does not force a prompt change.
    field_map: Mapping[str, str] = field(default_factory=dict)

    def to_payload(self, args: dict[str, Any]) -> dict[str, Any]:
        return {self.field_map.get(key, key): value for key, value in args.items()}


TOOLS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in [
        ToolSpec(
            name="serp_organic_results",
            endpoint="/v3/serp/google/organic/live/advanced",
            args_model=SerpOrganicArgs,
            cost_usd=0.002,
            description=(
                "Fetch live Google organic SERP results for one keyword. Use this to find out "
                "which domains rank for a query and at what position. One call per keyword."
            ),
        ),
        ToolSpec(
            name="ai_overview_snapshot",
            endpoint="/v3/serp/google/ai_mode/live/advanced",
            args_model=AiOverviewArgs,
            description=(
                "Fetch Google's AI Overview / AI Mode answer for one keyword, including the "
                "sources it cites. Use this to check whether a brand is referenced in Google's "
                "generated answer rather than only in the blue links."
            ),
            cost_usd=0.004,
        ),
        ToolSpec(
            name="llm_answer_visibility",
            endpoint="/v3/ai_optimization/chat_gpt/llm_responses/live",
            args_model=LlmVisibilityArgs,
            description=(
                "Sample how an LLM assistant answers a buyer-intent prompt and which brands and "
                "domains it names. Use this to measure visibility inside AI assistant answers."
            ),
            cost_usd=0.0008,
        ),
        ToolSpec(
            name="keyword_search_volume",
            endpoint="/v3/keywords_data/google_ads/search_volume/live",
            args_model=SearchVolumeArgs,
            cost_usd=0.09,
            description=(
                "Look up monthly search volume, competition index and CPC for up to 20 keywords "
                "in a single call. Batch keywords together instead of calling this per keyword."
            ),
        ),
        ToolSpec(
            name="related_keyword_ideas",
            endpoint="/v3/dataforseo_labs/google/keyword_ideas/live",
            args_model=KeywordIdeasArgs,
            description=(
                "Expand seed keywords into related queries real users search. Use this once, "
                "early, when the question is broad and the specific queries to check are unknown."
            ),
            cost_usd=0.0126,
            # DataForSEO Labs calls this field "keywords"; the tool exposes it as
            # "seed_keywords" so the model does not confuse it with the metrics tool.
            field_map={"seed_keywords": "keywords"},
        ),
    ]
}


def get_spec(name: str) -> ToolSpec:
    try:
        return TOOLS[name]
    except KeyError:
        raise ToolValidationError(
            f"unknown tool '{name}'",
            ErrorContext(tool=name, detail={"known_tools": sorted(TOOLS)}),
        ) from None


def openai_tool_definitions() -> list[dict[str, Any]]:
    """JSON-schema tool definitions in the shape every tool-calling provider accepts."""
    defs = []
    for spec in TOOLS.values():
        schema = spec.args_model.model_json_schema()
        schema.pop("title", None)
        defs.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": schema,
                },
            }
        )
    return defs


# The planner sometimes emits a plausible synonym instead of the declared field name.
# Rewriting a handful of known aliases is cheaper and more predictable than a second
# LLM round-trip, and anything not covered here still fails closed below.
_ALIASES: dict[str, dict[str, str]] = {
    "serp_organic_results": {"query": "keyword", "search_query": "keyword", "q": "keyword"},
    "ai_overview_snapshot": {"query": "keyword", "prompt": "keyword"},
    "llm_answer_visibility": {
        "prompt": "user_prompt",
        "query": "user_prompt",
        "model": "model_name",
    },
    "keyword_search_volume": {"keyword": "keywords", "terms": "keywords"},
    "related_keyword_ideas": {
        "seeds": "seed_keywords",
        "keywords": "seed_keywords",
        "seed": "seed_keywords",
    },
}

_LIST_FIELDS = {"keywords", "seed_keywords"}


def _coerce(name: str, raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    aliases = _ALIASES.get(name, {})
    for key, value in raw.items():
        out[aliases.get(key, key)] = value
    for name in _LIST_FIELDS & out.keys():
        if isinstance(out[name], str):
            out[name] = [part.strip() for part in out[name].split(",") if part.strip()]
    return out


@dataclass(frozen=True)
class ValidatedToolCall:
    name: str
    spec: ToolSpec
    args: BaseModel

    @property
    def payload(self) -> dict[str, Any]:
        return self.args.model_dump(mode="json")


def validate_tool_call(name: str, raw_args: dict[str, Any] | None) -> ValidatedToolCall:
    """Gate between the model's proposed call and the real HTTP request.

    Nothing reaches DataForSEO until it has been through the tool's own Pydantic
    model, so a hallucinated field or a missing required one costs us a rejection
    record instead of a billed request or a stack trace.
    """
    spec = get_spec(name)
    coerced = _coerce(name, raw_args or {})
    try:
        args = spec.args_model.model_validate(coerced)
    except ValidationError as exc:
        problems = [
            {
                "field": ".".join(str(p) for p in err["loc"]) or "<root>",
                "problem": err["msg"],
                "type": err["type"],
            }
            for err in exc.errors()
        ]
        raise ToolValidationError(
            f"invalid arguments for tool '{name}'",
            ErrorContext(tool=name, endpoint=spec.endpoint, detail={"problems": problems}),
        ) from exc
    return ValidatedToolCall(name=name, spec=spec, args=args)
