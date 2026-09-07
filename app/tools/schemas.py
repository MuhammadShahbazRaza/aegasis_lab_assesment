from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Keyword = Annotated[str, Field(min_length=2, max_length=200)]
LanguageCode = Annotated[str, Field(min_length=2, max_length=5)]

_LOCATION = "Full DataForSEO location name, e.g. 'United States' or 'London,England,United Kingdom'"
_LANGUAGE = "ISO language code for the search, e.g. 'en'"


class ToolArgs(BaseModel):
    # extra="forbid" is what turns a hallucinated argument into a caught rejection
    # rather than a silently ignored field.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class _Localized(ToolArgs):
    location_name: str = Field(default="United States", description=_LOCATION)
    language_code: LanguageCode = Field(default="en", description=_LANGUAGE)

    @field_validator("language_code")
    @classmethod
    def _lower(cls, value: str) -> str:
        return value.lower()


class SerpOrganicArgs(_Localized):
    keyword: Keyword = Field(
        description="Exact search query to submit to Google, "
        "e.g. 'best project management software'"
    )
    depth: int = Field(
        default=20,
        ge=10,
        le=100,
        description="How many organic positions to retrieve; 20 is enough to judge "
        "first-page visibility",
    )


class AiOverviewArgs(_Localized):
    keyword: Keyword = Field(
        description="Search query to check for a Google AI Overview / AI Mode answer"
    )


class LlmVisibilityArgs(ToolArgs):
    user_prompt: Annotated[str, Field(min_length=5, max_length=500)] = Field(
        description="Natural-language prompt to send to the LLM, phrased the way a buyer "
        "would actually ask it"
    )
    model_name: Literal["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini"] = Field(
        default="gpt-4o-mini", description="Which model's answer to sample"
    )
    web_search: bool = Field(
        default=True,
        description="Whether the sampled model should browse; browsing answers cite "
        "sources and are what brand visibility is measured against",
    )


class SearchVolumeArgs(_Localized):
    keywords: Annotated[list[Keyword], Field(min_length=1, max_length=20)] = Field(
        description="Keywords to look up monthly search volume and competition for. "
        "Send them in one call rather than one call per keyword"
    )

    @field_validator("keywords")
    @classmethod
    def _dedupe(cls, value: list[str]) -> list[str]:
        seen: dict[str, None] = {}
        for keyword in value:
            seen.setdefault(keyword.strip().lower(), None)
        return list(seen)


class KeywordIdeasArgs(_Localized):
    seed_keywords: Annotated[list[Keyword], Field(min_length=1, max_length=5)] = Field(
        description="Seed keywords to expand into related query ideas"
    )
    limit: int = Field(
        default=15, ge=1, le=100, description="Maximum number of related keyword ideas to return"
    )
