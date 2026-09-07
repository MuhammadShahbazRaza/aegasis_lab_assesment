import pytest

from app.clients.dataforseo import MockDataForSEOClient
from app.resilience.errors import ToolValidationError
from app.tools.executor import ToolExecutor
from app.tools.registry import openai_tool_definitions, validate_tool_call
from app.tools.schemas import SerpOrganicArgs


def test_valid_call_applies_declared_defaults():
    call = validate_tool_call("serp_organic_results", {"keyword": "best crm"})
    assert isinstance(call.args, SerpOrganicArgs)
    assert call.payload == {
        "keyword": "best crm",
        "location_name": "United States",
        "language_code": "en",
        "depth": 20,
    }


def test_missing_required_field_is_rejected_with_field_level_detail():
    with pytest.raises(ToolValidationError) as excinfo:
        validate_tool_call("serp_organic_results", {"location_name": "United States"})
    problems = excinfo.value.context.detail["problems"]
    assert [p["field"] for p in problems] == ["keyword"]
    assert excinfo.value.retryable is False


def test_out_of_range_value_is_rejected():
    with pytest.raises(ToolValidationError):
        validate_tool_call("serp_organic_results", {"keyword": "best crm", "depth": 5000})


def test_hallucinated_field_is_rejected_rather_than_silently_dropped():
    with pytest.raises(ToolValidationError) as excinfo:
        validate_tool_call(
            "serp_organic_results", {"keyword": "best crm", "include_paid_ads": True}
        )
    assert "include_paid_ads" in str(excinfo.value.context.detail)


def test_unknown_tool_name_is_rejected():
    with pytest.raises(ToolValidationError):
        validate_tool_call("call_dataforseo", {"anything": 1})


@pytest.mark.parametrize(
    ("tool", "raw", "field", "expected"),
    [
        ("serp_organic_results", {"query": "best crm"}, "keyword", "best crm"),
        (
            "llm_answer_visibility",
            {"prompt": "which crm is best"},
            "user_prompt",
            "which crm is best",
        ),
        ("keyword_search_volume", {"keywords": "a crm, b crm"}, "keywords", ["a crm", "b crm"]),
    ],
)
def test_known_aliases_are_repaired_before_validation(tool, raw, field, expected):
    call = validate_tool_call(tool, raw)
    assert call.payload[field] == expected


def test_duplicate_keywords_are_collapsed():
    call = validate_tool_call(
        "keyword_search_volume", {"keywords": ["Best CRM", "best crm", "crm pricing"]}
    )
    assert call.payload["keywords"] == ["best crm", "crm pricing"]


def test_tool_definitions_expose_types_and_required_fields():
    definitions = {d["function"]["name"]: d["function"] for d in openai_tool_definitions()}
    serp = definitions["serp_organic_results"]["parameters"]
    assert serp["required"] == ["keyword"]
    assert serp["properties"]["depth"]["type"] == "integer"
    assert serp["additionalProperties"] is False
    assert all(d["description"] for d in definitions.values())


def test_executor_returns_failed_invocation_instead_of_raising(settings, metrics):
    executor = ToolExecutor(MockDataForSEOClient(settings), settings, metrics)
    invocation = executor.invoke("serp_organic_results", {"depth": 20})

    assert invocation.ok is False
    assert invocation.raw is None
    assert invocation.error["type"] == "ToolValidationError"
    # A rejected call must never reach the provider.
    assert metrics.snapshot()["api_failures"]["serp_organic_results"] == 1


def test_provider_field_names_are_applied_only_where_they_differ():
    ideas = validate_tool_call("related_keyword_ideas", {"seed_keywords": ["seo software"]})
    wire = ideas.spec.to_payload(ideas.payload)
    # DataForSEO Labs calls this field "keywords"; the tool exposes "seed_keywords".
    assert "keywords" in wire and "seed_keywords" not in wire
    assert wire["keywords"] == ["seo software"]

    serp = validate_tool_call("serp_organic_results", {"keyword": "best crm"})
    assert serp.spec.to_payload(serp.payload) == serp.payload


def test_no_internal_keys_reach_the_wire_payload():
    from app.tools.executor import _request_body

    for name, args in (
        ("serp_organic_results", {"keyword": "best crm"}),
        ("related_keyword_ideas", {"seed_keywords": ["crm"]}),
    ):
        call = validate_tool_call(name, args)
        body = _request_body(call)[0]
        assert not [k for k in body if k.startswith("_")]
        assert set(body) <= set(call.spec.args_model.model_fields) | set(
            call.spec.field_map.values()
        )
