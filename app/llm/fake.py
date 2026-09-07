import json
import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

_ROLES = ("planner", "analyst", "reporter")


def _text_of(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return " ".join(str(part) for part in content)


def _field(text: str, label: str) -> str:
    match = re.search(rf"^{label}:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _plan_from_prompt(text: str) -> list[dict[str, Any]]:
    """Mirror the shape of a competent planner's output so the offline path exercises
    the same validation, fan-out and extraction edges a live provider would."""
    name = _field(text, "Brand") or "the brand"
    industry = (_field(text, "Industry") or name).lower()
    if industry in ("unspecified", ""):
        industry = name.lower()

    seeds = [f"best {industry}", f"{name} alternatives".lower(), f"{industry} pricing"]
    calls: list[dict[str, Any]] = [
        {
            "name": "keyword_search_volume",
            "args": {"keywords": seeds, "location_name": "United States", "language_code": "en"},
            "id": "call_volume",
        }
    ]
    for index, seed in enumerate(seeds[:2]):
        calls.append(
            {
                "name": "serp_organic_results",
                "args": {"keyword": seed, "location_name": "United States", "depth": 20},
                "id": f"call_serp_{index}",
            }
        )
        calls.append(
            {
                "name": "ai_overview_snapshot",
                "args": {"keyword": seed, "location_name": "United States"},
                "id": f"call_ai_{index}",
            }
        )
    calls.append(
        {
            "name": "llm_answer_visibility",
            "args": {
                "user_prompt": f"Which {industry} should I use and why?",
                "model_name": "gpt-4o-mini",
                "web_search": True,
            },
            "id": "call_llm",
        }
    )
    return calls


def _analysis_from_payload(text: str) -> dict[str, Any]:
    data = _load_json(text)
    brand = data.get("brand", "the brand")
    domain = data.get("domain", "")
    queries = data.get("queries", [])
    gaps = [q for q in queries if q.get("visibility_status") != "visible"][:3]

    findings = [
        f"{domain} does not appear for '{q['query_text']}' "
        f"({q.get('search_volume', 0):,} monthly searches, difficulty "
        f"{q.get('competitive_difficulty')}/100)"
        + (
            f"; {', '.join(q.get('competitors_present', [])[:3])} do."
            if q.get("competitors_present")
            else "."
        )
        for q in gaps
    ]
    wins = [q for q in queries if q.get("visibility_status") == "visible"]
    if wins:
        findings.append(
            f"{brand} already holds visibility on {len(wins)} of {len(queries)} tracked queries."
        )
    ai_gaps = [q for q in gaps if "ai_overview" in q.get("surfaces_checked", [])]
    if ai_gaps:
        findings.append(
            f"{len(ai_gaps)} gap(s) sit on queries that trigger an AI Overview, where "
            "an uncited brand is invisible regardless of organic rank."
        )
    return {
        "findings": findings,
        "query_notes": [
            {
                "query_text": q["query_text"],
                "evidence": (
                    f"status={q.get('visibility_status')} "
                    f"position={q.get('visibility_position')} "
                    f"volume={q.get('search_volume')} "
                    f"difficulty={q.get('competitive_difficulty')}"
                ),
            }
            for q in queries
        ],
    }


def _report_from_payload(text: str) -> dict[str, Any]:
    data = _load_json(text)
    brand = data.get("brand", "the brand")
    domain = data.get("domain", "")
    queries = data.get("queries", [])
    visible = data.get("queries_visible", 0)
    findings = data.get("findings", [])

    recommendations = []
    for q in queries[:4]:
        if q.get("visibility_status") == "visible":
            continue
        recommendations.append(
            {
                "target_query_uuid": q.get("target_query_uuid"),
                "content_type": "comparison_page" if q.get("competitors_present") else "blog_post",
                "title": f"{q['query_text'].title()} - {brand} buyer's guide",
                "rationale": (
                    f"Opportunity score {q.get('opportunity_score')} with no owned "
                    f"placement; competitors present: "
                    f"{', '.join(q.get('competitors_present', [])) or 'none tracked'}."
                ),
                "target_keywords": [q["query_text"]],
                "priority": "high" if (q.get("opportunity_score") or 0) >= 0.66 else "medium",
            }
        )

    body = "\n".join(f"- {f}" for f in findings)
    return {
        "headline": f"{brand} is visible on {visible} of {len(queries)} tracked queries",
        "summary_markdown": (
            f"## {brand} - AI and search visibility\n\n"
            f"`{domain}` appears in {visible} of the {len(queries)} queries analysed "
            "across organic results, Google AI Overviews and assistant answers.\n\n"
            f"{body}\n\n"
            "Priority is the set of high-volume queries where an AI surface answers the "
            "question and the brand is not among the cited sources - those are lost "
            "before a click is ever available."
        ),
        "recommendations": recommendations,
    }


def _load_json(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


class ScriptedChatModel(BaseChatModel):
    """Deterministic stand-in for a tool-calling provider.

    It exists so the DAG, the tool-argument validation path and the fallback edges
    stay testable without a network or an API key. It reads the role marker in the
    system prompt and answers in exactly the shape a real provider returns - tool
    calls for the planner, JSON content for the analyst and reporter. Tests override
    `responses` to force malformed arguments or empty plans.
    """

    responses: dict[str, Any] = {}
    calls: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _role_of(self, messages: list[BaseMessage]) -> str:
        joined = " ".join(_text_of(m) for m in messages)
        for role in _ROLES:
            if f"[role:{role}]" in joined:
                return role
        return "unknown"

    def _default_for(self, role: str, messages: list[BaseMessage]) -> Any:
        last = _text_of(messages[-1]) if messages else ""
        if role == "planner":
            return _plan_from_prompt(last)
        if role == "analyst":
            return _analysis_from_payload(last)
        if role == "reporter":
            return _report_from_payload(last)
        return ""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(messages)
        role = self._role_of(messages)
        scripted = self.responses.get(role, _MISSING)
        if scripted is _MISSING:
            scripted = self._default_for(role, messages)
        if callable(scripted):
            scripted = scripted(messages)

        if isinstance(scripted, AIMessage):
            message = scripted
        elif isinstance(scripted, list):
            message = AIMessage(content="", tool_calls=scripted)
        elif isinstance(scripted, dict):
            message = AIMessage(content=json.dumps(scripted))
        else:
            message = AIMessage(content=str(scripted or ""))

        if not message.usage_metadata:
            message.usage_metadata = {
                "input_tokens": 140,
                "output_tokens": 95,
                "total_tokens": 235,
            }
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self


class _Missing:
    pass


_MISSING = _Missing()
