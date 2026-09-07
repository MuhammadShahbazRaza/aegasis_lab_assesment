import operator
from typing import Annotated, Any, TypedDict

from app.domain import (
    FinalReport,
    NormalizedRecord,
    PlannedCall,
    ProfileContext,
    QueryInsight,
    RunStatus,
)
from app.tools.executor import ToolInvocation


def _merge_flags(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {**left, **right}


class PipelineState(TypedDict, total=False):
    run_id: str
    profile: ProfileContext
    question: str

    planned_calls: list[PlannedCall]
    plan_notes: list[str]
    rejected_tool_calls: list[dict[str, Any]]
    plan_source: str

    # Retrieval fans out across planned calls, so these two need reducers: parallel
    # branches each return their own slice and LangGraph concatenates them.
    invocations: Annotated[list[ToolInvocation], operator.add]
    retrieval_errors: Annotated[list[dict[str, Any]], operator.add]

    records: list[NormalizedRecord]
    insights: list[QueryInsight]
    report: FinalReport | None

    status: RunStatus
    degraded: bool
    flags: Annotated[dict[str, Any], _merge_flags]
    node_path: Annotated[list[str], operator.add]


def initial_state(run_id: str, profile: ProfileContext, question: str) -> PipelineState:
    return PipelineState(
        run_id=run_id,
        profile=profile,
        question=question,
        planned_calls=[],
        plan_notes=[],
        rejected_tool_calls=[],
        plan_source="",
        invocations=[],
        retrieval_errors=[],
        records=[],
        insights=[],
        report=None,
        status=RunStatus.COMPLETED,
        degraded=False,
        flags={},
        node_path=[],
    )
