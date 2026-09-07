from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.graph.instrument import instrumented
from app.graph.nodes.analysis import build_analysis
from app.graph.nodes.extraction import build_extraction
from app.graph.nodes.fallback import build_degraded_fallback, build_no_data_fallback
from app.graph.nodes.planner import build_plan_fallback, build_planner
from app.graph.nodes.report import build_report
from app.graph.nodes.retrieval import build_retrieval_gate, build_retrieval_worker
from app.graph.runtime import GraphRuntime
from app.graph.state import PipelineState

QUERY_PLANNER = "query_planner"
PLAN_FALLBACK = "plan_fallback"
RETRIEVAL_WORKER = "retrieval_worker"
RETRIEVAL_GATE = "retrieval_gate"
PARTIAL_DATA = "partial_data_fallback"
EXTRACTION = "extraction_normalizer"
ANALYSIS = "analysis_synthesizer"
NO_DATA = "no_data_fallback"
REPORT = "report_assembler"


def _dispatch(state: dict[str, Any]) -> list[Send] | str:
    """Turns the plan into N parallel retrieval branches.

    Send is what makes this a real DAG rather than a for-loop hidden inside one node:
    every planned call becomes its own node execution with its own timing, retry count
    and failure record, and LangGraph joins them all at the gate.
    """
    calls = state.get("planned_calls") or []
    if not calls:
        return NO_DATA
    return [
        Send(
            RETRIEVAL_WORKER,
            {"call": call, "profile": state["profile"], "run_id": state.get("run_id")},
        )
        for call in calls
    ]


def _route_after_plan(state: dict[str, Any]) -> list[Send] | str:
    # An empty or fully-rejected plan is not a crash: it routes to the deterministic
    # planner and the run continues with reduced ambition.
    if not state.get("planned_calls"):
        return PLAN_FALLBACK
    return _dispatch(state)


def _route_after_retrieval(state: dict[str, Any], threshold: float) -> str:
    planned = state.get("planned_calls") or []
    succeeded = [i for i in state.get("invocations", []) if i.ok]
    if not planned or not succeeded:
        return NO_DATA
    return EXTRACTION if len(succeeded) / len(planned) >= threshold else PARTIAL_DATA


def _route_after_extraction(state: dict[str, Any]) -> str:
    return ANALYSIS if state.get("records") else NO_DATA


def build_graph(runtime: GraphRuntime, sink: Callable[..., None] | None = None):
    def node(name: str, fn: Callable[[dict[str, Any]], dict[str, Any]]):
        return instrumented(name, runtime.metrics, sink)(fn)

    graph = StateGraph(PipelineState)
    graph.add_node(QUERY_PLANNER, node(QUERY_PLANNER, build_planner(runtime)))
    graph.add_node(PLAN_FALLBACK, node(PLAN_FALLBACK, build_plan_fallback(runtime)))
    graph.add_node(RETRIEVAL_WORKER, node(RETRIEVAL_WORKER, build_retrieval_worker(runtime)))
    graph.add_node(RETRIEVAL_GATE, node(RETRIEVAL_GATE, build_retrieval_gate(runtime)))
    graph.add_node(PARTIAL_DATA, node(PARTIAL_DATA, build_degraded_fallback(runtime)))
    graph.add_node(EXTRACTION, node(EXTRACTION, build_extraction(runtime)))
    graph.add_node(ANALYSIS, node(ANALYSIS, build_analysis(runtime)))
    graph.add_node(NO_DATA, node(NO_DATA, build_no_data_fallback(runtime)))
    graph.add_node(REPORT, node(REPORT, build_report(runtime)))

    graph.add_edge(START, QUERY_PLANNER)
    graph.add_conditional_edges(
        QUERY_PLANNER, _route_after_plan, [PLAN_FALLBACK, RETRIEVAL_WORKER, NO_DATA]
    )
    graph.add_conditional_edges(PLAN_FALLBACK, _dispatch, [RETRIEVAL_WORKER, NO_DATA])
    graph.add_edge(RETRIEVAL_WORKER, RETRIEVAL_GATE)

    threshold = runtime.settings.retrieval_success_threshold
    graph.add_conditional_edges(
        RETRIEVAL_GATE,
        lambda state: _route_after_retrieval(state, threshold),
        [EXTRACTION, PARTIAL_DATA, NO_DATA],
    )
    # Partial data still goes through the normal parser. Discarding calls that did
    # succeed because their siblings failed would waste data already paid for.
    graph.add_edge(PARTIAL_DATA, EXTRACTION)
    graph.add_conditional_edges(EXTRACTION, _route_after_extraction, [ANALYSIS, NO_DATA])
    graph.add_edge(ANALYSIS, REPORT)
    graph.add_edge(NO_DATA, REPORT)
    graph.add_edge(REPORT, END)

    return graph.compile()


def build_recheck_graph(runtime: GraphRuntime, sink: Callable[..., None] | None = None):
    """Subgraph used by the recheck endpoint: the same node functions, re-entered at
    retrieval. Rebuilding it from the same builders is what keeps a recheck honest -
    a second, hand-rolled code path would drift from the full run within a sprint."""

    def node(name: str, fn: Callable[[dict[str, Any]], dict[str, Any]]):
        return instrumented(name, runtime.metrics, sink)(fn)

    graph = StateGraph(PipelineState)
    graph.add_node(RETRIEVAL_WORKER, node(RETRIEVAL_WORKER, build_retrieval_worker(runtime)))
    graph.add_node(RETRIEVAL_GATE, node(RETRIEVAL_GATE, build_retrieval_gate(runtime)))
    graph.add_node(PARTIAL_DATA, node(PARTIAL_DATA, build_degraded_fallback(runtime)))
    graph.add_node(EXTRACTION, node(EXTRACTION, build_extraction(runtime)))
    graph.add_node(ANALYSIS, node(ANALYSIS, build_analysis(runtime)))
    graph.add_node(NO_DATA, node(NO_DATA, build_no_data_fallback(runtime)))

    graph.add_conditional_edges(START, _dispatch, [RETRIEVAL_WORKER, NO_DATA])
    graph.add_edge(RETRIEVAL_WORKER, RETRIEVAL_GATE)

    threshold = runtime.settings.retrieval_success_threshold
    graph.add_conditional_edges(
        RETRIEVAL_GATE,
        lambda state: _route_after_retrieval(state, threshold),
        [EXTRACTION, PARTIAL_DATA, NO_DATA],
    )
    graph.add_edge(PARTIAL_DATA, EXTRACTION)
    graph.add_conditional_edges(EXTRACTION, _route_after_extraction, [ANALYSIS, NO_DATA])
    graph.add_edge(ANALYSIS, END)
    graph.add_edge(NO_DATA, END)

    return graph.compile()


def mermaid_diagram() -> str:
    return """flowchart TD
    START([START]) --> QP[query_planner]
    QP -->|plan empty or every call rejected| PF[plan_fallback]
    QP -->|Send x N planned calls| RW[retrieval_worker]
    PF -->|Send x N heuristic calls| RW
    RW --> RG{retrieval_gate}
    RG -->|success ratio >= threshold| EX[extraction_normalizer]
    RG -->|0 < ratio < threshold| PD[partial_data_fallback]
    RG -->|nothing succeeded| ND[no_data_fallback]
    PD --> EX
    EX -->|records > 0| AN[analysis_synthesizer]
    EX -->|no records parsed| ND
    AN --> RP[report_assembler]
    ND --> RP
    RP --> END([END])"""
