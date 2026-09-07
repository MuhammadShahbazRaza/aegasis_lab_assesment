from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from app.clients.dataforseo import DataForSEOClient
from app.config import Settings
from app.observability.metrics import RunMetrics
from app.tools.executor import ToolExecutor


@dataclass
class GraphRuntime:
    """Everything a node needs that is not part of the graph state.

    Bound once when the graph is compiled rather than threaded through the state, so
    node signatures stay `(state) -> partial state` and the state stays serializable.
    """

    settings: Settings
    llm: BaseChatModel
    executor: ToolExecutor
    metrics: RunMetrics
    client: DataForSEOClient | None = None

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
