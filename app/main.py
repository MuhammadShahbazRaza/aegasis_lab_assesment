from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.errors import register_handlers
from app.api.routes import profiles, queries, runs
from app.config import get_settings
from app.db.base import init_db
from app.graph.builder import mermaid_diagram
from app.observability.logging import configure_logging, get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    init_db()
    if settings.llm_provider != "fake" and not settings.api_key_for_provider():
        log.warning(
            "no API key for the configured LLM provider; runs will degrade to the "
            "heuristic planner. Set the key in .env or use LLM_PROVIDER=fake.",
            extra={"llm_provider": settings.llm_provider},
        )
    log.info(
        "service ready",
        extra={
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "dataforseo_mode": settings.dataforseo_mode,
            "database_url": settings.database_url,
        },
    )
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Agentic Search Intelligence",
        version="1.0.0",
        description=(
            "A LangGraph DAG of single-responsibility agents that plans retrieval, "
            "calls DataForSEO, normalizes the results and reports on AI/search visibility."
        ),
        lifespan=lifespan,
    )
    register_handlers(app)
    app.include_router(profiles.router)
    app.include_router(runs.router)
    app.include_router(queries.router)

    @app.get("/", tags=["ops"])
    def index() -> dict[str, object]:
        """Landing page for anyone who opens the host in a browser: the endpoint list
        is more useful there than a bare 404."""
        return {
            "service": app.title,
            "version": app.version,
            "docs": "/docs",
            "graph": "/api/v1/graph",
            "health": "/healthz",
            "endpoints": [
                "POST   /api/v1/profiles",
                "GET    /api/v1/profiles/{profile_uuid}",
                "POST   /api/v1/profiles/{profile_uuid}/run",
                "GET    /api/v1/profiles/{profile_uuid}/queries",
                "GET    /api/v1/profiles/{profile_uuid}/recommendations",
                "POST   /api/v1/queries/{query_uuid}/recheck",
                "GET    /api/v1/runs/{run_uuid}",
            ],
        }

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict[str, object]:
        settings = get_settings()
        return {
            "status": "ok",
            "llm_provider": settings.llm_provider,
            "dataforseo_mode": settings.dataforseo_mode,
        }

    @app.get("/api/v1/graph", tags=["ops"])
    def graph_definition() -> dict[str, str]:
        return {"format": "mermaid", "diagram": mermaid_diagram()}

    return app


app = create_app()
