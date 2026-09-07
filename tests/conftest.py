import os
from collections.abc import Iterator

import pytest

os.environ.update(
    {
        "LLM_PROVIDER": "fake",
        "DATAFORSEO_MODE": "mock",
        "LOG_LEVEL": "WARNING",
        "LOG_FORMAT": "console",
        "RETRY_BASE_DELAY": "0.001",
        "RETRY_MAX_DELAY": "0.01",
        "MOCK_FAILURE_RATE": "0",
        "MOCK_ALWAYS_FAIL_TOOLS": "",
    }
)

from app.config import Settings, reload_settings  # noqa: E402
from app.db.base import init_db, reset_engine, session_factory  # noqa: E402
from app.db.repository import ProfileRepository  # noqa: E402
from app.observability.metrics import RunMetrics  # noqa: E402

PROFILE = {
    "name": "Surfer SEO",
    "domain": "surferseo.com",
    "industry": "SEO Software",
    "description": "AI-powered SEO content optimization tool",
    "competitors": ["clearscope.io", "marketmuse.com", "frase.io"],
}


@pytest.fixture
def settings(tmp_path, monkeypatch) -> Settings:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    reset_engine()
    resolved = reload_settings()
    init_db()
    yield resolved
    reset_engine()
    reload_settings()


@pytest.fixture
def session(settings) -> Iterator:
    factory = session_factory()
    db = factory()
    try:
        yield db
        db.commit()
    finally:
        db.close()


@pytest.fixture
def profile(session):
    created = ProfileRepository(session).create(**PROFILE)
    session.commit()
    return created


@pytest.fixture
def profile_context(session, profile):
    return ProfileRepository(session).context(profile)


@pytest.fixture
def metrics() -> RunMetrics:
    return RunMetrics("test-run")


@pytest.fixture
def client(settings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client
