from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _connect_args(url: str) -> dict[str, object]:
    # SQLite defaults to one-thread-per-connection; FastAPI's threadpool hands the
    # same session factory to different worker threads.
    return {"check_same_thread": False} if url.startswith("sqlite") else {}


def get_engine() -> Engine:
    global _engine, _session_factory
    if _engine is None:
        url = get_settings().database_url
        _engine = create_engine(url, connect_args=_connect_args(url), future=True)
        if url.startswith("sqlite"):
            event.listen(_engine, "connect", _enable_sqlite_pragmas)
        _session_factory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


def _enable_sqlite_pragmas(dbapi_connection, _record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


def init_db() -> None:
    from app.db import models  # noqa: F401  -- registers mappers before create_all

    Base.metadata.create_all(bind=get_engine())


def session_factory() -> sessionmaker[Session]:
    get_engine()
    assert _session_factory is not None
    return _session_factory


@contextmanager
def session_scope() -> Iterator[Session]:
    session = session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
