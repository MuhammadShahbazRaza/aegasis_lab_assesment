from collections.abc import Iterator

from sqlalchemy.orm import Session

from app.db.base import session_factory


def get_session() -> Iterator[Session]:
    session = session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
