from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import settings


@lru_cache
def get_engine():
    return create_engine(settings.neon_database_url, pool_pre_ping=True)


@contextmanager
def get_session() -> Session:
    session = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    from src.db.models import Base

    Base.metadata.create_all(bind=get_engine())