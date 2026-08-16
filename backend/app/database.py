from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings

if settings.database_url.startswith("sqlite"):
    # A bare SQLite connection can't be shared across threads, and FastAPI
    # runs sync path operations in a threadpool -- without this, a real
    # `uvicorn` process (not the single-connection StaticPool tests/conftest.py
    # sets up manually) booted against a sqlite:// DATABASE_URL fails with
    # "SQLite objects created in a thread can only be used in that same
    # thread." Production always uses Postgres; this only matters for
    # scripts/manual runs that point DATABASE_URL at sqlite outside pytest
    # (e.g. scripts/seed_v07_synthetic.py, or a lightweight one-off
    # verification run with no Postgres available).
    engine = create_engine(
        settings.database_url, connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
else:
    engine = create_engine(settings.database_url)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
