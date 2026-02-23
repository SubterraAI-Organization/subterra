from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


def get_database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    Path("data").mkdir(parents=True, exist_ok=True)
    return "sqlite:///data/subterra.sqlite3"


def _connect_args(url: str) -> dict:
    if url.startswith("sqlite:"):
        return {"check_same_thread": False}
    return {}


DATABASE_URL = get_database_url()
engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=_connect_args(DATABASE_URL))
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    from . import db_models  # noqa: F401

    Base.metadata.create_all(bind=engine)

    # Best-effort schema fix for early versions that used a reserved name.
    # Postgres only (docker-compose default). If the column was created as "metadata",
    # rename it to "meta" so ORM mapping remains valid.
    try:
        if not is_postgres():
            return
        from sqlalchemy import text
        from sqlalchemy import inspect

        insp = inspect(engine)
        cols = {c["name"] for c in insp.get_columns("annotations")}
        if "metadata" in cols and "meta" not in cols:
            with engine.begin() as conn:
                conn.execute(text('ALTER TABLE annotations RENAME COLUMN "metadata" TO meta'))
    except Exception:
        # Avoid blocking app startup on migration failures.
        pass


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def is_postgres() -> bool:
    return DATABASE_URL.startswith("postgresql")


def try_session() -> Optional[Session]:
    try:
        return SessionLocal()
    except Exception:
        return None
