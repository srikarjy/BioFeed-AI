from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app import crud
from app.ingestion.base import Source
from app.ml import service as ml_service
from app.models import IngestionRun


def ingest_source(db: Session, source: Source) -> int:
    """Fetch a single source and upsert its articles. Returns count added."""
    added = 0
    for item in source.fetch():
        _, created = crud.create_article(db, item)
        if created:
            added += 1
    return added


def run_ingestion(
    db: Session, sources: list[Source] | None = None
) -> tuple[dict[str, int], dict[str, str]]:
    """Run ingestion across the given sources (defaults to the registry).

    Each source is isolated so one dead/unreachable source doesn't abort the
    rest of the run.
    """
    if sources is None:
        from app.ingestion.registry import get_sources

        sources = get_sources()

    added: dict[str, int] = {}
    errors: dict[str, str] = {}
    for source in sources:
        try:
            added[source.name] = ingest_source(db, source)
        except Exception as exc:  # noqa: BLE001 - isolate per-source failures
            errors[source.name] = str(exc)
    return added, errors


def run_and_record(
    db: Session, sources: list[Source] | None = None
) -> IngestionRun:
    """Run ingestion, embed new articles, and persist an IngestionRun row."""
    started_at = datetime.now(timezone.utc)
    added, errors = run_ingestion(db, sources)
    # Same isolation rule as sources: an embedder failure (model download,
    # OOM) must not lose the run record; embed_missing will backfill next run.
    embedded = 0
    try:
        embedded = ml_service.embed_missing(db)
    except Exception as exc:  # noqa: BLE001 - isolate embedder failures
        errors["_embedding"] = str(exc)
    return crud.record_ingestion_run(db, started_at, added, errors, embedded=embedded)