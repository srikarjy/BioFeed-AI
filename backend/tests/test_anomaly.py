"""Cross-source burst detector: run against the SQLite fixture with
EMBEDDING_BACKEND=hash, so similarity is token-overlap rather than true
semantics -- enough to assert the detection wiring and idempotency.
"""

from datetime import datetime, timedelta, timezone

from app import crud
from app.anomaly import crud as anomaly_crud
from app.anomaly.detector import detect_recent
from app.ml import service as ml_service
from app.schemas import ArticleCreate

NOW = datetime(2026, 7, 28, tzinfo=timezone.utc)


def _make(db, title, source, published_at, summary=""):
    article, _ = crud.create_article(
        db,
        ArticleCreate(
            title=title,
            url=f"http://x/{source}/{title}",
            source=source,
            summary=summary,
            published_at=published_at,
        ),
    )
    return article


def test_detects_cross_source_burst_within_window(db_session):
    # Titles differ slightly per outlet (so cross-source dedup doesn't collapse
    # them into one row) but share enough vocabulary for the hash embedder to
    # place them near each other.
    _make(db_session, "FDA approves gene therapy for sickle cell disease", "FiercePharma", NOW)
    _make(db_session, "FDA approves gene therapy for sickle cell disease patients", "STAT News", NOW - timedelta(hours=2))
    _make(db_session, "FDA gene therapy approval for sickle cell disease", "GEN", NOW - timedelta(hours=10))
    ml_service.embed_missing(db_session)

    events = detect_recent(db_session)

    assert len(events) == 3  # each near-identical article corroborates the other two
    event = events[0]
    assert event.kind == "cross_source_burst"
    assert len(event.detail["related_sources"]) >= 2


def test_no_event_for_single_source_coverage(db_session):
    _make(db_session, "Quarterly biotech venture funding report", "FiercePharma", NOW)
    ml_service.embed_missing(db_session)

    assert detect_recent(db_session) == []


def test_no_event_outside_time_window(db_session):
    _make(db_session, "Novel antibody therapy for lymphoma", "FiercePharma", NOW)
    _make(db_session, "Novel antibody therapy for lymphoma patients", "STAT News", NOW - timedelta(days=10))
    ml_service.embed_missing(db_session)

    assert detect_recent(db_session) == []


def test_detection_is_idempotent(db_session):
    _make(db_session, "mRNA vaccine platform shows promise in trial", "FiercePharma", NOW)
    _make(db_session, "mRNA vaccine platform shows promise in trial results", "STAT News", NOW)
    _make(db_session, "mRNA vaccine platform trial shows promise", "GEN", NOW)
    ml_service.embed_missing(db_session)

    first_pass = detect_recent(db_session)
    second_pass = detect_recent(db_session)

    assert [e.id for e in first_pass] == [e.id for e in second_pass]
    assert len(anomaly_crud.list_events(db_session, limit=100)) == len(first_pass)
