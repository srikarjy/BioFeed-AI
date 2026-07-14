from datetime import datetime, timezone
from time import mktime

import feedparser
from sqlalchemy.orm import Session

from app import crud
from app.ingestion.feeds import FEEDS
from app.schemas import ArticleCreate


def _parsed_to_datetime(entry) -> datetime | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    return datetime.fromtimestamp(mktime(parsed), tz=timezone.utc)


def ingest_feed(db: Session, source: str, feed_url: str) -> int:
    """Fetch and parse a single feed, upserting articles. Returns count added."""
    parsed = feedparser.parse(feed_url)
    if parsed.bozo and not parsed.entries:
        raise ValueError(str(parsed.get("bozo_exception", "failed to parse feed")))

    added = 0
    for entry in parsed.entries:
        if not entry.get("link") or not entry.get("title"):
            continue
        article = ArticleCreate(
            title=entry.title,
            url=entry.link,
            source=source,
            summary=entry.get("summary"),
            published_at=_parsed_to_datetime(entry),
        )
        _, created = crud.create_article(db, article)
        if created:
            added += 1
    return added


def run_ingestion(db: Session) -> tuple[dict[str, int], dict[str, str]]:
    """Run ingestion across all configured feeds.

    Each feed is isolated so one dead/unreachable source doesn't abort the
    rest of the run.
    """
    added: dict[str, int] = {}
    errors: dict[str, str] = {}
    for source, feed_url in FEEDS:
        try:
            added[source] = ingest_feed(db, source, feed_url)
        except Exception as exc:  # noqa: BLE001 - isolate per-feed failures
            errors[source] = str(exc)
    return added, errors
