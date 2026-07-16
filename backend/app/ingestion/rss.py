from collections.abc import Iterable
from datetime import datetime, timezone
from time import mktime

import feedparser
from sqlalchemy.orm import Session

from app.ingestion.base import Source
from app.schemas import ArticleCreate


def _parsed_to_datetime(entry) -> datetime | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    return datetime.fromtimestamp(mktime(parsed), tz=timezone.utc)


class RSSSource(Source):
    def __init__(self, name: str, feed_url: str):
        self.name = name
        self.feed_url = feed_url

    def fetch(self) -> Iterable[ArticleCreate]:
        parsed = feedparser.parse(self.feed_url)
        if parsed.bozo and not parsed.entries:
            raise ValueError(str(parsed.get("bozo_exception", "failed to parse feed")))

        for entry in parsed.entries:
            if not entry.get("link") or not entry.get("title"):
                continue
            yield ArticleCreate(
                title=entry.title,
                url=entry.link,
                source=self.name,
                summary=entry.get("summary"),
                published_at=_parsed_to_datetime(entry),
            )


def ingest_feed(db: Session, source: str, feed_url: str) -> int:
    """Fetch and upsert a single RSS feed. Returns count added.

    Thin compatibility wrapper around RSSSource + the generic runner.
    """
    from app.ingestion.runner import ingest_source

    return ingest_source(db, RSSSource(source, feed_url))
