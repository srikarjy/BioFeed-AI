import re
from collections.abc import Iterable
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from time import mktime

import feedparser
from sqlalchemy.orm import Session

from app.ingestion.base import Source
from app.schemas import ArticleCreate

_WHITESPACE = re.compile(r"\s+")


class _TextExtractor(HTMLParser):
    """Collects the text nodes of an HTML fragment, dropping tags.

    RSS feeds routinely wrap titles and summaries in markup (`<a href=...>`,
    `<p>`, `<b>`); left in place, the tag names and URLs become embedding
    tokens and dilute the vector. Using the stdlib parser rather than a regex
    means malformed or partial markup degrades to plain text instead of
    leaking angle brackets.
    """

    def __init__(self):
        super().__init__()
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        self._chunks.append(data)

    def get_text(self) -> str:
        return "".join(self._chunks)


def _clean_text(value: str | None) -> str | None:
    """Strip HTML tags/entities and collapse whitespace. None passes through."""
    if value is None:
        return None
    parser = _TextExtractor()
    parser.feed(value)
    text = unescape(parser.get_text())
    return _WHITESPACE.sub(" ", text).strip() or None


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
            title = _clean_text(entry.get("title"))
            if not entry.get("link") or not title:
                continue
            yield ArticleCreate(
                title=title,
                url=entry.link,
                source=self.name,
                summary=_clean_text(entry.get("summary")),
                published_at=_parsed_to_datetime(entry),
            )


def ingest_feed(db: Session, source: str, feed_url: str) -> int:
    """Fetch and upsert a single RSS feed. Returns count added.

    Thin compatibility wrapper around RSSSource + the generic runner.
    """
    from app.ingestion.runner import ingest_source

    return ingest_source(db, RSSSource(source, feed_url))
