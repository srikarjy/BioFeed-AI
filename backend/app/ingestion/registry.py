"""Configured content sources for the ingestion pipeline.

This is the single place that declares what BioFeed pulls from. The runner is
source-agnostic, so adding a feed or a new source type is an edit here only.
"""

from app.ingestion.base import Source
from app.ingestion.feeds import FEEDS
from app.ingestion.rss import RSSSource


def get_sources() -> list[Source]:
    sources: list[Source] = [RSSSource(name, url) for name, url in FEEDS]
    return sources
