"""Static list of RSS feed sources for v0.1 ingestion.

v0.2 will replace this with a configurable/DB-backed source list and add
scheduling; for now this is a hardcoded starter set of verified biotech
news feeds.
"""

FEEDS: list[tuple[str, str]] = [
    ("FiercePharma", "https://www.fiercepharma.com/rss/xml"),
    ("STAT News", "https://www.statnews.com/feed/"),
    ("GEN", "https://www.genengnews.com/feed/"),
    ("BioPharma Dive", "https://www.biopharmadive.com/feeds/news/"),
]
