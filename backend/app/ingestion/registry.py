"""Configured content sources for the ingestion pipeline.

This is the single place that declares what BioFeed pulls from. The runner is
source-agnostic, so adding a feed or a new source type is an edit here only.
"""

import os

from app.ingestion.base import Source
from app.ingestion.feeds import FEEDS
from app.ingestion.rss import RSSSource
from app.ingestion.pubmed import PubMedSource
from app.ingestion.biorxiv import BioRxivSource


def get_sources() -> list[Source]:
    sources: list[Source] = [RSSSource(name, url) for name, url in FEEDS]
    
    # PubMed - enable if NCBI_API_KEY is set (for rate limits)
    if os.getenv("NCBI_API_KEY"):
        sources.append(PubMedSource(
            max_results=int(os.getenv("PUBMED_MAX_RESULTS", "100")),
            api_key=os.getenv("NCBI_API_KEY"),
            email=os.getenv("NCBI_EMAIL"),
        ))
    else:
        # Still add but with lower rate limit (no API key = 3 req/s)
        sources.append(PubMedSource(
            max_results=int(os.getenv("PUBMED_MAX_RESULTS", "50")),
            email=os.getenv("NCBI_EMAIL"),
        ))
    
    # bioRxiv/medRxiv
    sources.append(BioRxivSource(
        max_results=int(os.getenv("BIORXIV_MAX_RESULTS", "100")),
        days_back=int(os.getenv("BIORXIV_DAYS_BACK", "7")),
        server="biorxiv",
    ))
    sources.append(BioRxivSource(
        max_results=int(os.getenv("MEDRXIV_MAX_RESULTS", "50")),
        days_back=int(os.getenv("MEDRXIV_DAYS_BACK", "7")),
        server="medrxiv",
    ))
    
    return sources
