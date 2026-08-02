"""bioRxiv/medRxiv source using their API."""

import os
import re
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from html import unescape
from typing import Any
from urllib.parse import urlencode

import httpx

from app.ingestion.base import Source
from app.schemas import ArticleCreate


class BioRxivSource(Source):
    """Fetch preprints from bioRxiv/medRxiv via their API.
    
    API docs: https://api.biorxiv.org/
    No auth required, rate limit: 1 request/second.
    """

    name = "bioRxiv"
    BASE_URL = "https://api.biorxiv.org/details"
    
    # Subject areas relevant to biotech
    # Note: API returns categories with spaces, not hyphens
    DEFAULT_CATEGORIES = [
        "bioengineering",
        "cancer biology",
        "cell biology",
        "genomics",
        "immunology",
        "molecular biology",
        "neuroscience",
        "pharmacology and toxicology",
        "synthetic biology",
        "systems biology",
        "bioinformatics",
        "biophysics",
        "biochemistry",
        "evolutionary biology",
    ]

    def __init__(
        self,
        categories: list[str] | None = None,
        max_results: int = 100,
        days_back: int = 7,
        server: str = "biorxiv",  # or "medrxiv"
    ):
        self.categories = categories or self.DEFAULT_CATEGORIES
        self.max_results = max_results
        self.days_back = days_back
        self.server = server
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    def _build_url(self, cursor: int = 0) -> str:
        """Build API URL for a category and cursor."""
        # Date range: last N days
        from datetime import timedelta
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=self.days_back)
        date_range = f"{start_date}/{end_date}"
        # API format: /details/server/start_date/end_date/cursor
        return f"{self.BASE_URL}/{self.server}/{date_range}/{cursor}"

    def fetch(self) -> Iterable[ArticleCreate]:
        """Fetch preprints from bioRxiv/medRxiv. Synchronous wrapper."""
        import asyncio
        return asyncio.run(self._fetch_async())

    async def _fetch_async(self) -> list[ArticleCreate]:
        client = await self._get_client()
        all_articles = []
        
        # The API returns results in "collection" key, use cursor for pagination
        cursor = 0
        while len(all_articles) < self.max_results:
            url = self._build_url(cursor)
            try:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    break  # No more results
                raise
            
            # Articles are in "collection" key
            collection = data.get("collection", [])
            if not collection:
                break
            
            for msg in collection:
                # Filter by category if specified
                if self.categories and msg.get("category") not in self.categories:
                    continue
                article = self._parse_article(msg)
                if article:
                    all_articles.append(article)
                    if len(all_articles) >= self.max_results:
                        break
            
            # Next cursor - each page has up to 100 items
            cursor += len(collection)
            
            # If we got fewer than 100, we've reached the end
            if len(collection) < 100:
                break
            
            # Rate limit: 1 req/s
            await asyncio.sleep(1.0)
        
        return all_articles[:self.max_results]

    def _parse_article(self, msg: dict) -> ArticleCreate | None:
        """Parse a single article from bioRxiv API response."""
        from html import unescape
        import re
        
        # Required fields
        doi = msg.get("doi")
        if not doi:
            return None
        
        title = self._clean_text(msg.get("title", ""))
        if not title:
            return None
        
        # Abstract
        abstract = self._clean_text(msg.get("abstract", ""))
        
        # Authors
        authors_list = msg.get("authors", "")
        authors = authors_list.split("; ") if authors_list else None
        authors_str = "; ".join(authors) if authors else None
        
        # Date
        date_str = msg.get("date")
        pub_date = None
        if date_str:
            try:
                pub_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except ValueError:
                pass
        
        # Category/server
        category = msg.get("category", "")
        server = msg.get("server", self.server)
        
        # Version
        version = msg.get("version", 1)
        try:
            version = int(version)
        except (ValueError, TypeError):
            version = 1
        version_str = f"v{version}" if version > 1 else ""
        
        return ArticleCreate(
            title=title,
            url=f"https://www.{server}.org/content/10.1101/{doi}{version_str}",
            source=f"{server.capitalize()} ({category})" if category else server.capitalize(),
            summary=abstract,
            authors=authors_str,
            doi=doi,
            external_id=f"{server}:{doi}{version_str}",
            published_at=pub_date,
        )

    def _clean_text(self, text: str | None) -> str | None:
        if not text:
            return None
        # Remove XML/HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Unescape HTML entities
        text = unescape(text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text or None


# Convenience function for testing
async def test_biorxiv():
    source = BioRxivSource(max_results=5, days_back=7)
    articles = await source._fetch_async()
    await source.close()
    for a in articles:
        print(f"- {a.title[:80]}... (DOI: {a.doi})")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_biorxiv())