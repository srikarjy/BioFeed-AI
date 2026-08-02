"""PubMed source using NCBI E-utilities API."""

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


class PubMedSource(Source):
    """Fetch articles from PubMed via NCBI E-utilities.
    
    Uses esearch + efetch to get recent biotech-relevant papers.
    Requires NCBI_API_KEY env var for higher rate limits (10 req/s vs 3 req/s).
    """

    name = "PubMed"
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    # Biotech-relevant search terms
    DEFAULT_QUERY = (
        "(biotechnology[MeSH] OR CRISPR[Title/Abstract] OR \"gene therapy\"[Title/Abstract] "
        "OR \"mRNA vaccine\"[Title/Abstract] OR \"cell therapy\"[Title/Abstract] "
        "OR \"drug discovery\"[Title/Abstract] OR \"clinical trial\"[Publication Type]) "
        "AND (\"2024\"[Date - Publication] : \"3000\"[Date - Publication])"
    )

    def __init__(
        self,
        query: str | None = None,
        max_results: int = 100,
        api_key: str | None = None,
        email: str | None = None,
    ):
        self.query = query or self.DEFAULT_QUERY
        self.max_results = max_results
        self.api_key = api_key or os.getenv("NCBI_API_KEY")
        self.email = email or os.getenv("NCBI_EMAIL", "biofeed@example.com")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    def _build_esearch_url(self, retstart: int = 0) -> str:
        params = {
            "db": "pubmed",
            "term": self.query,
            "retmax": min(self.max_results, 10000),
            "retstart": retstart,
            "retmode": "json",
            "sort": "pub date",
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key
        return f"{self.BASE_URL}/esearch.fcgi?{urlencode(params)}"

    def _build_efetch_url(self, pmids: list[str]) -> str:
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key
        return f"{self.BASE_URL}/efetch.fcgi?{urlencode(params)}"

    def fetch(self) -> Iterable[ArticleCreate]:
        """Fetch articles from PubMed. Synchronous wrapper for the runner."""
        import asyncio
        return asyncio.run(self._fetch_async())

    async def _fetch_async(self) -> list[ArticleCreate]:
        client = await self._get_client()
        
        # Step 1: esearch to get PMIDs
        search_url = self._build_esearch_url()
        response = await client.get(search_url)
        response.raise_for_status()
        search_data = response.json()
        
        pmids = search_data.get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return []
        
        # Limit to max_results
        pmids = pmids[:self.max_results]
        
        # Step 2: efetch to get full records
        fetch_url = self._build_efetch_url(pmids)
        response = await client.get(fetch_url)
        response.raise_for_status()
        
        return self._parse_pubmed_xml(response.text)

    def _parse_pubmed_xml(self, xml_text: str) -> list[ArticleCreate]:
        """Parse PubMed XML into ArticleCreate objects."""
        from xml.etree import ElementTree as ET
        
        articles = []
        root = ET.fromstring(xml_text)
        
        for article_elem in root.findall(".//PubmedArticle"):
            try:
                article = self._parse_single_article(article_elem)
                if article:
                    articles.append(article)
            except Exception:
                continue  # Skip malformed records
        
        return articles

    def _get_element_text(self, elem) -> str | None:
        """Get all text content from an element including nested elements."""
        if elem is None:
            return None
        # Use itertext() to get text from element and all children
        text = "".join(elem.itertext())
        return self._clean_text(text)

    def _parse_single_article(self, article_elem) -> ArticleCreate | None:
        """Parse a single PubmedArticle element."""
        
        # PMID
        pmid_elem = article_elem.find(".//PMID")
        if pmid_elem is None or not pmid_elem.text:
            return None
        pmid = pmid_elem.text.strip()
        
        # Title - use itertext to get nested content (e.g., <i> tags)
        title_elem = article_elem.find(".//ArticleTitle")
        title = self._get_element_text(title_elem)
        if not title:
            return None
        
        # Abstract - use itertext for nested elements
        abstract_parts = []
        for abstract_elem in article_elem.findall(".//Abstract/AbstractText"):
            text = self._get_element_text(abstract_elem)
            if text:
                label = abstract_elem.get("Label", "")
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
        abstract = " ".join(abstract_parts) if abstract_parts else None
        
        # Authors
        authors = []
        for author_elem in article_elem.findall(".//Author"):
            last = author_elem.findtext("LastName", "")
            fore = author_elem.findtext("ForeName", "") or author_elem.findtext("Initials", "")
            if last:
                authors.append(f"{fore} {last}".strip())
        authors_str = "; ".join(authors) if authors else None
        
        # DOI
        doi = None
        for id_elem in article_elem.findall(".//ArticleId"):
            if id_elem.get("IdType") == "doi":
                doi = id_elem.text
                break
        
        # Publication date
        pub_date = self._parse_pub_date(article_elem)
        
        # Journal
        journal_elem = article_elem.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None else "PubMed"
        
        return ArticleCreate(
            title=title,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            source="PubMed",
            summary=abstract,
            authors=authors_str,
            doi=doi,
            external_id=pmid,
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

    def _parse_pub_date(self, article_elem) -> datetime | None:
        """Extract publication date from PubDate element."""
        pub_date_elem = article_elem.find(".//PubDate")
        if pub_date_elem is None:
            return None
        
        year = pub_date_elem.findtext("Year")
        month = pub_date_elem.findtext("Month", "Jan")
        day = pub_date_elem.findtext("Day", "1")
        
        if not year:
            return None
        
        # Handle month names
        month_map = {
            "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
            "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
        }
        month_num = month_map.get(month[:3], 1)
        
        try:
            return datetime(int(year), month_num, int(day), tzinfo=timezone.utc)
        except ValueError:
            return None


# Convenience function for testing
async def test_pubmed():
    source = PubMedSource(max_results=5)
    articles = await source._fetch_async()
    await source.close()
    for a in articles:
        print(f"- {a.title[:80]}... (PMID: {a.external_id})")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_pubmed())