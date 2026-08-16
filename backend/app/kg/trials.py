"""Clinical trial entity grounding via the real ClinicalTrials.gov API.

Unlike the static gazetteer (extractor.py), trials aren't a fixed list --
new NCT ids appear in new articles continuously, so this looks them up live
instead. `fetch_trial_title` is a real, working implementation (verified
against `https://clinicaltrials.gov/api/v2/studies/NCT04173585` during
development, which returned a real title) but is called over the network
at extraction time, so failures are isolated the same way source/embedder
failures are elsewhere in this codebase: a lookup failure skips that NCT id
rather than blocking extraction for the whole article, and does not
fabricate a placeholder entity.

Tests use `fetch_trial_title` as an injectable dependency
(`extract_trial_entities(..., fetcher=...)`) so CI doesn't depend on
network access -- see tests/test_kg.py for the fake-fetcher tests, and
scripts/verify_trial_lookup.py for one that does hit the real API.
"""

import re
from typing import Callable, Optional

import httpx

NCT_PATTERN = re.compile(r"\bNCT\d{8}\b")

TrialFetcher = Callable[[str], Optional[str]]


def extract_nct_ids(text: str) -> list[str]:
    """Distinct NCT ids mentioned in text, in first-seen order."""
    seen: list[str] = []
    for match in NCT_PATTERN.findall(text or ""):
        if match not in seen:
            seen.append(match)
    return seen


def fetch_trial_title(nct_id: str, timeout: float = 5.0) -> Optional[str]:
    """Look up a trial's official brief title from ClinicalTrials.gov.
    Returns None on any failure (timeout, 404, malformed response) -- never
    raises, so callers can treat "no title" as "skip this one" uniformly.
    """
    url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    try:
        resp = httpx.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        title = data.get("protocolSection", {}).get("identificationModule", {}).get("briefTitle")
        return title or None
    except (httpx.HTTPError, ValueError, KeyError):
        return None
