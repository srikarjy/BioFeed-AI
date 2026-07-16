from abc import ABC, abstractmethod
from collections.abc import Iterable

from app.schemas import ArticleCreate


class Source(ABC):
    """A content source that yields normalized articles.

    Each concrete source (RSS feed, PubMed query, bioRxiv API) owns its own
    fetching and parsing and emits ArticleCreate items. The runner is agnostic
    to where they came from, so adding a source never touches the runner.
    """

    name: str

    @abstractmethod
    def fetch(self) -> Iterable[ArticleCreate]:
        """Fetch from the upstream source and yield normalized articles.

        Should raise on a hard failure (network/parse error) so the runner can
        isolate and record it per-source without aborting the whole run.
        """
        raise NotImplementedError
