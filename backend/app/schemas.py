from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, HttpUrl


class InteractionType(str, Enum):
    READ = "read"
    BOOKMARK = "bookmark"
    LIKE = "like"
    HIDE = "hide"
    SEARCH = "search"


class ArticleBase(BaseModel):
    title: str
    url: str
    source: str
    summary: str | None = None
    authors: str | None = None
    doi: str | None = None
    external_id: str | None = None
    published_at: datetime | None = None


class ArticleCreate(ArticleBase):
    pass


class ArticleRead(ArticleBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    fetched_at: datetime


class ScoredArticleRead(ArticleRead):
    """An article plus its cosine similarity to the query/anchor article."""

    similarity: float
    reason: str | None = None  # "recommended because..."


class IngestResult(BaseModel):
    added: dict[str, int]
    errors: dict[str, str]


class IngestionRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    started_at: datetime
    finished_at: datetime | None
    added_total: int
    error_count: int
    detail: dict


# v0.6: Recommendation engine schemas

class UserCreate(BaseModel):
    email: str | None = None
    apple_user_id: str | None = None
    google_user_id: str | None = None
    display_name: str | None = None


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str | None = None
    display_name: str | None = None
    created_at: datetime


class InteractionCreate(BaseModel):
    article_id: int
    interaction_type: InteractionType
    read_time_seconds: int | None = None
    search_query: str | None = None


class InteractionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    article_id: int
    interaction_type: InteractionType
    read_time_seconds: int | None = None
    search_query: str | None = None
    created_at: datetime


class FeedItem(BaseModel):
    """A feed item with recommendation reason."""
    article: ArticleRead
    similarity: float
    reason: str


class FeedResponse(BaseModel):
    items: list[FeedItem]
    next_cursor: int | None = None
    cold_start: bool = False  # True if using fallback (no user embedding yet)


# v0.9: structured, multi-signal "recommended because..." explanations
# (app/ml/explain.py) -- richer than FeedItem.reason's single string.

class ExplanationSignalRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    label: str
    detail: str
    weight: float


class ExplanationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    summary: str
    signals: list[ExplanationSignalRead]
