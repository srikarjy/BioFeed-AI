from datetime import datetime

from pydantic import BaseModel, ConfigDict


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
