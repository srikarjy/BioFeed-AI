from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ArticleBase(BaseModel):
    title: str
    url: str
    source: str
    summary: str | None = None
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
