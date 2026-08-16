from datetime import datetime

from pydantic import BaseModel, ConfigDict


class EntityRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    entity_type: str
    external_source: str | None = None
    external_id: str | None = None
    aliases: list[str]
    created_at: datetime


class RelationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    subject_entity_id: int
    predicate: str
    object_entity_id: int
    evidence_article_id: int
    created_at: datetime


class ExtractResult(BaseModel):
    articles_processed: int
