from datetime import datetime

from pydantic import BaseModel, ConfigDict


class AnomalyEventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    article_id: int
    kind: str
    score: float
    detail: dict
    detected_at: datetime
