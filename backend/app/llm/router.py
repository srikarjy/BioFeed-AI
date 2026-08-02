"""Internal-only route: streams an LLM-generated explanation for a candidate
anomaly event. Pure downstream consumer of app.anomaly -- reads an
AnomalyEvent + its Article and evidence, never writes to either. Not mounted
under any existing prefix so it can be disabled (LLM_ENABLED=false) or
deleted without touching the public articles/search/ingestion routes.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.anomaly import crud as anomaly_crud
from app.database import get_db
from app.llm.client import VLLMClientError, stream_completion
from app.llm.config import llm_settings
from app.llm.prompts import build_explanation_prompt
from app.models import Article

router = APIRouter(prefix="/internal", tags=["llm"])


@router.get("/anomaly-explain/{event_id}")
async def explain_anomaly_event(event_id: int, db: Session = Depends(get_db)):
    if not llm_settings.enabled:
        raise HTTPException(status_code=503, detail="LLM explanation feature is disabled")

    event = anomaly_crud.get_event(db, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Anomaly event not found")

    article = db.get(Article, event.article_id)
    if article is None:
        raise HTTPException(status_code=404, detail="Flagged article not found")

    related_ids = event.detail.get("related_article_ids", [])
    related = [a for a in (db.get(Article, rid) for rid in related_ids) if a is not None]

    # Build the prompt synchronously, using the request-scoped db session,
    # before opening the stream -- the generator below never touches db, so
    # session lifecycle doesn't matter once streaming starts.
    prompt = build_explanation_prompt(event, article, related)

    async def event_stream():
        try:
            async for delta in stream_completion(prompt):
                yield f"data: {delta}\n\n"
        except VLLMClientError as exc:
            yield f"event: error\ndata: {exc}\n\n"
            return
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
