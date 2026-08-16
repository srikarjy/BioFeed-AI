from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.anomaly.router import router as anomaly_router
from app.auth.router import router as auth_router
from app.kg.router import router as kg_router
from app.llm.router import router as llm_router
from app.routers import articles, ingestion, search, recommendations, recommendations_v07
from app.scheduler import lifespan as scheduler_lifespan

app = FastAPI(title="BioFeed AI", version="0.9.0", lifespan=scheduler_lifespan)

app.include_router(articles.router)
app.include_router(ingestion.router)
app.include_router(search.router)
app.include_router(auth_router)
app.include_router(recommendations.router)
app.include_router(recommendations_v07.router)
app.include_router(kg_router)
app.include_router(anomaly_router)
app.include_router(llm_router)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/demo")
def demo():
    """A same-origin static page that exercises the live API from a browser
    (search, articles, KG, feed/explain via the fake-provider dev auth path)
    -- something clickable for a reviewer, not just documented endpoints."""
    return FileResponse(STATIC_DIR / "demo.html")
