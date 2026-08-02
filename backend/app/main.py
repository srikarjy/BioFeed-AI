from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.anomaly.router import router as anomaly_router
from app.llm.router import router as llm_router
from app.routers import articles, ingestion, search, recommendations, recommendations_v07
from app.scheduler import lifespan as scheduler_lifespan

app = FastAPI(title="BioFeed AI", version="0.7.0", lifespan=scheduler_lifespan)

app.include_router(articles.router)
app.include_router(ingestion.router)
app.include_router(search.router)
app.include_router(recommendations.router)
app.include_router(recommendations_v07.router)
app.include_router(anomaly_router)
app.include_router(llm_router)


@app.get("/health")
def health():
    return {"status": "ok"}
