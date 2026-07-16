from fastapi import FastAPI

from app.routers import articles, ingestion, search

app = FastAPI(title="BioFeed AI", version="0.1.0")

app.include_router(articles.router)
app.include_router(ingestion.router)
app.include_router(search.router)


@app.get("/health")
def health():
    return {"status": "ok"}
