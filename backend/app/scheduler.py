"""Scheduled ingestion using APScheduler."""

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.ingestion.runner import run_and_record


scheduler: AsyncIOScheduler | None = None


def run_scheduled_ingestion():
    """Run ingestion and record the run. Called by scheduler."""
    db = SessionLocal()
    try:
        run = run_and_record(db)
        print(f"[{datetime.now(timezone.utc).isoformat()}] Scheduled ingestion run {run.id}: "
              f"added={run.detail.get('added', {})}, errors={run.detail.get('errors', {})}")
    except Exception as e:
        print(f"[{datetime.now(timezone.utc).isoformat()}] Scheduled ingestion failed: {e}")
    finally:
        db.close()


def start_scheduler():
    """Start the APScheduler for periodic ingestion."""
    global scheduler
    
    if scheduler is not None:
        return
    
    interval_minutes = int(os.getenv("INGESTION_INTERVAL_MINUTES", "60"))
    
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        run_scheduled_ingestion,
        trigger=IntervalTrigger(minutes=interval_minutes),
        id="scheduled_ingestion",
        replace_existing=True,
        max_instances=1,  # Prevent overlapping runs
        coalesce=True,    # If multiple triggers fire, run once
    )
    scheduler.start()
    print(f"Scheduler started: ingestion every {interval_minutes} minutes")


def stop_scheduler():
    """Stop the scheduler."""
    global scheduler
    if scheduler:
        scheduler.shutdown()
        scheduler = None
        print("Scheduler stopped")


@asynccontextmanager
async def lifespan(app):
    """FastAPI lifespan handler for scheduler."""
    # Startup
    start_scheduler()
    yield
    # Shutdown
    stop_scheduler()