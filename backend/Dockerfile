FROM python:3.12-slim

WORKDIR /app

# torch + lightgbm both bundle their own OpenMP runtime; on the macOS dev
# machine, loading both in one process and then calling into lightgbm
# segfaults (SIGSEGV) unless OpenMP is pinned to a single thread. Set
# defensively here too -- not yet confirmed whether the Linux/slim image
# needs it (no GPU/Docker runtime available to verify in this environment),
# but it's a no-op-cost single-thread pin at this corpus scale either way.
# See app/ml/recommender_v07.py and scripts/train_v07.py for the repro.
ENV OMP_NUM_THREADS=1
ENV KMP_DUPLICATE_LIB_OK=TRUE

COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-ml.txt

COPY . .

CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
