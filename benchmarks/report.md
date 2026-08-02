# Anomaly-explain benchmark results

Numbers labeled `stub-*` were measured against the local CPU stand-in server (see scripts/stub_vllm_server.py) to validate the FastAPI/SSE wiring end to end -- they are NOT a measurement of the target 7B AWQ model on GPU. Only `vllm-*` labeled runs against a real vLLM server on a rented GPU are representative. No GPU instance was available when this was written; that run is still pending (see the README for how to produce it).

| label | concurrency | requests | ok | failed | req/s | tokens/s | p50 (ms) | p95 (ms) | p99 (ms) | TTFT p50 (ms) | TTFT p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stub-cpu-local | 10 | 40 | 40 | 0 | 8.128 | 382.005 | 1228.1 | 1244.7 | 1246.1 | 1228.0 | 1244.6 |
| stub-cpu-local | 50 | 100 | 100 | 0 | 33.866 | 1591.71 | 1457.2 | 1514.6 | 1515.5 | 1457.1 | 1514.6 |

## How `stub-cpu-local` was produced

Not run via `scripts/bench_anomaly_explain.py` against a real socket (this
sandbox has no Docker/Postgres available) -- instead the FastAPI app was
driven in-process over an ASGI transport (`httpx.ASGITransport`), backed by
an in-memory SQLite DB seeded with the same three-source burst fixture used
in `tests/test_anomaly.py`, against a real `stub_vllm_server.py` process on
`localhost:9100`. The request path (FastAPI route -> httpx client -> SSE
relay) is real; the DB layer and transport are stand-ins.

**Known artifact**: TTFT ≈ total latency above. `httpx.ASGITransport`
buffers the ASGI response before exposing lines to `aiter_lines()`, so the
"first token" timestamp collapses to "last token" timestamp in this
in-process setup -- it does not mean the SSE stream isn't incrementally
flushed over a real socket. Confirm true TTFT once running against a real
bound port (either the stub server directly, or vLLM on GPU) via
`scripts/bench_anomaly_explain.py`, which uses a real HTTP connection where
`aiter_lines()` yields as bytes arrive.

## To get real vLLM/GPU numbers

1. Provision an A10G/L4 instance, run `llm_serving/startup.sh`.
2. Point the backend's `.env` at it: `LLM_VLLM_BASE_URL=http://<gpu-host>:8000/v1`.
3. Seed an event (`POST /ingest/run`, `POST /internal/anomaly/detect`, `GET /internal/anomaly/events`).
4. `python scripts/bench_anomaly_explain.py --event-id <id> --concurrency 10 --requests 50 --label vllm-a10g`
5. Repeat at concurrency 50. Rows append to this file automatically.
