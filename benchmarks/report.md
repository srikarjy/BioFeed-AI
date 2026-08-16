# Anomaly-explain benchmark results

Numbers labeled `stub-*` were measured against the local CPU stand-in server (see scripts/stub_vllm_server.py) to validate the FastAPI/SSE wiring end to end -- they are NOT a measurement of the target 7B AWQ model on GPU. `vllm-t4-direct` hits real vLLM on a real GPU directly, bypassing the FastAPI app. `vllm-t4-full-relay` is the real thing end to end: real vLLM on a real GPU, through the actual FastAPI route → `app/llm/client.py` → SSE relay a client would actually hit.

| label | concurrency | requests | ok | failed | req/s | tokens/s | p50 (ms) | p95 (ms) | p99 (ms) | TTFT p50 (ms) | TTFT p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stub-cpu-local | 10 | 40 | 40 | 0 | 8.128 | 382.005 | 1228.1 | 1244.7 | 1246.1 | 1228.0 | 1244.6 |
| stub-cpu-local | 50 | 100 | 100 | 0 | 33.866 | 1591.71 | 1457.2 | 1514.6 | 1515.5 | 1457.1 | 1514.6 |
| vllm-t4-direct | 10 | 30 | 30 | 0 | 1.819 | 253.874 | 4921.2 | 6957.8 | 6958.7 | 104.8 | 2104.1 |
| vllm-t4-direct | 30 | 60 | 60 | 0 | 2.678 | 385.077 | 10548.0 | 11796.7 | 11800.8 | 5249.4 | 6009.6 |
| vllm-t4-full-relay | 5 | 15 | 15 | 0 | 1.444 | 151.537 | 2746.1 | 5037.3 | 5118.5 | 147.4 | 2240.9 |

## How `vllm-t4-full-relay` was produced (2026-08-15) — the real thing, end to end

Real vLLM (same model/engine args as `vllm-t4-direct`) plus the real FastAPI
backend, both running in the same Hugging Face Jobs `t4-small` container
(`hf jobs run --flavor t4-small -v <clean backend+scripts copy>:/app/upload:ro
vllm/vllm-openai:latest ...`). No Postgres: `DATABASE_URL=sqlite:///...`,
which required a real fix first (see `app/database.py` — SQLite needs
`check_same_thread=False` + `StaticPool` or a real `uvicorn` process serving
sync path operations from a threadpool crashes; production always used
Postgres, so this gap existed unnoticed until an actual end-to-end GPU run
needed something faster to stand up than Postgres). A three-source burst
was seeded through the real detector (`app.anomaly.detector.detect_recent`,
same fixture shape as `tests/test_anomaly.py`), then
`scripts/bench_anomaly_explain.py` hit the real
`GET /internal/anomaly-explain/{event_id}` route — FastAPI → `app/llm/client.py`
→ real vLLM → SSE back to the client, no stand-ins anywhere in the path.
15/15 requests succeeded. Job total 378s (~$0.04 at $0.40/hr). Job id
`srikarjy025/6a810b7a1f5885ae605bc20a`.

**Read against `vllm-t4-direct`**: TTFT p50 is *higher* through the full
relay at the same rough concurrency (147ms vs. 105ms) — the FastAPI hop,
SQLite query, and prompt-building add real overhead, small but nonzero.
Total latency and tokens/s are in the same range as `vllm-t4-direct`
(concurrency 5 here vs. 10 there; not a clean apples-to-apples comparison,
worth re-running at matched concurrency before citing both together).

**Grafana dashboard metric names, checked against this run's real
`/metrics` output**: 7 of the 10 metrics `observability/grafana/dashboards/vllm-anomaly-explain.json`
queries are present as named (`num_requests_running`, `num_requests_waiting`,
`time_to_first_token_seconds`, `e2e_request_latency_seconds`,
`prompt_tokens_total`, `generation_tokens_total`, `request_success_total`).
**3 are missing under those names on this vLLM version**:
`gpu_cache_usage_perc`, `cpu_cache_usage_perc`, `time_per_output_token_seconds`
— likely renamed or restructured in a vLLM release since the dashboard was
written; the dashboard needs updating against the real metric names before
its panels for those 3 would render. Not yet re-verified against a live
Grafana instance (still just a curl comparison, not a screenshot) — see
`PROJECT_STATUS.md`.

## How `vllm-t4-direct` was produced (2026-08-15)

Real vLLM serving `TheBloke/Mistral-7B-Instruct-v0.2-AWQ` on a Hugging Face
Jobs `t4-small` GPU (1x T4, 16GB), launched via `hf jobs run --flavor
t4-small vllm/vllm-openai:latest ...` — the same AWQ checkpoint and engine
args as `llm_serving/serve.sh` (`--max-model-len 2048
--gpu-memory-utilization 0.85 --max-num-seqs 16`, slightly tighter than the
A10G defaults to fit a T4's 16GB). Model loaded and served requests in
~155s from a cold container. Total job wall time 525s (~$0.06 at
$0.40/hr). Job id `srikarjy025/6a80fb241f5885ae605bbf35`.

**What this measures, and what it doesn't**: the benchmark script hits
vLLM's `/v1/chat/completions` directly with the same prompt shape as
`app/llm/prompts.build_explanation_prompt`, streaming, measuring the same
metrics as `bench_anomaly_explain.py` (req/s, tokens/s, latency
percentiles, TTFT). This is real vLLM-on-GPU serving performance — TTFT
p50 105ms at concurrency 10 is a genuine number, not a stand-in. It does
**not** go through the FastAPI app, Postgres, or the SSE relay
(`app/llm/client.py` → `app/anomaly/router.py`) the way
`bench_anomaly_explain.py` does against `stub-cpu-local` above, because
that would need the full backend + Postgres running on the same
ephemeral GPU box, which wasn't attempted this pass. The FastAPI/SSE
relay itself is already verified against the stub server (see below); what
was still unverified before this run was whether vLLM-on-GPU actually
performs the way `llm_serving/serve.sh`'s engine-arg comments assumed —
now confirmed at T4 scale (an A10G/L4, the originally targeted class, will
sit meaningfully lower than these T4 numbers on TTFT/latency given more
VRAM/compute headroom, but that's not measured here).

**Read against the CPU stand-in**: TTFT actually *improves* a lot at
concurrency 10 (105ms vs. `stub-cpu-local`'s 1228ms, because the stub was
measuring a collapsed TTFT artifact — see below), while total-request
latency is higher (real 7B decoding takes real time; the stub server was
not generating real tokens at a comparable cost). Latency degrades sharply
from concurrency 10 to 30 (p50 4.9s → 10.5s) — expected on a single T4 with
`--max-num-seqs 16` under 30 concurrent requests; this is the kind of
number the `--max-num-seqs` / GPU-class tradeoff mentioned in
`llm_serving/serve.sh` is about.

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

## Still open

- **A10G/L4-class numbers**: everything above ran on a T4, the cheapest HF
  Jobs GPU flavor. The originally targeted class in `llm_serving/serve.sh`
  is A10G/L4 (24GB vs. the T4's 16GB) — expect meaningfully better
  TTFT/latency there. Re-run with `--flavor a10g-small` or `l4x1` for a
  direct comparison.
- **Matched-concurrency comparison** between `vllm-t4-direct` and
  `vllm-t4-full-relay` (currently 10 vs. 5) to isolate the FastAPI-hop
  overhead cleanly.
- **Grafana dashboard fix**: update the 3 stale metric names found above
  (`gpu_cache_usage_perc`, `cpu_cache_usage_perc`,
  `time_per_output_token_seconds`) to whatever this vLLM version actually
  exports, then verify the dashboard renders against a live Prometheus
  scrape (not just a metric-name diff).
- **Real Postgres in the loop**: this run used SQLite for speed of
  standing up an ephemeral GPU box; production always uses Postgres, and
  CI already exercises Postgres separately, but a from-scratch
  GPU-host-with-Postgres run hasn't been done.

## Reproducing any of the `vllm-*` rows

1. `hf jobs run --flavor t4-small -v <backend-copy>:/app/upload:ro vllm/vllm-openai:latest ...`
   (or `llm_serving/startup.sh` on a rented A10G/L4 instance).
2. For `vllm-t4-direct`-style (vLLM only): point a benchmark script at
   `<host>:8000/v1/chat/completions` directly.
3. For `vllm-t4-full-relay`-style (the real thing): also start the FastAPI
   backend on the same host, seed an event (`POST /ingest/run`,
   `POST /internal/anomaly/detect`, `GET /internal/anomaly/events`, or the
   direct-seed script shown in this file's git history), then
   `python scripts/bench_anomaly_explain.py --base-url http://localhost:8080 --event-id <id> --concurrency 10 --requests 50 --label vllm-a10g-full-relay`.
4. Rows append to this file automatically.
