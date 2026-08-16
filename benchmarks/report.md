# Anomaly-explain benchmark results

Numbers labeled `stub-*` were measured against the local CPU stand-in server (see scripts/stub_vllm_server.py) to validate the FastAPI/SSE wiring end to end -- they are NOT a measurement of the target 7B AWQ model on GPU. `vllm-t4-direct` (below) is a real measurement of the target model on a real GPU, but hits vLLM directly rather than through the FastAPI SSE relay -- see "How `vllm-t4-direct` was produced" for exactly what that does and doesn't cover.

| label | concurrency | requests | ok | failed | req/s | tokens/s | p50 (ms) | p95 (ms) | p99 (ms) | TTFT p50 (ms) | TTFT p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stub-cpu-local | 10 | 40 | 40 | 0 | 8.128 | 382.005 | 1228.1 | 1244.7 | 1246.1 | 1228.0 | 1244.6 |
| stub-cpu-local | 50 | 100 | 100 | 0 | 33.866 | 1591.71 | 1457.2 | 1514.6 | 1515.5 | 1457.1 | 1514.6 |
| vllm-t4-direct | 10 | 30 | 30 | 0 | 1.819 | 253.874 | 4921.2 | 6957.8 | 6958.7 | 104.8 | 2104.1 |
| vllm-t4-direct | 30 | 60 | 60 | 0 | 2.678 | 385.077 | 10548.0 | 11796.7 | 11800.8 | 5249.4 | 6009.6 |

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

## Still open: real numbers through the full FastAPI/SSE relay

`vllm-t4-direct` above proves vLLM-on-GPU serving works and is fast (TTFT
~105ms at low concurrency). What's still unmeasured is the same request
through the actual product path — FastAPI route → `app/llm/client.py` →
vLLM → SSE back to the caller — which is what `stub-cpu-local` validated
the *wiring* for for, on the CPU stand-in. To get that:

1. Provision a GPU instance/job with the backend + Postgres also running
   (e.g. `docker compose up` on a GPU-enabled host, or a longer-lived HF
   Job/Space that runs both containers), or run `llm_serving/startup.sh` on
   a rented A10G/L4 instance and point a locally-run backend at it via
   `LLM_VLLM_BASE_URL=http://<gpu-host>:8000/v1`.
2. Seed an event (`POST /ingest/run`, `POST /internal/anomaly/detect`, `GET /internal/anomaly/events`).
3. `python scripts/bench_anomaly_explain.py --event-id <id> --concurrency 10 --requests 50 --label vllm-a10g`
4. Repeat at higher concurrency. Rows append to this file automatically.
