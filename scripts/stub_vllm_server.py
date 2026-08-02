"""Local CPU stand-in for vLLM's OpenAI-compatible server.

This is NOT a benchmark of the target 7B AWQ model on a GPU. It exists so
the FastAPI /internal/anomaly-explain route and the load-testing script can
be exercised end to end without a rented GPU instance. Per-token latency is
a configurable artificial delay, not a measurement -- any numbers produced
against this stub are wiring-validation numbers, not vLLM/GPU numbers, and
the benchmark report says so explicitly.

Run: uvicorn scripts.stub_vllm_server:app --port 8000 --app-dir .
"""

import asyncio
import json
import os
import time

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, StreamingResponse

app = FastAPI(title="stub-vllm-openai-server")

TOKEN_DELAY_SECONDS = float(os.environ.get("STUB_TOKEN_DELAY_SECONDS", "0.02"))
FIRST_TOKEN_DELAY_SECONDS = float(os.environ.get("STUB_FIRST_TOKEN_DELAY_SECONDS", "0.15"))

_STOCK_EXPLANATION = (
    "This article was flagged because multiple independent outlets published "
    "closely matching coverage within a short window, which is the kind of "
    "corroboration that often precedes wider market or clinical attention. "
    "The overlap between the reports suggests they describe the same "
    "underlying event rather than coincidentally similar topics."
).split(" ")


async def _token_stream(model: str):
    request_id = f"stub-{time.time_ns()}"
    await asyncio.sleep(FIRST_TOKEN_DELAY_SECONDS)
    for i, word in enumerate(_STOCK_EXPLANATION):
        content = word if i == 0 else f" {word}"
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(TOKEN_DELAY_SECONDS)
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "stub-model")
    if not body.get("stream", False):
        text = " ".join(_STOCK_EXPLANATION)
        return {"id": "stub", "choices": [{"message": {"role": "assistant", "content": text}}]}
    return StreamingResponse(_token_stream(model), media_type="text/event-stream")


@app.get("/metrics")
def metrics():
    """Minimal Prometheus-format stub so the local docker-compose stack has
    something to scrape end to end. The real deployment points Prometheus at
    vLLM's native /metrics on the GPU host instead of this endpoint.
    """
    return PlainTextResponse(
        "# HELP stub_vllm_requests_total Total stub requests served\n"
        "# TYPE stub_vllm_requests_total counter\n"
        "stub_vllm_requests_total 0\n"
    )
