"""Async load test for GET /internal/anomaly-explain/{event_id}.

Fires concurrent SSE requests against the real FastAPI route (not directly
at vLLM), so results reflect the full path: FastAPI -> httpx client ->
vLLM's OpenAI-compatible server -> SSE back to the caller. Measures p50/p95/
p99 total latency, throughput (req/s and tokens/s), and time-to-first-token.

Requires a real AnomalyEvent to reference. Seed one first, e.g.:
    curl -X POST "http://localhost:8080/ingest/run"
    curl -X POST "http://localhost:8080/internal/anomaly/detect"
    curl "http://localhost:8080/internal/anomaly/events?limit=1"   # get an id

Usage:
    python scripts/bench_anomaly_explain.py --event-id 1 --concurrency 10 --requests 50 --label stub-cpu
    python scripts/bench_anomaly_explain.py --event-id 1 --concurrency 50 --requests 200 --label vllm-a10g
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx


@dataclass
class RequestResult:
    ok: bool
    total_seconds: float | None = None
    ttft_seconds: float | None = None
    token_count: int = 0
    error: str | None = None


async def _run_one(client: httpx.AsyncClient, url: str) -> RequestResult:
    start = time.perf_counter()
    ttft = None
    token_count = 0
    try:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                if ttft is None:
                    ttft = time.perf_counter() - start
                if line.strip() == "data: [DONE]":
                    break
                token_count += 1
        total = time.perf_counter() - start
        return RequestResult(
            ok=True, total_seconds=total, ttft_seconds=ttft, token_count=token_count
        )
    except Exception as exc:  # noqa: BLE001 - record and continue, don't abort the run
        return RequestResult(ok=False, error=str(exc))


async def run_load_test(
    base_url: str, event_id: int, concurrency: int, total_requests: int
) -> list[RequestResult]:
    url = f"{base_url.rstrip('/')}/internal/anomaly-explain/{event_id}"
    semaphore = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []

    async with httpx.AsyncClient(timeout=60.0) as client:

        async def worker():
            async with semaphore:
                results.append(await _run_one(client, url))

        await asyncio.gather(*(worker() for _ in range(total_requests)))
    return results


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * pct
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def summarize(results: list[RequestResult], wall_seconds: float) -> dict:
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    latencies = [r.total_seconds for r in ok if r.total_seconds is not None]
    ttfts = [r.ttft_seconds for r in ok if r.ttft_seconds is not None]
    total_tokens = sum(r.token_count for r in ok)
    return {
        "requests_total": len(results),
        "requests_ok": len(ok),
        "requests_failed": len(failed),
        "wall_seconds": round(wall_seconds, 3),
        "throughput_req_per_sec": round(len(ok) / wall_seconds, 3) if wall_seconds else 0.0,
        "throughput_tokens_per_sec": round(total_tokens / wall_seconds, 3) if wall_seconds else 0.0,
        "latency_p50_ms": round(_percentile(latencies, 0.50) * 1000, 1),
        "latency_p95_ms": round(_percentile(latencies, 0.95) * 1000, 1),
        "latency_p99_ms": round(_percentile(latencies, 0.99) * 1000, 1),
        "ttft_p50_ms": round(_percentile(ttfts, 0.50) * 1000, 1),
        "ttft_p95_ms": round(_percentile(ttfts, 0.95) * 1000, 1),
        "sample_errors": [r.error for r in failed[:5]],
    }


_REPORT_HEADER = (
    "| label | concurrency | requests | ok | failed | req/s | tokens/s | "
    "p50 (ms) | p95 (ms) | p99 (ms) | TTFT p50 (ms) | TTFT p95 (ms) |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)


def _append_markdown_row(report_path: Path, summary: dict) -> None:
    if not report_path.exists():
        report_path.write_text(
            "# Anomaly-explain benchmark results\n\n"
            "Numbers labeled `stub-*` were measured against the local CPU "
            "stand-in server (see scripts/stub_vllm_server.py) to validate "
            "the FastAPI/SSE wiring end to end -- they are NOT a measurement "
            "of the target 7B AWQ model on GPU. Only `vllm-*` labeled runs "
            "against a real vLLM server on a rented GPU are representative.\n\n"
            + _REPORT_HEADER
        )
    row = (
        f"| {summary['label']} | {summary['concurrency']} | {summary['requests_total']} | "
        f"{summary['requests_ok']} | {summary['requests_failed']} | "
        f"{summary['throughput_req_per_sec']} | {summary['throughput_tokens_per_sec']} | "
        f"{summary['latency_p50_ms']} | {summary['latency_p95_ms']} | {summary['latency_p99_ms']} | "
        f"{summary['ttft_p50_ms']} | {summary['ttft_p95_ms']} |\n"
    )
    with report_path.open("a") as f:
        f.write(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--event-id", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument(
        "--label",
        default="run",
        help="Identifies this run in the report, e.g. 'stub-cpu' or 'vllm-a10g'",
    )
    parser.add_argument("--out-dir", default="benchmarks")
    args = parser.parse_args()

    start = time.perf_counter()
    results = asyncio.run(
        run_load_test(args.base_url, args.event_id, args.concurrency, args.requests)
    )
    wall = time.perf_counter() - start

    summary = summarize(results, wall)
    summary["label"] = args.label
    summary["concurrency"] = args.concurrency
    summary["timestamp"] = datetime.now(timezone.utc).isoformat()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.label}_c{args.concurrency}.json"
    json_path.write_text(json.dumps(summary, indent=2))
    _append_markdown_row(out_dir / "report.md", summary)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {json_path} and appended to {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
