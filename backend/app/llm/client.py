"""Thin async client for a vLLM OpenAI-compatible chat completion endpoint.

Deliberately not the official openai SDK: vLLM's server implements a subset
of the OpenAI API and pulling in the full SDK for one streaming call is more
dependency than this needs. httpx is already a project dependency.
"""

import json
from collections.abc import AsyncIterator

import httpx

from app.llm.config import llm_settings


class VLLMClientError(RuntimeError):
    pass


async def stream_completion(prompt: str) -> AsyncIterator[str]:
    """Yield text deltas from the vLLM chat/completions SSE stream."""
    payload = {
        "model": llm_settings.vllm_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": llm_settings.max_tokens,
        "temperature": llm_settings.temperature,
        "stream": True,
    }
    url = f"{llm_settings.vllm_base_url.rstrip('/')}/chat/completions"

    try:
        async with httpx.AsyncClient(timeout=llm_settings.request_timeout_seconds) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[len("data: ") :]
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content")
                    if delta:
                        yield delta
    except httpx.HTTPError as exc:
        raise VLLMClientError(f"vLLM request failed: {exc}") from exc
