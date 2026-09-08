"""Shared httpx plumbing: headers, retry loop, error mapping, SSE parsing.

Kept in one place so sync and async resources behave identically.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterator
from typing import Any, Dict, Optional

import httpx

from ._utils import is_retryable, parse_retry_after, retry_delay
from .exceptions import APIConnectionError, error_from_response


def build_headers(api_key: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    headers = {"Authorization": f"Bearer {api_key}"}
    if extra:
        headers.update(extra)
    return headers


def _parse_body(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        text = response.text
        return {"message": text[:2000]} if text else {}


def _maybe_raise(status_code: int, body: Any) -> None:
    if 200 <= status_code < 300:
        return
    raise error_from_response(status_code, body)


def request_sync(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    max_retries: int,
    **kwargs: Any,
) -> httpx.Response:
    """Send one request with retry-with-backoff on 429/5xx."""
    attempt = 0
    while True:
        try:
            resp = client.request(method, url, **kwargs)
        except (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
            httpx.RemoteProtocolError,
        ) as exc:
            if attempt >= max_retries:
                raise APIConnectionError(f"Connection failed: {exc}") from exc
            time.sleep(retry_delay(attempt))
            attempt += 1
            continue
        if is_retryable(resp.status_code) and attempt < max_retries:
            time.sleep(retry_delay(attempt, parse_retry_after(resp.headers.get("retry-after"))))
            attempt += 1
            continue
        return resp


async def request_async(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    max_retries: int,
    **kwargs: Any,
) -> httpx.Response:
    attempt = 0
    while True:
        try:
            resp = await client.request(method, url, **kwargs)
        except (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
            httpx.RemoteProtocolError,
        ) as exc:
            if attempt >= max_retries:
                raise APIConnectionError(f"Connection failed: {exc}") from exc
            await asyncio.sleep(retry_delay(attempt))
            attempt += 1
            continue
        if is_retryable(resp.status_code) and attempt < max_retries:
            await asyncio.sleep(
                retry_delay(attempt, parse_retry_after(resp.headers.get("retry-after")))
            )
            attempt += 1
            continue
        return resp


def raise_for_response(response: httpx.Response) -> Any:
    """Raise a typed ``QubridError`` on non-2xx, else return parsed JSON body."""
    body = _parse_body(response)
    _maybe_raise(response.status_code, body)
    return body


def iter_sse_lines(payload: str) -> Iterator[str]:
    """Yield raw ``data:`` payloads from an SSE text buffer."""
    for line in payload.splitlines():
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            yield line[5:].strip()


def parse_sse_chunk(raw: str) -> Optional[Dict[str, Any]]:
    """Parse one SSE ``data:`` payload → dict, or ``None`` for ``[DONE]``."""
    if raw == "[DONE]":
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else {"content": parsed}
