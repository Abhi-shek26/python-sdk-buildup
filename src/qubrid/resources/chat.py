"""Chat completions resource (sync + async).

``POST /chat/completions`` on the canonical ``/v1`` base — OpenAI-compatible.
Verified against ``qubrid-docs/api-reference/chat/openapi.json`` and
``qubrid-examples/chat/python/chat.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any, Dict, List, Literal, Optional, Union, overload

import httpx

from .._http import iter_sse_lines, parse_sse_chunk, raise_for_response, request_async, request_sync
from .._types import ChatCompletion, ChatCompletionChunk, ChatMessage


def _serialize_messages(messages: List[Union[ChatMessage, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in messages:
        if isinstance(m, ChatMessage):
            out.append(m.model_dump(exclude_none=True))
        else:
            out.append(dict(m))
    return out


class ChatCompletionsSync:
    """Synchronous ``client.chat.completions`` namespace."""

    def __init__(self, http: httpx.Client, base_url: str, max_retries: int) -> None:
        self._http = http
        self._base_url = base_url
        self._max_retries = max_retries

    @overload
    def create(
        self,
        *,
        model: str,
        messages: List[Any],
        stream: Literal[False] = ...,
        max_tokens: ...,
        temperature: ...,
        top_p: ...,
    ) -> ChatCompletion: ...
    @overload
    def create(
        self,
        *,
        model: str,
        messages: List[Any],
        stream: Literal[True],
        max_tokens: ...,
        temperature: ...,
        top_p: ...,
    ) -> Iterator[ChatCompletionChunk]: ...

    def create(
        self,
        *,
        model: str,
        messages: List[Union[ChatMessage, Dict[str, Any]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """Create a chat completion.

        Args:
            model: Model id as plain string, e.g. ``"openai/gpt-oss-120b"``.
            messages: ``[{"role": ..., "content": ...}]`` (dicts or :class:`ChatMessage`).
            max_tokens: Max tokens to generate (docs default 4096).
            temperature: Sampling randomness (docs default 0.7).
            top_p: Nucleus sampling (docs default 1).
            stream: If True, return an iterator of :class:`ChatCompletionChunk`
                parsed from SSE instead of a single :class:`ChatCompletion`.
            extra_body: Extra JSON fields merged into the request (forward-compat).
            timeout: Per-request timeout override in seconds.

        Raises:
            AuthenticationError: 401.
            RateLimitError: 429. InvalidRequestError: 400/422.
            ServerError: 5xx. APIConnectionError: transport failure.
        """
        payload: Dict[str, Any] = {"model": model, "messages": _serialize_messages(messages)}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if extra_body:
            payload.update(extra_body)

        if stream:
            return self._stream(payload, timeout=timeout)
        payload["stream"] = False
        kwargs: Dict[str, Any] = {"json": payload}
        if timeout is not None:
            kwargs["timeout"] = timeout
        resp = request_sync(
            self._http,
            "POST",
            f"{self._base_url}/chat/completions",
            max_retries=self._max_retries,
            **kwargs,
        )
        return ChatCompletion.model_validate(raise_for_response(resp))

    def _stream(
        self, payload: Dict[str, Any], timeout: Optional[float]
    ) -> Iterator[ChatCompletionChunk]:
        payload = {**payload, "stream": True}
        kwargs: Dict[str, Any] = {"json": payload}
        if timeout is not None:
            kwargs["timeout"] = timeout
        # Note: retries for streams apply to connection setup only.
        resp = request_sync(
            self._http,
            "POST",
            f"{self._base_url}/chat/completions",
            max_retries=self._max_retries,
            **kwargs,
        )
        body_text = resp.text
        if not 200 <= resp.status_code < 300:
            try:
                body = resp.json()
            except ValueError:
                body = {"message": body_text[:2000]}
            from ..exceptions import error_from_response

            raise error_from_response(resp.status_code, body)
        for raw in iter_sse_lines(body_text):
            parsed = parse_sse_chunk(raw)
            if parsed is None:
                break
            yield ChatCompletionChunk.model_validate(parsed)


class ChatCompletionsAsync:
    """Asynchronous ``client.chat.completions`` namespace."""

    def __init__(self, http: httpx.AsyncClient, base_url: str, max_retries: int) -> None:
        self._http = http
        self._base_url = base_url
        self._max_retries = max_retries

    async def create(
        self,
        *,
        model: str,
        messages: List[Union[ChatMessage, Dict[str, Any]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        """Async variant of :meth:`ChatCompletionsSync.create`."""
        payload: Dict[str, Any] = {"model": model, "messages": _serialize_messages(messages)}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if extra_body:
            payload.update(extra_body)

        if stream:
            return self._stream(payload, timeout=timeout)
        payload["stream"] = False
        kwargs: Dict[str, Any] = {"json": payload}
        if timeout is not None:
            kwargs["timeout"] = timeout
        resp = await request_async(
            self._http,
            "POST",
            f"{self._base_url}/chat/completions",
            max_retries=self._max_retries,
            **kwargs,
        )
        return ChatCompletion.model_validate(raise_for_response(resp))

    async def _stream(
        self, payload: Dict[str, Any], timeout: Optional[float]
    ) -> AsyncIterator[ChatCompletionChunk]:
        payload = {**payload, "stream": True}
        kwargs: Dict[str, Any] = {"json": payload}
        if timeout is not None:
            kwargs["timeout"] = timeout
        resp = await request_async(
            self._http,
            "POST",
            f"{self._base_url}/chat/completions",
            max_retries=self._max_retries,
            **kwargs,
        )
        body_text = resp.text
        if not 200 <= resp.status_code < 300:
            try:
                body = resp.json()
            except ValueError:
                body = {"message": body_text[:2000]}
            from ..exceptions import error_from_response

            raise error_from_response(resp.status_code, body)
        for raw in iter_sse_lines(body_text):
            parsed = parse_sse_chunk(raw)
            if parsed is None:
                break
            yield ChatCompletionChunk.model_validate(parsed)
