"""Chat completions: success, streaming, errors, auth, retry (sync + async)."""

from __future__ import annotations

import httpx
import pytest
import respx

from qubrid import AsyncQubridClient, QubridClient
from qubrid.exceptions import AuthenticationError, InvalidRequestError, RateLimitError

from .conftest import BASE


@respx.mock
def test_chat_create_success(sync_client: QubridClient, chat_ok: dict):
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=chat_ok)
    )
    resp = sync_client.chat.completions.create(
        model="openai/gpt-oss-120b", messages=[{"role": "user", "content": "Hi"}]
    )
    assert route.called
    assert resp.content == "Hello!"
    assert resp.usage.total_tokens == 7
    # exact request shape matches docs example
    sent = route.calls[0].request
    import json

    body = json.loads(sent.content)
    assert body["model"] == "openai/gpt-oss-120b"
    assert body["messages"] == [{"role": "user", "content": "Hi"}]
    assert sent.headers["Authorization"] == "Bearer test-key"


@respx.mock
def test_chat_streaming(sync_client: QubridClient):
    sse = (
        'data: {"id":"c1","model":"m","choices":[{"index":0,'
        '"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'
        'data: {"id":"c1","model":"m","choices":[{"index":0,'
        '"delta":{"content":" world"},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
    )
    chunks = list(
        sync_client.chat.completions.create(
            model="openai/gpt-oss-120b", messages=[{"role": "user", "content": "Hi"}], stream=True
        )
    )
    assert [c.delta for c in chunks] == ["Hello", " world"]


@respx.mock
def test_chat_invalid_request(sync_client: QubridClient, api_error: dict):
    respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(400, json=api_error))
    with pytest.raises(InvalidRequestError) as ei:
        sync_client.chat.completions.create(model="bad", messages=[])
    assert ei.value.param == "model"
    assert ei.value.request_id == "req-1"


@respx.mock
def test_chat_unauthorized():
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(
            401,
            json={
                "error": {
                    "message": "bad key",
                    "type": "authentication_error",
                    "code": "invalid_key",
                    "param": None,
                    "request_id": "r",
                }
            },
        )
    )
    client = QubridClient(api_key="bad", max_retries=0)
    with pytest.raises(AuthenticationError):
        client.chat.completions.create(model="m", messages=[])


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("QUBRID_API_KEY", raising=False)
    with pytest.raises(AuthenticationError, match="QUBRID_API_KEY"):
        QubridClient()


@respx.mock
def test_chat_retries_on_429_then_succeeds(chat_ok: dict):
    client = QubridClient(api_key="k", max_retries=1)
    respx.post(f"{BASE}/chat/completions").mock(
        side_effect=[
            httpx.Response(
                429,
                json={
                    "error": {
                        "message": "slow down",
                        "type": "rate_limit_error",
                        "code": "rate_limit",
                        "param": None,
                        "request_id": "r",
                    }
                },
            ),
            httpx.Response(200, json=chat_ok),
        ]
    )
    resp = client.chat.completions.create(model="m", messages=[])
    assert resp.content == "Hello!"


@respx.mock
def test_chat_rate_limit_exhausted_raises():
    client = QubridClient(api_key="k", max_retries=0)
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(
            429,
            json={
                "error": {
                    "message": "slow",
                    "type": "rate_limit_error",
                    "code": "rate_limit",
                    "param": None,
                    "request_id": "r",
                }
            },
        )
    )
    with pytest.raises(RateLimitError):
        client.chat.completions.create(model="m", messages=[])


@respx.mock
@pytest.mark.asyncio
async def test_chat_async_success(async_client: AsyncQubridClient, chat_ok: dict):
    respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(200, json=chat_ok))
    resp = await async_client.chat.completions.create(model="m", messages=[])
    assert resp.content == "Hello!"


@respx.mock
@pytest.mark.asyncio
async def test_chat_async_stream(async_client: AsyncQubridClient):
    sse = 'data: {"model":"m","choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\ndata: [DONE]\n\n'
    respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(200, text=sse))
    gen = await async_client.chat.completions.create(model="m", messages=[], stream=True)
    chunks = [c async for c in gen]  # type: ignore[union-attr]
    assert chunks[0].delta == "Hi"
