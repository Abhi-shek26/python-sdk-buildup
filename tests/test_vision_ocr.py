"""Vision + OCR: payload shape matches docs examples; success + error paths."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from qubrid import AsyncQubridClient, QubridClient
from qubrid.exceptions import NotFoundError

from .conftest import BASE

IMG = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"


def _chat_ok(text: str) -> dict:
    return {
        "id": "c",
        "object": "chat.completion",
        "created": 1,
        "model": "m",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
    }


@respx.mock
def test_vision_payload_shape(sync_client: QubridClient):
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_ok("A statue."))
    )
    resp = sync_client.vision.analyze(prompt="What is in this image?", image_urls=IMG)
    assert resp.content == "A statue."
    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "Qwen/Qwen3-VL-Plus"
    content = body["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "What is in this image?"}
    assert content[1] == {"type": "image_url", "image_url": {"url": IMG}}
    assert body["max_tokens"] == 16384
    assert body["temperature"] == 0.1


@respx.mock
def test_vision_model_not_found(sync_client: QubridClient):
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(
            404,
            json={
                "error": {
                    "message": "model gone",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                    "param": "model",
                    "request_id": "r",
                }
            },
        )
    )
    with pytest.raises(NotFoundError):
        sync_client.vision.analyze(prompt="p", image_urls=IMG, model="nope/model")


@respx.mock
def test_ocr_defaults(sync_client: QubridClient):
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_ok("EXTRACTED"))
    )
    resp = sync_client.ocr.extract(image_urls=IMG)
    assert resp.content == "EXTRACTED"
    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "tencent/HunyuanOCR"
    assert body["stream"] is False
    assert body["max_tokens"] == 8192


@respx.mock
@pytest.mark.asyncio
async def test_vision_async(async_client: AsyncQubridClient):
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_ok("async ok"))
    )
    resp = await async_client.vision.analyze(prompt="p", image_urls=[IMG, IMG])
    assert resp.content == "async ok"


@respx.mock
@pytest.mark.asyncio
async def test_ocr_async(async_client: AsyncQubridClient):
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_ok("t"))
    )
    resp = await async_client.ocr.extract(image_urls=IMG)
    assert resp.content == "t"
