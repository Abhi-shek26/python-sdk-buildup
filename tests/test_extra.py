"""Embeddings (experimental passthrough), images, video: success + error paths."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from qubrid import AsyncQubridClient, QubridClient
from qubrid.exceptions import InvalidRequestError, QuotaExceededError

from .conftest import BASE


@respx.mock
def test_embeddings_passthrough(sync_client: QubridClient):
    body = {
        "object": "list",
        "model": "BAAI/bge-large-en-v1.5",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
        "usage": {"prompt_tokens": 3, "total_tokens": 3},
    }
    route = respx.post(f"{BASE}/embeddings").mock(return_value=httpx.Response(200, json=body))
    resp = sync_client.embeddings.create(model="BAAI/bge-large-en-v1.5", input="hello")
    assert resp.data[0].embedding == [0.1, 0.2]
    sent = json.loads(route.calls[0].request.content)
    assert sent == {"model": "BAAI/bge-large-en-v1.5", "input": "hello"}


@respx.mock
def test_embeddings_quota(sync_client: QubridClient):
    respx.post(f"{BASE}/embeddings").mock(
        return_value=httpx.Response(
            402,
            json={
                "error": {
                    "message": "Insufficient credits",
                    "type": "insufficient_quota",
                    "code": "insufficient_quota",
                    "param": None,
                    "request_id": "r",
                }
            },
        )
    )
    with pytest.raises(QuotaExceededError):
        sync_client.embeddings.create(model="m", input="x")


@respx.mock
def test_images_generate(sync_client: QubridClient):
    body = {"created": 1, "data": [{"url": "https://cdn.qubrid.com/i.webp"}]}
    route = respx.post(f"{BASE}/images/generations").mock(
        return_value=httpx.Response(200, json=body)
    )
    resp = sync_client.images.generate(
        model="p-image", prompt="a cat", aspect_ratio="16:9", output_format="webp"
    )
    assert resp.data[0].url == "https://cdn.qubrid.com/i.webp"
    sent = json.loads(route.calls[0].request.content)
    assert sent["model"] == "p-image" and sent["prompt"] == "a cat"


@respx.mock
def test_images_edit_remote_urls_json(sync_client: QubridClient):
    body = {"created": 1, "data": [{"url": "https://cdn.qubrid.com/e.webp"}]}
    route = respx.post(f"{BASE}/images/edits").mock(return_value=httpx.Response(200, json=body))
    resp = sync_client.images.edit(
        model="p-image-edit", prompt="add hat", image="https://example.com/in.png"
    )
    assert resp.data[0].url.endswith("e.webp")
    sent = json.loads(route.calls[0].request.content)
    assert sent["images"] == ["https://example.com/in.png"]


@respx.mock
def test_images_edit_local_file_multipart(sync_client: QubridClient):
    body = {"created": 1, "data": [{"url": "https://cdn.qubrid.com/e2.webp"}]}
    route = respx.post(f"{BASE}/images/edits").mock(return_value=httpx.Response(200, json=body))
    resp = sync_client.images.edit(
        model="p-image-edit", prompt="edit", image=("in.png", b"PNGDATA")
    )
    assert resp.data[0].url.endswith("e2.webp")
    assert "multipart/form-data" in route.calls[0].request.headers["content-type"]


@respx.mock
def test_videos_generate_url_and_video_url_alias(sync_client: QubridClient):
    respx.post(f"{BASE}/videos/generations").mock(
        return_value=httpx.Response(
            200, json={"created": 1, "data": [{"video_url": "https://cdn.qubrid.com/v.mp4"}]}
        )
    )
    resp = sync_client.videos.generate(model="p-video", prompt="waves", duration=5)
    assert resp.url == "https://cdn.qubrid.com/v.mp4"  # video_url alias handled


@respx.mock
def test_images_invalid(sync_client: QubridClient):
    respx.post(f"{BASE}/images/generations").mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "message": "bad prompt",
                    "type": "invalid_request_error",
                    "code": "invalid_request",
                    "param": "prompt",
                    "request_id": "r",
                }
            },
        )
    )
    with pytest.raises(InvalidRequestError):
        sync_client.images.generate(model="p-image", prompt="")


@respx.mock
@pytest.mark.asyncio
async def test_extra_async(async_client: AsyncQubridClient):
    respx.post(f"{BASE}/embeddings").mock(
        return_value=httpx.Response(
            200, json={"object": "list", "data": [{"index": 0, "embedding": [1.0]}], "model": "m"}
        )
    )
    e = await async_client.embeddings.create(model="m", input="x")
    assert e.data[0].embedding == [1.0]
    respx.post(f"{BASE}/images/generations").mock(
        return_value=httpx.Response(200, json={"created": 1, "data": [{"url": "u"}]})
    )
    img = await async_client.images.generate(model="p-image", prompt="p")
    assert img.data[0].url == "u"
    respx.post(f"{BASE}/videos/generations").mock(
        return_value=httpx.Response(200, json={"created": 1, "data": [{"url": "v"}]})
    )
    v = await async_client.videos.generate(prompt="p")
    assert v.url == "v"
