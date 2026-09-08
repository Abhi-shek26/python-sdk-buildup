"""Audio TTS + STT: success, legacy routing, multipart, errors (sync + async)."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from qubrid import AsyncQubridClient, QubridClient
from qubrid.exceptions import InvalidRequestError

from .conftest import BASE, LEGACY


@respx.mock
def test_tts_success(sync_client: QubridClient):
    body = {"created": 1710000000, "data": [{"url": "https://cdn.qubrid.com/generated/audio.mp3"}]}
    route = respx.post(f"{BASE}/audio/generations").mock(
        return_value=httpx.Response(200, json=body)
    )
    resp = sync_client.audio.speech.create(text="Hello world", voice="Cherry")
    assert resp.url == "https://cdn.qubrid.com/generated/audio.mp3"
    sent = json.loads(route.calls[0].request.content)
    assert sent == {
        "model": "qwen3-tts-flash",
        "text": "Hello world",
        "voice": "Cherry",
        "language_type": "Auto",
    }


@respx.mock
def test_tts_invalid_request(sync_client: QubridClient):
    respx.post(f"{BASE}/audio/generations").mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "message": "Missing required parameter: 'text'.",
                    "type": "invalid_request_error",
                    "code": "invalid_request",
                    "param": "text",
                    "request_id": "r",
                }
            },
        )
    )
    with pytest.raises(InvalidRequestError) as ei:
        sync_client.audio.speech.create(text="", voice="Cherry")
    assert ei.value.param == "text"


@respx.mock
def test_stt_routes_to_legacy_prefix(sync_client: QubridClient):
    route = respx.post(f"{LEGACY}/audio/transcribe").mock(
        return_value=httpx.Response(200, json={"text": "hello"})
    )
    resp = sync_client.audio.transcriptions.create(file=("clip.wav", b"RIFF...."))
    assert resp.text == "hello"
    assert route.called
    ctype = route.calls[0].request.headers["content-type"]
    assert "multipart/form-data" in ctype


@respx.mock
def test_stt_from_path(sync_client: QubridClient, tmp_path):
    f = tmp_path / "a.wav"
    f.write_bytes(b"fake-audio")
    route = respx.post(f"{LEGACY}/audio/transcribe").mock(
        return_value=httpx.Response(200, json={"text": "hi", "language": "en"})
    )
    resp = sync_client.audio.transcriptions.create(file=f)
    assert resp.text == "hi"
    assert resp.language == "en"
    assert route.called


@respx.mock
@pytest.mark.asyncio
async def test_audio_async(async_client: AsyncQubridClient):
    respx.post(f"{BASE}/audio/generations").mock(
        return_value=httpx.Response(200, json={"created": 1, "data": [{"url": "https://x/y.mp3"}]})
    )
    r = await async_client.audio.speech.create(text="hi", voice="Elias")
    assert r.url == "https://x/y.mp3"
    respx.post(f"{LEGACY}/audio/transcribe").mock(
        return_value=httpx.Response(200, json={"text": "async hi"})
    )
    t = await async_client.audio.transcriptions.create(file=("a.wav", b"xx"))
    assert t.text == "async hi"
