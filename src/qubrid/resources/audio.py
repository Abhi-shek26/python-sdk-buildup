"""Audio resources: TTS (``/v1``) + STT transcriptions (legacy prefix).

TTS: ``POST {base}/audio/generations`` JSON ``{model, text, voice, language_type}``
  → ``{created, data: [{url}]}`` (``qubrid-docs/api-reference/voice/openapi.json``).
STT: ``POST {legacy_base}/audio/transcribe`` multipart ``{file, model}``
  → ``{text}`` (``qubrid-docs/api-reference/voice/openapistt.json``).
  STT is the ONLY endpoint documented under the legacy
  ``/api/v1/qubridai`` prefix — the client routes it there automatically.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Tuple, Union

import httpx

from .._http import raise_for_response, request_async, request_sync
from .._types import SpeechGeneration, Transcription

FileInput = Union[str, os.PathLike, BinaryIO, Tuple[str, bytes], Tuple[str, bytes, str]]


def _coerce_upload(file: FileInput, default_filename: str = "audio") -> Tuple[str, bytes, str]:
    """Normalise path / bytes / file-object / tuple → ``(filename, bytes, mime)``."""
    if isinstance(file, tuple):
        if len(file) == 2:
            name, data = file
            return name, data, "application/octet-stream"
        name, data, mime = file  # type: ignore[misc]
        return name, data, mime
    if isinstance(file, (str, os.PathLike)):
        p = Path(file)
        data = p.read_bytes()
        return p.name or default_filename, data, "application/octet-stream"
    if hasattr(file, "read"):
        data = file.read()  # type: ignore[union-attr]
        if isinstance(data, str):
            data = data.encode()
        name = getattr(file, "name", default_filename)
        return str(name).split("/")[-1], data, "application/octet-stream"
    raise TypeError("file must be a path, file object, or (filename, bytes) tuple")


class SpeechSync:
    """Synchronous ``client.audio.speech`` namespace (TTS)."""

    def __init__(self, http: httpx.Client, base_url: str, max_retries: int) -> None:
        self._http = http
        self._base_url = base_url
        self._max_retries = max_retries

    def create(
        self,
        *,
        model: str = "qwen3-tts-flash",
        text: str,
        voice: str = "Cherry",
        language_type: Optional[str] = "Auto",
        timeout: Optional[float] = None,
    ) -> SpeechGeneration:
        """Synthesize speech from text; returns hosted audio URL(s).

        .. warning:: As of 2026-09-08, successful usage is unverified: the
            endpoint is reachable and accepts the documented request shape,
            but Qubrid's safety filter rejected every tested input with
            ``400 content_policy_violation`` — including the docs' own example
            sentence (``"Today is a wonderful day to build something people
            love!"``). Treat TTS as unstable until a passing input is found.

        Args:
            model: TTS model id (examples use ``qwen3-tts-flash``).
            text: Text to speak.
            voice: Voice preset, e.g. Cherry, Elias, Arthur, Nini, Ebona,
                Seren, Pip, Stella (see ``qubrid-docs/audio/`` demos).
            language_type: e.g. Auto, English, Chinese, German, ... (optional).
        """
        payload: Dict[str, Any] = {"model": model, "text": text, "voice": voice}
        if language_type is not None:
            payload["language_type"] = language_type
        kwargs: Dict[str, Any] = {"json": payload}
        if timeout is not None:
            kwargs["timeout"] = timeout
        resp = request_sync(
            self._http,
            "POST",
            f"{self._base_url}/audio/generations",
            max_retries=self._max_retries,
            **kwargs,
        )
        return SpeechGeneration.model_validate(raise_for_response(resp))


class SpeechAsync:
    """Asynchronous ``client.audio.speech`` namespace (TTS)."""

    def __init__(self, http: httpx.AsyncClient, base_url: str, max_retries: int) -> None:
        self._http = http
        self._base_url = base_url
        self._max_retries = max_retries

    async def create(
        self,
        *,
        model: str = "qwen3-tts-flash",
        text: str,
        voice: str = "Cherry",
        language_type: Optional[str] = "Auto",
        timeout: Optional[float] = None,
    ) -> SpeechGeneration:
        """Async variant of :meth:`SpeechSync.create`."""
        payload: Dict[str, Any] = {"model": model, "text": text, "voice": voice}
        if language_type is not None:
            payload["language_type"] = language_type
        kwargs: Dict[str, Any] = {"json": payload}
        if timeout is not None:
            kwargs["timeout"] = timeout
        resp = await request_async(
            self._http,
            "POST",
            f"{self._base_url}/audio/generations",
            max_retries=self._max_retries,
            **kwargs,
        )
        return SpeechGeneration.model_validate(raise_for_response(resp))


class TranscriptionsSync:
    """Synchronous ``client.audio.transcriptions`` namespace (STT)."""

    def __init__(self, http: httpx.Client, legacy_base_url: str, max_retries: int) -> None:
        self._http = http
        self._legacy_base_url = legacy_base_url
        self._max_retries = max_retries

    def create(
        self,
        *,
        model: str = "openai/whisper-large-v3",
        file: FileInput,
        filename: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Transcription:
        """Transcribe audio → text (multipart ``file`` + ``model``).

        Args:
            model: STT model id (only documented value is
                ``openai/whisper-large-v3``).
            file: Path, open binary file, or ``(filename, bytes)`` tuple.
            filename: Override the uploaded filename.
            timeout: Uploads can be slow — pass a larger value for long audio.
        """
        name, data, mime = _coerce_upload(file)
        files = {"file": (filename or name, io.BytesIO(data), mime)}
        data_fields = {"model": model}
        kwargs: Dict[str, Any] = {"files": files, "data": data_fields}
        if timeout is not None:
            kwargs["timeout"] = timeout
        resp = request_sync(
            self._http,
            "POST",
            f"{self._legacy_base_url}/audio/transcribe",
            max_retries=self._max_retries,
            **kwargs,
        )
        return Transcription.model_validate(raise_for_response(resp))


class TranscriptionsAsync:
    """Asynchronous ``client.audio.transcriptions`` namespace (STT)."""

    def __init__(self, http: httpx.AsyncClient, legacy_base_url: str, max_retries: int) -> None:
        self._http = http
        self._legacy_base_url = legacy_base_url
        self._max_retries = max_retries

    async def create(
        self,
        *,
        model: str = "openai/whisper-large-v3",
        file: FileInput,
        filename: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Transcription:
        """Async variant of :meth:`TranscriptionsSync.create`."""
        name, data, mime = _coerce_upload(file)
        files = {"file": (filename or name, io.BytesIO(data), mime)}
        data_fields = {"model": model}
        kwargs: Dict[str, Any] = {"files": files, "data": data_fields}
        if timeout is not None:
            kwargs["timeout"] = timeout
        resp = await request_async(
            self._http,
            "POST",
            f"{self._legacy_base_url}/audio/transcribe",
            max_retries=self._max_retries,
            **kwargs,
        )
        return Transcription.model_validate(raise_for_response(resp))


class AudioSync:
    """Synchronous ``client.audio`` namespace grouping speech + transcriptions."""

    def __init__(
        self, http: httpx.Client, base_url: str, legacy_base_url: str, max_retries: int
    ) -> None:
        self.speech = SpeechSync(http, base_url, max_retries)
        self.transcriptions = TranscriptionsSync(http, legacy_base_url, max_retries)


class AudioAsync:
    """Asynchronous ``client.audio`` namespace grouping speech + transcriptions."""

    def __init__(
        self, http: httpx.AsyncClient, base_url: str, legacy_base_url: str, max_retries: int
    ) -> None:
        self.speech = SpeechAsync(http, base_url, max_retries)
        self.transcriptions = TranscriptionsAsync(http, legacy_base_url, max_retries)
