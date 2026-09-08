"""Top-level clients: :class:`QubridClient` (sync) and :class:`AsyncQubridClient`.

Canonical base ``https://platform.qubrid.com/v1`` serves everything except STT,
which is auto-routed to the legacy ``https://platform.qubrid.com/api/v1/qubridai``
prefix (the only STT path in ``api-reference/voice/openapistt.json``).
"""

from __future__ import annotations

from typing import Optional

import httpx

from .__version__ import __version__
from ._utils import (
    DEFAULT_BASE_URL,
    DEFAULT_LEGACY_BASE_URL,
    normalize_base_url,
    resolve_api_key,
)
from .resources.audio import AudioAsync, AudioSync
from .resources.chat import ChatCompletionsAsync, ChatCompletionsSync
from .resources.extra import (
    EmbeddingsAsync,
    EmbeddingsSync,
    ImagesAsync,
    ImagesSync,
    VideosAsync,
    VideosSync,
)
from .resources.vision_ocr import OCRAsync, OCRSync, VisionAsync, VisionSync

__all__ = ["AsyncQubridClient", "QubridClient"]


class _ChatSync:
    def __init__(self, http: httpx.Client, base_url: str, max_retries: int) -> None:
        self.completions = ChatCompletionsSync(http, base_url, max_retries)


class _ChatAsync:
    def __init__(self, http: httpx.AsyncClient, base_url: str, max_retries: int) -> None:
        self.completions = ChatCompletionsAsync(http, base_url, max_retries)


class QubridClient:
    """Synchronous Qubrid client.

    Example:
        >>> client = QubridClient()  # reads QUBRID_API_KEY
        >>> resp = client.chat.completions.create(
        ...     model="openai/gpt-oss-120b",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ... )

    Args:
        api_key: Bearer token; falls back to ``QUBRID_API_KEY`` env var.
        base_url: Override for the canonical ``/v1`` prefix.
        legacy_base_url: Override for the STT ``/api/v1/qubridai`` prefix.
        timeout: Default per-request timeout in seconds.
        max_retries: Retries with backoff on 429/5xx (default 2).
        http_client: Bring your own configured ``httpx.Client`` (mainly for tests).
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        legacy_base_url: str = DEFAULT_LEGACY_BASE_URL,
        timeout: float = 60.0,
        max_retries: int = 2,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.api_key = resolve_api_key(api_key)
        self.base_url = normalize_base_url(base_url)
        self.legacy_base_url = normalize_base_url(legacy_base_url)
        self.timeout = timeout
        self.max_retries = max_retries
        self._owns_http = http_client is None
        self._http = http_client or httpx.Client(
            timeout=httpx.Timeout(timeout),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": f"qubrid-python/{__version__}",
            },
        )
        completions = ChatCompletionsSync(self._http, self.base_url, max_retries)
        self.chat = _ChatSync(self._http, self.base_url, max_retries)
        # Reuse the same completions wiring for vision/OCR thin wrappers.
        self.vision = VisionSync(completions)
        self.ocr = OCRSync(completions)
        self.audio = AudioSync(self._http, self.base_url, self.legacy_base_url, max_retries)
        self.embeddings = EmbeddingsSync(self._http, self.base_url, max_retries)
        self.images = ImagesSync(self._http, self.base_url, max_retries)
        self.videos = VideosSync(self._http, self.base_url, max_retries)

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> QubridClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class AsyncQubridClient:
    """Asynchronous Qubrid client (same resources, ``await``-able methods).

    Example:
        >>> async with AsyncQubridClient() as client:
        ...     resp = await client.chat.completions.create(...)
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        legacy_base_url: str = DEFAULT_LEGACY_BASE_URL,
        timeout: float = 60.0,
        max_retries: int = 2,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.api_key = resolve_api_key(api_key)
        self.base_url = normalize_base_url(base_url)
        self.legacy_base_url = normalize_base_url(legacy_base_url)
        self.timeout = timeout
        self.max_retries = max_retries
        self._owns_http = http_client is None
        self._http = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": f"qubrid-python/{__version__}",
            },
        )
        completions = ChatCompletionsAsync(self._http, self.base_url, max_retries)
        self.chat = _ChatAsync(self._http, self.base_url, max_retries)
        self.vision = VisionAsync(completions)
        self.ocr = OCRAsync(completions)
        self.audio = AudioAsync(self._http, self.base_url, self.legacy_base_url, max_retries)
        self.embeddings = EmbeddingsAsync(self._http, self.base_url, max_retries)
        self.images = ImagesAsync(self._http, self.base_url, max_retries)
        self.videos = VideosAsync(self._http, self.base_url, max_retries)

    async def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> AsyncQubridClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
