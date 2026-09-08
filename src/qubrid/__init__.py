"""Qubrid AI Python SDK — GPU-accelerated inference (chat, vision, OCR, audio, images, video).

Standalone client (``httpx`` + ``pydantic`` only, no ``openai`` dependency).
Get a key at https://platform.qubrid.com/api-keys and set ``QUBRID_API_KEY``.
"""

from __future__ import annotations

from .__version__ import __version__
from ._types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatMessage,
    EmbeddingResponse,
    ImageGeneration,
    SpeechGeneration,
    Transcription,
    VideoGeneration,
)
from .client import AsyncQubridClient, QubridClient
from .exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    InvalidRequestError,
    NotFoundError,
    PermissionDeniedError,
    QubridError,
    QuotaExceededError,
    RateLimitError,
    ServerError,
)

__all__ = [
    "APIConnectionError",
    "APIError",
    "AsyncQubridClient",
    "AuthenticationError",
    "ChatCompletion",
    "ChatCompletionChunk",
    "ChatMessage",
    "EmbeddingResponse",
    "ImageGeneration",
    "InvalidRequestError",
    "NotFoundError",
    "PermissionDeniedError",
    "QubridClient",
    "QubridError",
    "QuotaExceededError",
    "RateLimitError",
    "ServerError",
    "SpeechGeneration",
    "Transcription",
    "VideoGeneration",
    "__version__",
]
