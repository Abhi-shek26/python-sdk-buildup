"""Pydantic request/response models mirroring Qubrid's OpenAPI specs.

Sources of truth:
- ``qubrid-docs/api-reference/{chat,vision,ocr,voice,image,video}/openapi.json``
- ``qubrid-examples/{chat,vision,ocr,voice}/`` sample payloads.

Model IDs are plain ``str`` (catalog changes often — never an enum).
All response models use ``extra="allow"`` where the API may add fields
(e.g. ``x_metrics``) so the SDK doesn't break on additions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


# ---------------------------------------------------------------- chat ----
class TextPart(_Lenient):
    type: Literal["text"] = "text"
    text: str


class ImageUrl(_Lenient):
    url: str


class ImagePart(_Lenient):
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl


ContentPart = Union[TextPart, ImagePart]


class ChatMessage(_Lenient):
    role: str = Field(description="system | user | assistant")
    content: Union[str, List[Union[TextPart, ImagePart, Dict[str, Any]]]]


class ChatUsage(_Lenient):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class ChatMetrics(_Lenient):
    ttft_seconds: Optional[float] = None
    tps: Optional[float] = None
    total_time_seconds: Optional[float] = None


class ChatChoiceMessage(_Lenient):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatChoice(_Lenient):
    index: int = 0
    message: ChatChoiceMessage = Field(default_factory=ChatChoiceMessage)
    finish_reason: Optional[str] = None


class ChatCompletion(_Lenient):
    """Non-streaming ``POST /chat/completions`` response."""

    id: Optional[str] = None
    object: Optional[str] = None
    created: Optional[int] = None
    model: Optional[str] = None
    choices: List[ChatChoice] = Field(default_factory=list)
    usage: Optional[ChatUsage] = None
    x_metrics: Optional[ChatMetrics] = None

    @property
    def content(self) -> Optional[str]:
        """First choice's message content (convenience accessor)."""
        if self.choices:
            return self.choices[0].message.content
        return None


class ChatDelta(_Lenient):
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(_Lenient):
    index: int = 0
    delta: ChatDelta = Field(default_factory=ChatDelta)
    finish_reason: Optional[str] = None


class ChatCompletionChunk(_Lenient):
    """One SSE ``data:`` event from a streamed chat completion."""

    id: Optional[str] = None
    object: Optional[str] = None
    created: Optional[int] = None
    model: Optional[str] = None
    choices: List[StreamChoice] = Field(default_factory=list)

    @property
    def delta(self) -> Optional[str]:
        if self.choices:
            return self.choices[0].delta.content
        return None


# ---------------------------------------------------------------- voice (TTS/STT) ----
class SpeechData(_Lenient):
    url: str


class SpeechGeneration(_Lenient):
    """``POST /audio/generations`` response (TTS audio hosted at a URL)."""

    created: Optional[int] = None
    data: List[SpeechData] = Field(default_factory=list)

    @property
    def url(self) -> Optional[str]:
        return self.data[0].url if self.data else None


class Transcription(_Lenient):
    """``POST /audio/transcribe`` (legacy prefix) response."""

    text: str
    language: Optional[str] = None


# ---------------------------------------------------------------- embeddings ----
class EmbeddingItem(_Lenient):
    object: Optional[str] = None
    index: int = 0
    embedding: List[float] = Field(default_factory=list)


class EmbeddingUsage(_Lenient):
    prompt_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class EmbeddingResponse(_Lenient):
    """``POST /embeddings`` — EXPERIMENTAL, no Qubrid OpenAPI spec (see README)."""

    object: Optional[str] = None
    data: List[EmbeddingItem] = Field(default_factory=list)
    model: Optional[str] = None
    usage: Optional[EmbeddingUsage] = None


# ---------------------------------------------------------------- images ----
class ImageData(_Lenient):
    url: Optional[str] = None
    b64_json: Optional[str] = None


class ImageGeneration(_Lenient):
    """``POST /images/generations`` response."""

    created: Optional[int] = None
    data: List[ImageData] = Field(default_factory=list)


class ImageEditResult(ImageGeneration):
    """``POST /images/edits`` response (same envelope as generations)."""


# ---------------------------------------------------------------- video ----
class VideoData(_Lenient):
    url: Optional[str] = None
    # Some quickstart JS reads `video_url`; accept both.
    video_url: Optional[str] = None

    @property
    def resolved_url(self) -> Optional[str]:
        return self.url or self.video_url


class VideoGeneration(_Lenient):
    """``POST /videos/generations`` response."""

    created: Optional[int] = None
    data: List[VideoData] = Field(default_factory=list)

    @property
    def url(self) -> Optional[str]:
        return self.data[0].resolved_url if self.data else None
