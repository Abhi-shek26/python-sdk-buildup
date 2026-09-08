"""Vision + OCR resources — thin, typed wrappers over ``POST /chat/completions``.

Vision spec: ``qubrid-docs/api-reference/vision/openapi.json``,
example model ``Qwen/Qwen3-VL-Plus``.
OCR spec: ``qubrid-docs/api-reference/ocr/openapi.json``,
example model ``tencent/HunyuanOCR`` (live-verified; the examples'
``Hunyuan/Hunyuan-OCR-1B`` returns 404).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import httpx

from .chat import ChatCompletionsAsync, ChatCompletionsSync


def build_image_content(prompt: str, image_urls: Union[str, List[str]]) -> List[Dict[str, Any]]:
    """Build OpenAI-style ``content`` parts for one prompt + N images."""
    urls = [image_urls] if isinstance(image_urls, str) else list(image_urls)
    parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for u in urls:
        parts.append({"type": "image_url", "image_url": {"url": u}})
    return parts


class VisionSync:
    """Synchronous ``client.vision`` namespace."""

    def __init__(self, completions: ChatCompletionsSync) -> None:
        self._completions = completions

    def analyze(
        self,
        *,
        model: str = "Qwen/Qwen3-VL-Plus",
        prompt: str,
        image_urls: Union[str, List[str]],
        max_tokens: Optional[int] = 16384,
        temperature: Optional[float] = 0.1,
        top_p: Optional[float] = None,
        stream: bool = False,
        system_prompt: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Analyze image(s) with a vision model.

        Args:
            model: Vision model id (plain string, default ``Qwen/Qwen3-VL-Plus``).
            prompt: Question/instruction about the image(s).
            image_urls: One URL or list of URLs (publicly reachable).
            max_tokens: Docs example uses 16384.
            temperature: Docs example uses 0.1.
            stream: If True, yields :class:`ChatCompletionChunk` events.
            system_prompt: Optional system message prepended to the request.
        """
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": build_image_content(prompt, image_urls)})
        return self._completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
            timeout=timeout,
        )


class VisionAsync:
    """Asynchronous ``client.vision`` namespace."""

    def __init__(self, completions: ChatCompletionsAsync) -> None:
        self._completions = completions

    async def analyze(
        self,
        *,
        model: str = "Qwen/Qwen3-VL-Plus",
        prompt: str,
        image_urls: Union[str, List[str]],
        max_tokens: Optional[int] = 16384,
        temperature: Optional[float] = 0.1,
        top_p: Optional[float] = None,
        stream: bool = False,
        system_prompt: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Async variant of :meth:`VisionSync.analyze`."""
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": build_image_content(prompt, image_urls)})
        return await self._completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
            timeout=timeout,
        )


class OCRSync:
    """Synchronous ``client.ocr`` namespace."""

    def __init__(self, completions: ChatCompletionsSync) -> None:
        self._completions = completions

    def extract(
        self,
        *,
        image_urls: Union[str, List[str]],
        model: str = "tencent/HunyuanOCR",
        prompt: str = "Extract all text from this image verbatim.",
        max_tokens: Optional[int] = 8192,
        temperature: Optional[float] = 0.1,
        timeout: Optional[float] = None,
    ) -> Any:
        """Extract text from image(s) via an OCR model.

        Default is ``tencent/HunyuanOCR`` — the only OCR model id the live
        API accepts (verified 2026-09-08: ``Hunyuan/Hunyuan-OCR-1B`` from the
        runnable examples returns 404 ``model_not_found``). OCR examples use
        ``stream=False``.
        """
        messages = [{"role": "user", "content": build_image_content(prompt, image_urls)}]
        return self._completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            timeout=timeout,
        )


class OCRAsync:
    """Asynchronous ``client.ocr`` namespace."""

    def __init__(self, completions: ChatCompletionsAsync) -> None:
        self._completions = completions

    async def extract(
        self,
        *,
        image_urls: Union[str, List[str]],
        model: str = "tencent/HunyuanOCR",
        prompt: str = "Extract all text from this image verbatim.",
        max_tokens: Optional[int] = 8192,
        temperature: Optional[float] = 0.1,
        timeout: Optional[float] = None,
    ) -> Any:
        """Async variant of :meth:`OCRSync.extract`."""
        messages = [{"role": "user", "content": build_image_content(prompt, image_urls)}]
        return await self._completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            timeout=timeout,
        )


def _unused_httpx_reference() -> None:  # keep httpx import used for type-checkers
    assert httpx.Client is not None
