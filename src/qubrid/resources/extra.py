"""Embeddings, images, and video resources.

.. warning::
    **Embeddings are EXPERIMENTAL.** Qubrid's docs list an embedding model
    (``BAAI/bge-large-en-v1.5`` in ``Serverless Models.mdx``) but define no
    embeddings REST endpoint. This resource is an OpenAI-compatible
    passthrough (``POST {base}/embeddings`` with ``{model, input}``) per the
    user's explicit choice. If Qubrid documents a real endpoint, this will be
    updated — expect the shape to change.

Images: ``POST {base}/images/generations`` (``api-reference/image/openapi.json``).
Image edits: ``POST {base}/images/edits`` (quickstarts only, no OpenAPI file —
multipart for local files, JSON ``images:[urls]`` for remote).
Video: ``POST {base}/videos/generations`` (``api-reference/video/openapi.json``).
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import httpx

from .._http import raise_for_response, request_async, request_sync
from .._types import EmbeddingResponse, ImageEditResult, ImageGeneration, VideoGeneration


class EmbeddingsSync:
    """Synchronous ``client.embeddings`` namespace (EXPERIMENTAL, see module doc)."""

    def __init__(self, http: httpx.Client, base_url: str, max_retries: int) -> None:
        self._http = http
        self._base_url = base_url
        self._max_retries = max_retries

    def create(
        self,
        *,
        model: str,
        input: Union[str, List[str], List[int], List[List[int]]],
        encoding_format: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> EmbeddingResponse:
        """Create embeddings (OpenAI-compatible passthrough, EXPERIMENTAL)."""
        payload: Dict[str, Any] = {"model": model, "input": input}
        if encoding_format is not None:
            payload["encoding_format"] = encoding_format
        if extra_body:
            payload.update(extra_body)
        kwargs: Dict[str, Any] = {"json": payload}
        if timeout is not None:
            kwargs["timeout"] = timeout
        resp = request_sync(
            self._http,
            "POST",
            f"{self._base_url}/embeddings",
            max_retries=self._max_retries,
            **kwargs,
        )
        return EmbeddingResponse.model_validate(raise_for_response(resp))


class EmbeddingsAsync:
    """Asynchronous ``client.embeddings`` namespace (EXPERIMENTAL, see module doc)."""

    def __init__(self, http: httpx.AsyncClient, base_url: str, max_retries: int) -> None:
        self._http = http
        self._base_url = base_url
        self._max_retries = max_retries

    async def create(
        self,
        *,
        model: str,
        input: Union[str, List[str], List[int], List[List[int]]],
        encoding_format: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> EmbeddingResponse:
        """Async variant of :meth:`EmbeddingsSync.create` (EXPERIMENTAL)."""
        payload: Dict[str, Any] = {"model": model, "input": input}
        if encoding_format is not None:
            payload["encoding_format"] = encoding_format
        if extra_body:
            payload.update(extra_body)
        kwargs: Dict[str, Any] = {"json": payload}
        if timeout is not None:
            kwargs["timeout"] = timeout
        resp = await request_async(
            self._http,
            "POST",
            f"{self._base_url}/embeddings",
            max_retries=self._max_retries,
            **kwargs,
        )
        return EmbeddingResponse.model_validate(raise_for_response(resp))


ImageInput = Union[str, os.PathLike, BinaryIO, Tuple[str, bytes]]


def _read_image_bytes(image: ImageInput) -> Optional[Tuple[str, bytes]]:
    """Return ``(filename, bytes)`` for local files, ``None`` for remote URLs."""
    if isinstance(image, tuple):
        name, data = image
        return name, data
    if isinstance(image, (str, os.PathLike)):
        s = str(image)
        if s.startswith(("http://", "https://")):
            return None
        p = Path(s)
        return p.name, p.read_bytes()
    if hasattr(image, "read"):
        data = image.read()  # type: ignore[union-attr]
        if isinstance(data, str):
            data = data.encode()
        name = str(getattr(image, "name", "image.png")).split("/")[-1]
        return name, data
    raise TypeError("image must be a URL, local path, file object, or (filename, bytes) tuple")


class ImagesSync:
    """Synchronous ``client.images`` namespace."""

    def __init__(self, http: httpx.Client, base_url: str, max_retries: int) -> None:
        self._http = http
        self._base_url = base_url
        self._max_retries = max_retries

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        aspect_ratio: Optional[str] = None,
        image_size: Optional[str] = None,
        output_format: Optional[str] = None,
        output_quality: Optional[int] = None,
        seed: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> ImageGeneration:
        """Generate image(s) from a text prompt.

        .. warning:: As of 2026-09-08, this endpoint returns ``405 Method
            Not Allowed`` on the live Qubrid deployment despite matching the
            documented OpenAPI spec (``POST /v1/images/generations``) exactly.
            Request shape is unverified against a real success response.
            Treat as unstable until confirmed working.

        Args:
            model: e.g. ``"p-image"``, ``"Tongyi-MAI/Z-Image-Turbo"``,
                ``"stabilityai/stable-diffusion-3.5-large"``.
            prompt: Text description of the desired image.
            aspect_ratio: e.g. ``"16:9"``. image_size: e.g. ``"optimize_for_quality"``.
            output_format: e.g. ``"webp"``. output_quality: 0-100.
        """
        payload: Dict[str, Any] = {"model": model, "prompt": prompt}
        for k, v in (
            ("aspect_ratio", aspect_ratio),
            ("image_size", image_size),
            ("output_format", output_format),
            ("output_quality", output_quality),
            ("seed", seed),
        ):
            if v is not None:
                payload[k] = v
        if extra_body:
            payload.update(extra_body)
        kwargs: Dict[str, Any] = {"json": payload}
        if timeout is not None:
            kwargs["timeout"] = timeout
        resp = request_sync(
            self._http,
            "POST",
            f"{self._base_url}/images/generations",
            max_retries=self._max_retries,
            **kwargs,
        )
        return ImageGeneration.model_validate(raise_for_response(resp))

    def edit(
        self,
        *,
        model: str = "p-image-edit",
        prompt: str,
        image: Union[ImageInput, List[ImageInput], None] = None,
        aspect_ratio: Optional[str] = None,
        seed: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> ImageEditResult:
        """Edit image(s). Local files → multipart; remote URLs → JSON.

        .. warning:: As of 2026-09-08, the sibling ``images/generations``
            endpoint returns ``405 Method Not Allowed`` on the live deployment
            despite matching the spec exactly; this endpoint is untested live
            and likely affected by the same server-side routing issue. Request
            shape is unverified against a real success response.

        .. note:: No OpenAPI spec exists for this endpoint (quickstarts only:
            ``P-Image Edit.mdx``). Field names follow those quickstarts.
        """
        images: List[ImageInput] = []
        if image is not None:
            images = image if isinstance(image, list) else [image]
        local = [(i, _read_image_bytes(i)) for i in images]
        kwargs: Dict[str, Any] = {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        if any(b is not None for _, b in local):
            # Multipart: model/prompt fields + file parts.
            form: Dict[str, Any] = {"model": model, "prompt": prompt}
            if aspect_ratio is not None:
                form["aspect_ratio"] = aspect_ratio
            if seed is not None:
                form["seed"] = str(seed)
            files = []
            for (_, blob), orig in zip(local, images):
                if blob is None:  # mixed local+remote not supported; skip remote
                    continue
                fname, data = blob
                files.append(("image", (fname, io.BytesIO(data), "application/octet-stream")))
            kwargs.update({"data": form, "files": files or None})
            resp = request_sync(
                self._http,
                "POST",
                f"{self._base_url}/images/edits",
                max_retries=self._max_retries,
                **kwargs,
            )
        else:
            payload: Dict[str, Any] = {"model": model, "prompt": prompt}
            urls = [str(i) for i in images]
            if urls:
                payload["images"] = urls
            if aspect_ratio is not None:
                payload["aspect_ratio"] = aspect_ratio
            if seed is not None:
                payload["seed"] = seed
            kwargs["json"] = payload
            resp = request_sync(
                self._http,
                "POST",
                f"{self._base_url}/images/edits",
                max_retries=self._max_retries,
                **kwargs,
            )
        return ImageEditResult.model_validate(raise_for_response(resp))


class ImagesAsync:
    """Asynchronous ``client.images`` namespace."""

    def __init__(self, http: httpx.AsyncClient, base_url: str, max_retries: int) -> None:
        self._http = http
        self._base_url = base_url
        self._max_retries = max_retries

    async def generate(
        self,
        *,
        model: str,
        prompt: str,
        aspect_ratio: Optional[str] = None,
        image_size: Optional[str] = None,
        output_format: Optional[str] = None,
        output_quality: Optional[int] = None,
        seed: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> ImageGeneration:
        """Async variant of :meth:`ImagesSync.generate`."""
        payload: Dict[str, Any] = {"model": model, "prompt": prompt}
        for k, v in (
            ("aspect_ratio", aspect_ratio),
            ("image_size", image_size),
            ("output_format", output_format),
            ("output_quality", output_quality),
            ("seed", seed),
        ):
            if v is not None:
                payload[k] = v
        if extra_body:
            payload.update(extra_body)
        kwargs: Dict[str, Any] = {"json": payload}
        if timeout is not None:
            kwargs["timeout"] = timeout
        resp = await request_async(
            self._http,
            "POST",
            f"{self._base_url}/images/generations",
            max_retries=self._max_retries,
            **kwargs,
        )
        return ImageGeneration.model_validate(raise_for_response(resp))

    async def edit(
        self,
        *,
        model: str = "p-image-edit",
        prompt: str,
        image: Union[ImageInput, List[ImageInput], None] = None,
        aspect_ratio: Optional[str] = None,
        seed: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> ImageEditResult:
        """Async variant of :meth:`ImagesSync.edit`."""
        images: List[ImageInput] = []
        if image is not None:
            images = image if isinstance(image, list) else [image]
        local = [(i, _read_image_bytes(i)) for i in images]
        kwargs: Dict[str, Any] = {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        if any(b is not None for _, b in local):
            form: Dict[str, Any] = {"model": model, "prompt": prompt}
            if aspect_ratio is not None:
                form["aspect_ratio"] = aspect_ratio
            if seed is not None:
                form["seed"] = str(seed)
            files = []
            for _, blob in local:
                if blob is None:
                    continue
                fname, data = blob
                files.append(("image", (fname, io.BytesIO(data), "application/octet-stream")))
            kwargs.update({"data": form, "files": files or None})
            resp = await request_async(
                self._http,
                "POST",
                f"{self._base_url}/images/edits",
                max_retries=self._max_retries,
                **kwargs,
            )
        else:
            payload: Dict[str, Any] = {"model": model, "prompt": prompt}
            urls = [str(i) for i in images]
            if urls:
                payload["images"] = urls
            if aspect_ratio is not None:
                payload["aspect_ratio"] = aspect_ratio
            if seed is not None:
                payload["seed"] = seed
            kwargs["json"] = payload
            resp = await request_async(
                self._http,
                "POST",
                f"{self._base_url}/images/edits",
                max_retries=self._max_retries,
                **kwargs,
            )
        return ImageEditResult.model_validate(raise_for_response(resp))


class VideosSync:
    """Synchronous ``client.videos`` namespace."""

    def __init__(self, http: httpx.Client, base_url: str, max_retries: int) -> None:
        self._http = http
        self._base_url = base_url
        self._max_retries = max_retries

    def generate(
        self,
        *,
        model: str = "p-video",
        prompt: Optional[str] = None,
        duration: Optional[int] = None,
        resolution: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        image: Optional[str] = None,
        audio: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> VideoGeneration:
        """Generate video (only documented model is ``p-video``; ≤10s).

        .. warning:: As of 2026-09-08, untested live and expected to be
            affected by the same server-side routing issue as images: the
            sibling ``POST /v1/images/generations`` returns ``405 Method Not
            Allowed`` despite matching its spec exactly. Deliberately not
            smoke-tested to avoid spending a call re-confirming that. Request
            shape is unverified against a real success response.

        .. note:: One quickstart reads ``data[0].video_url`` while the OpenAPI
            spec says ``url`` — the SDK exposes both via
            :attr:`VideoData.resolved_url`.
        """
        payload: Dict[str, Any] = {"model": model}
        for k, v in (
            ("prompt", prompt),
            ("duration", duration),
            ("resolution", resolution),
            ("aspect_ratio", aspect_ratio),
            ("image", image),
            ("audio", audio),
        ):
            if v is not None:
                payload[k] = v
        if extra_body:
            payload.update(extra_body)
        kwargs: Dict[str, Any] = {"json": payload}
        if timeout is not None:
            kwargs["timeout"] = timeout
        resp = request_sync(
            self._http,
            "POST",
            f"{self._base_url}/videos/generations",
            max_retries=self._max_retries,
            **kwargs,
        )
        return VideoGeneration.model_validate(raise_for_response(resp))


class VideosAsync:
    """Asynchronous ``client.videos`` namespace."""

    def __init__(self, http: httpx.AsyncClient, base_url: str, max_retries: int) -> None:
        self._http = http
        self._base_url = base_url
        self._max_retries = max_retries

    async def generate(
        self,
        *,
        model: str = "p-video",
        prompt: Optional[str] = None,
        duration: Optional[int] = None,
        resolution: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        image: Optional[str] = None,
        audio: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> VideoGeneration:
        """Async variant of :meth:`VideosSync.generate`."""
        payload: Dict[str, Any] = {"model": model}
        for k, v in (
            ("prompt", prompt),
            ("duration", duration),
            ("resolution", resolution),
            ("aspect_ratio", aspect_ratio),
            ("image", image),
            ("audio", audio),
        ):
            if v is not None:
                payload[k] = v
        if extra_body:
            payload.update(extra_body)
        kwargs: Dict[str, Any] = {"json": payload}
        if timeout is not None:
            kwargs["timeout"] = timeout
        resp = await request_async(
            self._http,
            "POST",
            f"{self._base_url}/videos/generations",
            max_retries=self._max_retries,
            **kwargs,
        )
        return VideoGeneration.model_validate(raise_for_response(resp))
