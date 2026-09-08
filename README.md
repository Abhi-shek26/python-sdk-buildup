# Qubrid Python SDK — `qubrid`

First-party, standalone Python client for [Qubrid AI](https://platform.qubrid.com) GPU-accelerated inference. No `openai` dependency (`httpx` + `pydantic` only).

**Docs truth:** endpoint shapes verified against [`QubridAI-Inc/docs`](https://github.com/QubridAI-Inc/docs) (`api-reference/*/openapi.json`) and runnable payloads in [`QubridAI-Inc/qubrid-examples`](https://github.com/QubridAI-Inc/qubrid-examples).

| Resource | Endpoint | Base |
|---|---|---|
| `client.chat.completions` | `POST /chat/completions` (OpenAI-compatible, `stream` supported) | `…/v1` |
| `client.vision` | same `POST /chat/completions` (image_url parts) | `…/v1` |
| `client.ocr` | same `POST /chat/completions` (default `tencent/HunyuanOCR`, live-verified) | `…/v1` |
| `client.audio.speech` (TTS) | `POST /audio/generations` `{model,text,voice,language_type}` | `…/v1` |
| `client.audio.transcriptions` (STT) | `POST /audio/transcribe` multipart `{file,model}` | `…/api/v1/qubridai` (legacy — auto-routed) |
| `client.images` | `POST /images/generations`, `POST /images/edits` | `…/v1` |
| `client.videos` | `POST /videos/generations` | `…/v1` |
| `client.embeddings` | `POST /embeddings` — ⚠️ **EXPERIMENTAL**, no Qubrid OpenAPI spec | `…/v1` |

**Not in the SDK (no documented REST API — UI-only):** RAG (beta upload + Chat UI), no-code Fine-Tuning studio, GPU Instances/Clusters/Bare-Metal provisioning, models listing (`GET /models`). Model IDs are plain `str` (catalog of 100+ changes often — never an enum).

## Live verification status (smoke-tested 2026-09-08)

Confirmed working end-to-end: chat (+ streaming), vision, OCR (default `tencent/HunyuanOCR`), STT.

- **TTS — unverified:** endpoint reachable, shape accepted, but the safety filter rejected every tested input (`400 content_policy_violation`), including the docs' own example sentence.
- **Images / video — currently unroutable:** `POST /v1/images/generations` returns `405 Method Not Allowed` despite matching the OpenAPI spec exactly; `images.edit` / `videos.generate` untested (same family, deliberately not probed). Code retained per-spec; treat as unstable.
- **Embeddings — doesn't exist:** `POST /v1/embeddings` returns 404 with an empty body. The method is kept as an experimental stub only.

## Install

```bash
pip install qubrid
# or from source:
pip install -e ".[dev]"
```

Requires Python 3.9+, `QUBRID_API_KEY` from https://platform.qubrid.com/api-keys:

```bash
export QUBRID_API_KEY="qb_..."
```

## Quickstart

```python
from qubrid import QubridClient

client = QubridClient()  # reads QUBRID_API_KEY

resp = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Summarize this ticket into next steps."}],
    max_tokens=4096,
    temperature=0.7,
)
print(resp.content)
print(resp.usage.total_tokens)

# Streaming
for chunk in client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3.2",
    messages=[{"role": "user", "content": "Write a haiku about GPUs."}],
    stream=True,
):
    print(chunk.delta, end="", flush=True)

# Async
import asyncio
from qubrid import AsyncQubridClient

async def main():
    async with AsyncQubridClient() as c:
        r = await c.chat.completions.create(
            model="openai/gpt-oss-120b", messages=[{"role": "user", "content": "Hi"}])
        print(r.content)

asyncio.run(main())
```

## One example per category

```python
# Vision (Qwen/Qwen3-VL-Plus + image_url parts)
v = client.vision.analyze(
    prompt="What is in this image?",
    image_urls="https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
)
print(v.content)

# OCR (default tencent/HunyuanOCR — live-verified; examples' Hunyuan/Hunyuan-OCR-1B 404s)
o = client.ocr.extract(image_urls="https://example.com/receipt.jpg")
print(o.content)

# TTS → hosted mp3 URL
s = client.audio.speech.create(text="Today is a wonderful day to build something people love!",
                               voice="Cherry", language_type="Auto")
print(s.url)

# STT (multipart upload, auto-routed to legacy prefix)
t = client.audio.transcriptions.create(model="openai/whisper-large-v3", file="clip.wav")
print(t.text)

# Images
img = client.images.generate(model="p-image", prompt="a lighthouse at dusk",
                             aspect_ratio="16:9", output_format="webp")
print(img.data[0].url)
edit = client.images.edit(model="p-image-edit", prompt="add a red flag",
                          image="https://example.com/in.png")

# Video (only documented model: p-video, ≤10s)
vid = client.videos.generate(model="p-video", prompt="ocean waves", duration=5)
print(vid.url)

# Embeddings — EXPERIMENTAL passthrough (no Qubrid spec; may change)
e = client.embeddings.create(model="BAAI/bge-large-en-v1.5", input="hello world")
print(e.data[0].embedding[:4])
```

More in [`examples/`](examples/).

## Configuration

```python
QubridClient(
    api_key="qb_...",              # or QUBRID_API_KEY env
    base_url="https://platform.qubrid.com/v1",                 # override canonical
    legacy_base_url="https://platform.qubrid.com/api/v1/qubridai",  # STT only
    timeout=60.0,                  # seconds; per-call `timeout=` also supported
    max_retries=2,                 # retry-with-backoff on 429/5xx
)
```

## Errors

```python
from qubrid import QubridError, AuthenticationError, RateLimitError

try:
    client.chat.completions.create(model="m", messages=[])
except AuthenticationError:
    ...  # bad/missing key (401)
except RateLimitError as e:
    print(e.status_code, e.code, e.request_id, e.body)
except QubridError as e:
    ...  # base class: InvalidRequestError, NotFoundError,
         # PermissionDeniedError, QuotaExceededError, ServerError, APIConnectionError
```

## Docs & flags

- STT lives on the legacy prefix — the client handles it; only override `legacy_base_url` for staging/mocks.
- Live-verified 2026-09-08: runnable examples' `Hunyuan/Hunyuan-OCR-1B` returns 404 `model_not_found`; the working id is `tencent/HunyuanOCR` (the SDK default).
- Video quickstart JS reads `data[0].video_url` while the spec says `url` — both accepted (`VideoData.resolved_url`).
- Image-edit has no OpenAPI file (quickstarts only); local files → multipart, remote URLs → JSON `images:[...]`.
- Anything under `api-reference/endpoint/` (`/plants`) is a Mintlify template placeholder, not a real API.

## Known environment quirks

- **Garbled unicode on Windows terminals:** model output containing emoji, accented characters, or other non-ASCII text may render as `�` in default Windows consoles (e.g. `cmd.exe` / PowerShell with a legacy code page). This is a console-encoding artifact, not an SDK bug — the returned strings are correct. If affected, run with `PYTHONIOENCODING=utf-8` or use a UTF-8-capable terminal (e.g. Windows Terminal).
