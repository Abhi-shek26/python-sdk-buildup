# Changelog

All notable changes to the `qubrid` package. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.0] — 2026-09-08

### Added
- Initial standalone SDK (`httpx` + `pydantic`, no `openai` dependency).
- `QubridClient` / `AsyncQubridClient` with `QUBRID_API_KEY` env fallback, `base_url` + `legacy_base_url` overrides, configurable timeout/retries.
- `chat.completions.create` (streaming SSE generator + non-streaming), live-verified vs `api-reference/chat/openapi.json`.
- `vision.analyze`, `ocr.extract` thin wrappers over `/chat/completions`, live-verified (OCR default corrected to `tencent/HunyuanOCR` — examples' `Hunyuan/Hunyuan-OCR-1B` 404s).
- `audio.transcriptions.create` (STT multipart, auto-routed to legacy `/api/v1/qubridai/audio/transcribe`), live-verified.
- `audio.speech.create` (TTS `POST /audio/generations`), `images.generate` / `images.edit`, `videos.generate` (accepts both `url` and `video_url`), `embeddings.create` — retained per-spec but **not working live** (see Fixed/Unverified below).
- Typed errors (`AuthenticationError`, `RateLimitError`, `InvalidRequestError`, … with `code`/`request_id`/`body`), retry-with-backoff on 429/5xx.
- Mocked test suite (`respx`), README quickstart + per-category examples.

### Live-verification findings (2026-09-08 smoke test, ~$0.0005 spend)
- Fixed: OCR default `Hunyuan/Hunyuan-OCR-1B` → `tencent/HunyuanOCR` (former is 404 `model_not_found` live; latter verified 200 with usage 1699/17).
- Unverified: TTS returns `400 content_policy_violation` on all tested inputs including the docs' own example sentence; `images/generations` returns `405 Method Not Allowed` despite exact spec match (`images.edit`, `videos.generate` deliberately untested, same family); `POST /v1/embeddings` is 404 with empty body (endpoint doesn't exist).

### Final verification (2026-09-08, v0.1.0 polish)
- Live-verified end-to-end against the real Qubrid API: chat (streaming + non-streaming), vision, OCR (corrected default `tencent/HunyuanOCR`), STT.
- Confirmed non-functional: embeddings (endpoint doesn't exist, 404 with empty body); images/video generation (`405 Method Not Allowed` despite exact OpenAPI spec match — likely account/region routing issue on Qubrid's side; siblings untested by decision); TTS (reachable, request shape accepted, but safety filter rejected all tested inputs including Qubrid's own documented example sentence).
- Clean-room install verified: fresh venv, non-editable `pip install`, import + live quickstart call succeeded from outside the source tree.
- Total live-testing spend across the full verification effort: ~$0.0006.
- Docs-only fixes: streaming example guards against `None` deltas; README notes Windows console unicode quirk (`PYTHONIOENCODING=utf-8`). No core SDK logic changed.
