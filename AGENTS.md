# AGENTS.md

First-party Python SDK for Qubrid AI inference (`qubrid` on PyPI). Standalone: `httpx` + `pydantic` only — never add an `openai` dependency.

## Commands

- Install (editable + dev): `pip install -e ".[dev]"`
- Tests: `python -m pytest tests/ -q` (27 tests, `respx`-mocked HTTP — no network/key needed)
- Lint: `ruff check src tests` · Format: `ruff format src tests`
- `Optional[...]` is deliberate (py39 runtime compat for pydantic); `UP006/7/35/45` are silenced in `pyproject.toml` — do not "modernize" to `X | None`.

## Architecture

- `src/qubrid/client.py` — `QubridClient` / `AsyncQubridClient` (env `QUBRID_API_KEY`, `base_url` + `legacy_base_url`, `timeout`, `max_retries`). Owns one `httpx` client; resources are thin wrappers over it.
- `src/qubrid/_http.py` — retry-with-backoff on 429/5xx (honors `Retry-After`), error mapping, SSE `data:` parsing. Sync/async parity lives here.
- `src/qubrid/_types.py` — pydantic models mirroring `QubridAI-Inc/docs` `api-reference/*/openapi.json`. Model IDs are plain `str` (catalog churns).
- Resources: `chat.py` (`POST /chat/completions`, streaming + not), `vision_ocr.py` (same path, docs-example defaults), `audio.py` (TTS on `/v1`, STT multipart auto-routed to legacy `/api/v1/qubridai`), `extra.py` (embeddings/images/video).
- `src/qubrid/exceptions.py` — `QubridError` + typed subclasses carrying `status_code/code/type/param/request_id/body`.

## Gotchas (verified vs docs repo + examples repo, 2026-09-08)

- Two live prefixes: canonical `https://platform.qubrid.com/v1`, legacy `.../api/v1/qubridai` for STT **only**. Do not mix field names across them.
- Embeddings has **no** Qubrid OpenAPI spec — `embeddings.create` is an explicitly experimental OpenAI-compatible passthrough (user-approved). Flag shape changes.
- Image-edits has no OpenAPI file (quickstarts only): local files → multipart, remote URLs → JSON `images:[...]`. Video accepts both `url` and `video_url` (`resolved_url`).
- No REST API exists for RAG / fine-tuning / GPU provisioning / models listing (UI-only) — do not invent endpoints; raise/flag instead.
- `api-reference/endpoint/` (`/plants`) is a Mintlify template placeholder, not real.
- OCR default `tencent/HunyuanOCR` (live-verified 2026-09-08; examples' `Hunyuan/Hunyuan-OCR-1B` 404s `model_not_found`).
