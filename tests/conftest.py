"""Shared fixtures: clients wired to respx-mocked httpx transports."""

from __future__ import annotations

import pytest

from qubrid import AsyncQubridClient, QubridClient

BASE = "https://platform.qubrid.com/v1"
LEGACY = "https://platform.qubrid.com/api/v1/qubridai"


@pytest.fixture
def sync_client() -> QubridClient:
    return QubridClient(api_key="test-key", max_retries=0)


@pytest.fixture
def async_client() -> AsyncQubridClient:
    return AsyncQubridClient(api_key="test-key", max_retries=0)


@pytest.fixture
def chat_ok() -> dict:
    return {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1710000000,
        "model": "openai/gpt-oss-120b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


@pytest.fixture
def api_error() -> dict:
    return {
        "error": {
            "message": "Missing required parameter: 'model'.",
            "type": "invalid_request_error",
            "code": "invalid_request",
            "param": "model",
            "request_id": "req-1",
        }
    }
