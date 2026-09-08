"""Internal helpers: env/auth, URL normalisation, retry timing."""

from __future__ import annotations

import os
from typing import Optional

from .exceptions import AuthenticationError

DEFAULT_BASE_URL = "https://platform.qubrid.com/v1"
DEFAULT_LEGACY_BASE_URL = "https://platform.qubrid.com/api/v1/qubridai"
API_KEY_ENV_VAR = "QUBRID_API_KEY"


def resolve_api_key(api_key: Optional[str]) -> str:
    """Return explicit key or ``QUBRID_API_KEY`` env var, else raise."""
    key = api_key or os.environ.get(API_KEY_ENV_VAR)
    if not key:
        raise AuthenticationError(
            f"No API key provided. Pass api_key=... or set the {API_KEY_ENV_VAR} "
            "environment variable (get one at https://platform.qubrid.com/api-keys)."
        )
    return key


def normalize_base_url(url: str) -> str:
    """Strip trailing slashes so ``base + path`` joins are predictable."""
    return url.rstrip("/")


def retry_delay(attempt: int, retry_after: Optional[float] = None) -> float:
    """Exponential backoff (0.5s, 1s, 2s, ...) capped by server Retry-After."""
    if retry_after is not None and retry_after >= 0:
        return min(retry_after, 30.0)
    return min(0.5 * (2**attempt), 8.0)


def parse_retry_after(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def is_retryable(status_code: int) -> bool:
    return status_code in RETRYABLE_STATUS
