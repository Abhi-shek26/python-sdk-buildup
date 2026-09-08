"""Custom exception hierarchy for the Qubrid SDK.

All errors raised from non-2xx API responses carry the parsed error body
(``message``, ``type``, ``code``, ``param``, ``request_id``) plus the raw
``status_code`` and ``body`` so callers can log or branch on them.
"""

from __future__ import annotations

from typing import Any, Optional


class QubridError(Exception):
    """Base class for all Qubrid SDK errors."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        code: Optional[str] = None,
        type: Optional[str] = None,
        param: Optional[str] = None,
        request_id: Optional[str] = None,
        body: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.type = type
        self.param = param
        self.request_id = request_id
        self.body = body

    def __str__(self) -> str:
        parts = [self.message]
        if self.status_code is not None:
            parts.append(f"(status={self.status_code}")
            if self.code:
                parts.append(f" code={self.code}")
            if self.request_id:
                parts.append(f" request_id={self.request_id}")
            parts.append(")")
            # join without extra spaces: "msg (status=.. code=..)"
            s = parts[0] + " " + "".join(parts[1:])
            return s
        return self.message


class AuthenticationError(QubridError):
    """401 — missing/invalid ``QUBRID_API_KEY``."""


class PermissionDeniedError(QubridError):
    """403 — key valid but not allowed to access this resource."""


class NotFoundError(QubridError):
    """404 — unknown endpoint or model id."""


class InvalidRequestError(QubridError):
    """400/422/413 — malformed request, bad params, file too large."""


class QuotaExceededError(QubridError):
    """402 / ``insufficient_quota`` — out of credits."""


class RateLimitError(QubridError):
    """429 — slow down and retry (SDK retries these automatically)."""


class ServerError(QubridError):
    """5xx — Qubrid side failure (SDK retries these automatically)."""


class APIConnectionError(QubridError):
    """Transport-level failure (DNS, timeout, reset) after retries."""


# Backwards-compatible alias requested in the spec.
APIError = ServerError


def _extract_api_error(body: Any) -> dict:
    """Pull ``{message, type, code, param, request_id}`` out of a response body."""
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            return {
                "message": err.get("message"),
                "type": err.get("type"),
                "code": err.get("code"),
                "param": err.get("param"),
                "request_id": err.get("request_id"),
            }
        if isinstance(body.get("message"), str):
            return {
                "message": body.get("message"),
                "type": None,
                "code": None,
                "param": None,
                "request_id": body.get("request_id"),
            }
    return {"message": None, "type": None, "code": None, "param": None, "request_id": None}


def error_from_response(status_code: int, body: Any) -> QubridError:
    """Map an HTTP status + parsed body to the right exception class."""
    extracted = _extract_api_error(body)
    message = extracted["message"] or f"Qubrid API request failed (status {status_code})"
    err_type = extracted["type"] or ""
    code = extracted["code"]
    kwargs = {
        "status_code": status_code,
        "code": code,
        "type": extracted["type"],
        "param": extracted["param"],
        "request_id": extracted["request_id"],
        "body": body,
    }
    if status_code == 401:
        return AuthenticationError(message, **kwargs)
    if status_code == 403:
        return PermissionDeniedError(message, **kwargs)
    if status_code == 404:
        return NotFoundError(message, **kwargs)
    if status_code == 429:
        return RateLimitError(message, **kwargs)
    if status_code == 402 or code == "insufficient_quota" or err_type == "insufficient_quota":
        return QuotaExceededError(message, **kwargs)
    if status_code in (400, 413, 422):
        return InvalidRequestError(message, **kwargs)
    if 500 <= status_code <= 599:
        return ServerError(message, **kwargs)
    # Fallback for other 4xx: treat quota/type hints first, else invalid-request.
    if 400 <= status_code < 500:
        return InvalidRequestError(message, **kwargs)
    return ServerError(message, **kwargs)
