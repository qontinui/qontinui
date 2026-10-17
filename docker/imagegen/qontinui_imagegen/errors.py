"""The facade's one refusal envelope (plan Phase 10, "Admission").

Every answer that is not an image is OpenAI's error shape, extended so a client can branch on
``error.type`` and never on the HTTP status alone (llama-swap itself also answers 429, as a
plain-text "Too many requests" at v217, when its per-model concurrency limit is hit)::

    {"error": {"message": ..., "type": "busy" | "deferred" | "disabled" | "failed"
                                        | "invalid_request_error",
               "code": ..., "param": ..., "retry_after_s": <seconds or null>, ...}}

``busy`` (a full queue, or the pre-load VRAM check) is HTTP 503; ``deferred`` (bulk work
outside the bulk window) is HTTP 429; ``disabled`` (the machine switch) is HTTP 503 with no
retry time, because only an operator turns it back on; ``failed`` is HTTP 500, and
``invalid_request_error`` HTTP 400. A refusal that names a retry time also carries it as a
``Retry-After`` header, rounded up to whole seconds.
"""

from __future__ import annotations

import math
from typing import Any


class FacadeError(Exception):
    """A typed answer that is not an image."""

    def __init__(
        self,
        status: int,
        error_type: str,
        code: str,
        message: str,
        *,
        retry_after_s: float | None = None,
        param: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.error_type = error_type
        self.code = code
        self.message = message
        self.retry_after_s = retry_after_s
        self.param = param
        self.extra = dict(extra or {})

    def body(self) -> dict[str, Any]:
        error: dict[str, Any] = {
            "message": self.message,
            "type": self.error_type,
            "code": self.code,
            "param": self.param,
            "retry_after_s": (
                None if self.retry_after_s is None else math.ceil(self.retry_after_s)
            ),
        }
        error.update(self.extra)
        return {"error": error}

    def headers(self) -> dict[str, str]:
        if self.retry_after_s is None:
            return {}
        return {"Retry-After": str(max(1, math.ceil(self.retry_after_s)))}


def busy(code: str, message: str, retry_after_s: float, **extra: Any) -> FacadeError:
    return FacadeError(503, "busy", code, message, retry_after_s=retry_after_s, extra=extra)


def deferred(message: str, retry_after_s: float) -> FacadeError:
    return FacadeError(429, "deferred", "outside_bulk_window", message, retry_after_s=retry_after_s)


def disabled(code: str, message: str) -> FacadeError:
    return FacadeError(503, "disabled", code, message)


def failed(code: str, message: str, **extra: Any) -> FacadeError:
    return FacadeError(500, "failed", code, message, extra=extra)


def invalid(code: str, message: str, param: str | None = None) -> FacadeError:
    return FacadeError(400, "invalid_request_error", code, message, param=param)
