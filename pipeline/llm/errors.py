from __future__ import annotations

from typing import Any


class ProviderConfigurationError(ValueError):
    pass


class CostAccountingError(RuntimeError):
    pass


def is_payload_too_large(exc: BaseException) -> bool:
    for source in (exc, getattr(exc, "response", None)):
        status = getattr(source, "status_code", None)
        if status == 413 or status == "413":
            return True
    if getattr(exc, "error_type", None) == "payload_too_large":
        return True
    body: Any = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            return any(
                error.get(key) == "payload_too_large" for key in ("type", "code", "error_type")
            )
    return False
