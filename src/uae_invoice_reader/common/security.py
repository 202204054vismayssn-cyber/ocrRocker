"""Provide shared API-key authentication dependencies."""

from hmac import compare_digest

from fastapi import Depends
from fastapi.security import APIKeyHeader

from uae_invoice_reader.common.config import Settings, get_settings
from uae_invoice_reader.common.exceptions import AppError, ConfigurationError


# auto_error=False lets us return the same clear response for a missing or wrong key.
_savvy_api_key_header = APIKeyHeader(name="SAVVY_API_KEY", auto_error=False)
_admin_api_key_header = APIKeyHeader(name="ADMIN_API_KEY", auto_error=False)


def require_savvy_api_key(
    supplied_key: str | None = Depends(_savvy_api_key_header),
    settings: Settings = Depends(get_settings),
) -> None:
    """Allow extraction only when the request header matches the `.env` key (optional if unset/empty)."""

    if settings.savvy_api_key is None:
        return

    expected_key = settings.savvy_api_key.get_secret_value().strip()
    if not expected_key:
        return

    # Constant-time comparison avoids leaking useful timing information.
    if not supplied_key or not compare_digest(supplied_key, expected_key):
        raise AppError(401, "unauthorized", "Missing or invalid SAVVY_API_KEY header")


def require_admin_api_key(
    supplied_key: str | None = Depends(_admin_api_key_header),
    settings: Settings = Depends(get_settings),
) -> None:
    """Protect container-control operations with a separate private key."""

    if settings.admin_api_key is None:
        raise ConfigurationError("ADMIN_API_KEY is missing from .env")

    expected_key = settings.admin_api_key.get_secret_value()
    if not expected_key:
        raise ConfigurationError("ADMIN_API_KEY is empty in .env")
    if not supplied_key or not compare_digest(supplied_key, expected_key):
        raise AppError(401, "unauthorized", "Missing or invalid ADMIN_API_KEY header")

