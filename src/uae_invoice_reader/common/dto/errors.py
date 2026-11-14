"""Stable error response shared by system and country routes."""

from typing import Literal

from uae_invoice_reader.common.dto.base import PublicDTO


class ErrorBodyDTO(PublicDTO):
    code: str
    detail: str
    source: str = "docker_api"
    retryable: bool = False


class ErrorResponseDTO(PublicDTO):
    success: Literal[False] = False
    request_id: str
    error: ErrorBodyDTO
