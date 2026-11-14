"""Shared Data Transfer Objects for the 4-step local batch extraction workflow."""

from datetime import UTC, datetime
from typing import Any

from pydantic import Field

from uae_invoice_reader.common.dto.base import PublicDTO


class BatchPrepareResponseDTO(PublicDTO):
    """Returned by POST /batch/prepare after local OCR and text compaction."""

    success: bool = True
    request_id: str
    character_stored: str
    total_pages: int = Field(default=1, ge=1)
    character_count: int = Field(default=0, ge=0)


class BatchSubmitItemDTO(PublicDTO):
    """Single prepared invoice item sent in POST /batch/submit."""

    custom_id: str | None = Field(default=None, alias="id")
    character_stored: str = Field(alias="text")


class BatchSubmitRequestDTO(PublicDTO):
    """Payload for POST /batch/submit containing 1 to 100 prepared texts."""

    items: list[BatchSubmitItemDTO] = Field(min_length=1, max_length=100)


class BatchSubmitResponseDTO(PublicDTO):
    """Immediate response from POST /batch/submit returning the batch job name."""

    success: bool = True
    batch_job_name: str
    total_items: int
    state: str = "PROCESSING"


class BatchStatusResponseDTO(PublicDTO):
    """Point-in-time status returned by GET /batch/status/{batchJobName}."""

    success: bool = True
    batch_job_name: str
    state: str  # "PROCESSING", "COMPLETED", "FAILED"
    total_items: int
    completed_items: int
    failed_items: int
    created_at: str
    updated_at: str


class BatchItemResultDTO(PublicDTO):
    """Result of an individual invoice within a completed batch."""

    custom_id: str | None = None
    success: bool = True
    data: Any | None = None
    error: str | None = None
    token_usage: dict[str, Any] | None = None


class BatchResultResponseDTO(PublicDTO):
    """Full extraction results returned by GET /batch/result/{batchJobName}."""

    success: bool = True
    batch_job_name: str
    state: str
    total_items: int
    completed_items: int
    failed_items: int
    results: list[BatchItemResultDTO] = Field(default_factory=list)
