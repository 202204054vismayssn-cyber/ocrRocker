"""Public contracts for live processing status and safe container control."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SystemDTO(BaseModel):
    """Reject accidental fields in operational API responses."""

    model_config = ConfigDict(extra="forbid")


class ActiveProcessingDTO(SystemDTO):
    """Non-sensitive facts about one currently active invoice operation."""

    request_id: str
    operation: str
    phase: str
    started_at: datetime
    elapsed_seconds: float = Field(ge=0)


class OperationalErrorDTO(SystemDTO):
    """One sanitized recent failure from Docker, OCR, or LLM."""

    occurred_at: datetime
    request_id: str
    source: str
    code: str
    http_status: int = Field(ge=400, le=599)
    retryable: bool
    detail: str


class ProcessingStatusResponseDTO(SystemDTO):
    """Live in-memory state from the extraction concurrency gate."""

    success: Literal[True] = True
    request_id: str
    is_processing: bool
    accepting_requests: bool
    active_count: int = Field(ge=0)
    max_concurrent: int = Field(ge=1)
    available_slots: int = Field(ge=0)
    active_requests: list[ActiveProcessingDTO] = Field(default_factory=list)
    recent_error_count: int = Field(default=0, ge=0)
    recent_errors: list[OperationalErrorDTO] = Field(default_factory=list)


class RestartResponseDTO(SystemDTO):
    """Confirmation returned before the container terminates and restarts."""

    success: Literal[True] = True
    request_id: str
    status: Literal["restart_scheduled"] = "restart_scheduled"
    restart_in_seconds: float = Field(ge=1)
    message: str = "The API container restart has been scheduled"
