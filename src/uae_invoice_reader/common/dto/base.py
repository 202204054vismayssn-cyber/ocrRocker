"""Base DTOs, schemas, and country-neutral validation helpers."""

import re
from decimal import Decimal
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer


class PublicDTO(BaseModel):
    """Base DTO for public API responses; allows extra fields from LLMs gracefully."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


StrictModel = PublicDTO


def clean_money(v: Any) -> Decimal | None:
    """Sanitize currency strings, commas, empty strings into valid Decimals without failing on malformed LLM outputs."""
    if v is None or v == "":
        return None
    if isinstance(v, Decimal):
        return v
    if isinstance(v, (int, float)):
        try:
            return Decimal(str(v))
        except Exception:
            return None
    if isinstance(v, str):
        cleaned = re.sub(r"[^\d.-]", "", v.strip())
        if not cleaned or cleaned in ("-", ".", "-.", ".-"):
            return None
        try:
            return Decimal(cleaned)
        except Exception:
            pass
        match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        if match:
            try:
                return Decimal(match.group(0))
            except Exception:
                pass
        return None
    return None


Money = Annotated[
    Decimal | None,
    BeforeValidator(clean_money),
    PlainSerializer(
        lambda value: float(value) if value is not None else None,
        return_type=float | None,
        when_used="json",
    ),
]


class TokenUsageDTO(PublicDTO):
    """Token counts and execution time reported by Ollama."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    thoughts_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    execution_time_ms: int | None = Field(default=None, ge=0)


class ExtractionMetadataDTO(PublicDTO):
    """Metadata regarding pages, OCR characters, and model inference."""

    total_pages: int = Field(default=1, ge=1)
    character_count: int = Field(default=0, ge=0)
    finish_reason: str | None = "stop"
    model_name: str = "gemma"
    model_provider: str = "ollama"
