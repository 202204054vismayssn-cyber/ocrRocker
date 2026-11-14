"""Readable business fields that the public API returns for a UAE invoice."""

from pydantic import Field

from uae_invoice_reader.common.dto.base import Money, StrictModel, clean_money


class ValidationResult(StrictModel):
    """Whether checks passed and explanations for anything uncertain."""

    is_valid: bool
    warnings: list[str] = Field(default_factory=list)


__all__ = ["Money", "StrictModel", "ValidationResult", "clean_money"]
