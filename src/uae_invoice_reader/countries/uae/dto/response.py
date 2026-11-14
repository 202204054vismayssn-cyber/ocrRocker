"""Exact UAE invoice JSON shapes returned to an API client."""

from typing import Literal

from pydantic import Field

from uae_invoice_reader.common.dto.base import (
    ExtractionMetadataDTO,
    PublicDTO,
    TokenUsageDTO,
)
from uae_invoice_reader.countries.uae.models.invoice import Money, ValidationResult


class DatabaseInvoiceItemDTO(PublicDTO):
    """Only item keys stored in the UAE invoice table JSON column."""

    item: str | None = None
    qty: Money | None = None
    unit_price: Money | None = None
    vat_rate: Money | None = None
    net_amount: Money | None = None
    line_total: Money | None = None


class DatabaseInvoiceDTO(PublicDTO):
    """Invoice values named exactly like their target database columns."""

    vendor_name: str | None = None
    vendor_address: str | None = None
    vendor_trn_no: str | None = None
    invoice_no: str | None = None
    invoice_date: str | None = None
    po_no: str | None = None
    buyer_name: str | None = None
    buyer_address: str | None = None
    buyer_trn_no: str | None = None
    vat_rate: Money | None = None
    net_amount: Money | None = None
    total_tax: Money | None = None
    grand_total: Money | None = None
    items: list[DatabaseInvoiceItemDTO] = Field(default_factory=list)
    bank_name: str | None = None
    account_number: str | None = None
    iban: str | None = None
    swift_bic: str | None = None
    bank_branch: str | None = None
    currency: str | None = "AED"


class ExtractionResponseDTO(PublicDTO):
    """Successful response from direct UAE extraction/completion endpoints."""

    success: Literal[True] = True
    request_id: str
    extracted_text: str = Field(alias="character_stored")
    data: DatabaseInvoiceDTO
    token_usage: TokenUsageDTO
    validation: ValidationResult
    metadata: ExtractionMetadataDTO


class CompletionExtractionResponseDTO(ExtractionResponseDTO):
    """Normal extraction response correlated to the PHP invoice row."""

    invoice_id: str = Field(min_length=1, max_length=100)
