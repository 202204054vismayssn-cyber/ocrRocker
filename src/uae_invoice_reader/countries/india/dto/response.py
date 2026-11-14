"""Public response contracts for India invoice extraction."""

from typing import Literal

from pydantic import Field

from uae_invoice_reader.common.dto.base import (
    ExtractionMetadataDTO,
    PublicDTO,
    TokenUsageDTO,
)
from uae_invoice_reader.countries.india.models.invoice import Money


class InvoiceItemDTO(PublicDTO):
    sr_no: str | None = None
    description: str | None = None
    quantity: Money | None = None
    unit: str | None = None
    rate: Money | None = None
    amount: Money | None = None
    hsn_sac: str | None = None


class HsnSummaryDTO(PublicDTO):
    hsn: str | None = None
    tax_rate: Money | None = None
    taxable_amount: Money | None = None
    cgst: Money | None = None
    sgst: Money | None = None
    igst: Money | None = None


class IndiaInvoiceDTO(PublicDTO):
    customer_name: str | None = None
    customer_address: str | None = None
    customer_gst: str | None = None
    vendor_name: str | None = None
    vendor_address: str | None = None
    vendor_gst: str | None = None
    vendor_pan: str | None = None
    invoice_number: str | None = None
    invoice_date: str | None = None
    taxable_amount: Money | None = None
    igst_amount: Money | None = None
    cgst_amount: Money | None = None
    sgst_amount: Money | None = None
    invoice_amount: Money | None = None
    hsn_sac_code: str | None = None
    discount: Money | None = None
    state_cess: Money | None = None
    other_charges: Money | None = None
    irn_number: str | int | None = 0
    customer_pan: str | None = None
    bank_name: str | None = None
    bank_branch: str | None = None
    bank_account_number: str | None = None
    ifsc_code: str | None = None
    due_date: str | None = None
    items: list[InvoiceItemDTO] = Field(default_factory=list)
    hsn_summary: list[HsnSummaryDTO] = Field(default_factory=list)


class ExtractionResponseDTO(PublicDTO):
    success: Literal[True] = True
    request_id: str
    extracted_text: str = Field(alias="character_stored")
    data: IndiaInvoiceDTO
    token_usage: TokenUsageDTO
    metadata: ExtractionMetadataDTO


class CompletionExtractionResponseDTO(ExtractionResponseDTO):
    invoice_id: str = Field(min_length=1, max_length=100)
