"""Public response contracts for USA invoice extraction."""

from typing import Literal

from pydantic import Field

from uae_invoice_reader.common.dto.base import (
    ExtractionMetadataDTO,
    PublicDTO,
    TokenUsageDTO,
)
from uae_invoice_reader.countries.usa.models.invoice import Money


class InvoiceItemDTO(PublicDTO):
    line_number: str | None = None
    item_description: str | None = None
    qty: Money | None = None
    unit: str | None = None
    unit_price: Money | None = None
    tax_rate: Money | None = None
    line_total: Money | None = None


class UsaInvoiceDTO(PublicDTO):
    invoice_no: str | None = None
    invoice_date: str | None = None
    po_no: str | None = None
    net_amount: Money | None = None
    sales_tax_rate: Money | None = None
    total_sales_tax: Money | None = None
    shipping_handling_freight: Money | None = None
    grand_total: Money | None = None
    payable_amount: Money | None = None
    items: list[InvoiceItemDTO] = Field(default_factory=list)
    vendor_name: str | None = None
    vendor_ein: str | None = None
    state_tax_permit_license: str | None = None
    vendor_address: str | None = None
    customer_name: str | None = None
    customer_tax_id_ein: str | None = None
    resale_exemption_certificate_no: str | None = None
    customer_billing_address: str | None = None
    bank_name: str | None = None
    routing_number: str | None = None
    account_number: str | None = None
    swift_bic: str | None = None
    bank_address_location: str | None = None
    remark: str | None = None


class ExtractionResponseDTO(PublicDTO):
    success: Literal[True] = True
    request_id: str
    extracted_text: str = Field(alias="character_stored")
    data: UsaInvoiceDTO
    token_usage: TokenUsageDTO
    metadata: ExtractionMetadataDTO


class CompletionExtractionResponseDTO(ExtractionResponseDTO):
    invoice_id: str = Field(min_length=1, max_length=100)
