"""Check invoice data with predictable UAE rules."""

import re
from datetime import date
from decimal import Decimal
from typing import Any

from uae_invoice_reader.countries.uae.models.invoice import ValidationResult

_TRN = re.compile(r"^\d{15}$")
_CURRENCY = re.compile(r"^[A-Z]{3}$")
_TOLERANCE = Decimal("0.02")


class InvoiceValidationService:
    """Collect warnings without changing values printed on the invoice."""

    def validate(self, invoice: Any, source_text: str) -> ValidationResult:
        """Run all UAE invoice checks and return every warning found."""

        warnings: list[str] = []

        supplier_name = getattr(invoice, "vendor_name", None) or getattr(invoice, "supplier_name", None)
        invoice_number = getattr(invoice, "invoice_no", None) or getattr(invoice, "invoice_number", None)
        invoice_date_val = getattr(invoice, "invoice_date", None)
        due_date_val = getattr(invoice, "due_date", None)
        subtotal = getattr(invoice, "net_amount", None) or getattr(invoice, "subtotal", None)
        total_amount = getattr(invoice, "grand_total", None) or getattr(invoice, "total_amount", None)
        supplier_trn = getattr(invoice, "vendor_trn_no", None) or getattr(invoice, "supplier_trn", None)
        customer_trn = getattr(invoice, "buyer_trn_no", None) or getattr(invoice, "customer_trn", None)
        currency_val = getattr(invoice, "currency", None)
        vat_amount = getattr(invoice, "total_tax", None) or getattr(invoice, "vat_amount", None)
        items = getattr(invoice, "items", []) or []

        # Required fields
        for name, value in (
            ("vendor_name", supplier_name),
            ("invoice_no", invoice_number),
            ("invoice_date", invoice_date_val),
            ("net_amount", subtotal),
            ("grand_total", total_amount),
        ):
            if value in (None, ""):
                warnings.append(f"{name} is missing")

        # TRN checks
        self._trn("vendor_trn_no", supplier_trn, warnings, required=True)
        self._trn("buyer_trn_no", customer_trn, warnings, required=False)

        # Dates
        for name, value in (("invoice_date", invoice_date_val), ("due_date", due_date_val)):
            if value:
                try:
                    date.fromisoformat(value)
                except ValueError:
                    warnings.append(f"{name} is not a valid YYYY-MM-DD date")

        # Currency
        if currency_val and _CURRENCY.fullmatch(currency_val) is None:
            warnings.append("currency must be a three-letter uppercase code")

        # Arithmetic
        if total_amount is not None and subtotal is not None:
            expected = Decimal(str(subtotal))
            if vat_amount is not None:
                expected += Decimal(str(vat_amount))
            if abs(expected - Decimal(str(total_amount))) > _TOLERANCE:
                warnings.append("subtotal, VAT, and grand_total do not balance")

        # Source evidence
        if source_text:
            normalized = re.sub(r"\s+", "", source_text).casefold()
            for name, value in (
                ("invoice_no", invoice_number),
                ("vendor_trn_no", supplier_trn),
                ("buyer_trn_no", customer_trn),
            ):
                if value and re.sub(r"\s+", "", str(value)).casefold() not in normalized:
                    warnings.append(f"{name} was not found in extracted_text")

        return ValidationResult(is_valid=not warnings, warnings=warnings)

    @staticmethod
    def _trn(
        name: str, value: str | None, warnings: list[str], *, required: bool = True
    ) -> None:
        """A UAE Tax Registration Number must contain exactly 15 digits."""

        if not value:
            if required:
                warnings.append(f"{name} is missing")
            return
        clean_value = str(value).strip()
        if _TRN.fullmatch(clean_value) is None:
            warnings.append(f"{name} must contain exactly 15 digits")
