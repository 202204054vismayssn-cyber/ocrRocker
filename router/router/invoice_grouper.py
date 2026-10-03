"""Invoice grouping and merging: combines per-page results into logical invoices.

Multi-page invoices are common — the first page has customer/vendor details,
continuation pages carry the item table, and support pages hold IRN/Ack info.
This module groups pages by invoice number and merges them into one canonical
record per logical invoice.
"""

from __future__ import annotations

import copy
import re

from router._modules import ocr
from router.batch import MULTI_PAGE_HANDLING


def _is_present(value) -> bool:
    """Return True when a field value is present and not just whitespace."""
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def _normal_identifier(value) -> str:
    """Normalize an invoice number for grouping by removing whitespace and uppercasing."""
    return re.sub(r"\s+", "", str(value or "")).upper()


def _group_page_results(page_results: list[dict]) -> list[list[dict]]:
    """Group continuation and support pages under their logical invoice header."""
    groups: list[list[dict]] = []
    current: list[dict] = []
    current_identifier = ""

    for result in page_results:
        context = result.get("_router_context") or {}
        profile = context.get("profile") or {}
        fields = result.get("fields") or {}
        identifier = _normal_identifier(
            profile.get("invoice_number") or fields.get("invoice_number")
        )
        starts_invoice = bool(profile.get("starts_invoice") or identifier)

        starts_new_group = False
        if current:
            if starts_invoice and identifier:
                starts_new_group = bool(
                    not current_identifier or identifier != current_identifier
                )
            elif starts_invoice:
                starts_new_group = True

        if starts_new_group:
            groups.append(current)
            current = []
            current_identifier = ""

        current.append(result)
        if identifier and not current_identifier:
            current_identifier = identifier

    if current:
        groups.append(current)
    return groups


def _item_key(item: dict) -> tuple:
    """Generate a deduplication key for a line item across merged pages."""
    return (
        item.get("serial_number"),
        re.sub(r"\s+", " ", str(item.get("description") or "")).strip().lower(),
        str(item.get("hsn_sac") or "").strip(),
        item.get("amount"),
        item.get("taxable_value"),
    )


def _usable_item(item) -> bool:
    """Return True when a line item has enough meaningful content to keep."""
    if not isinstance(item, dict):
        return False
    return bool(
        str(item.get("description") or "").strip()
        or str(item.get("hsn_sac") or "").strip()
        or isinstance(item.get("amount"), (int, float))
        or isinstance(item.get("taxable_value"), (int, float))
    )


def _sum_item_values(items: list[dict], field: str):
    """Sum numeric values in a field across all line items that have it."""
    values = [
        item.get(field) for item in items
        if isinstance(item.get(field), (int, float))
    ]
    return round(sum(values), 2) if values else None


def _aggregate_timing(page_results: list[dict]) -> dict:
    """Combine timing information across all pages in an invoice group."""
    keys = {key for result in page_results for key in (result.get("timing") or {})}
    timing = {}
    for key in sorted(keys):
        values = [(result.get("timing") or {}).get(key) for result in page_results]
        numeric = [value for value in values if isinstance(value, (int, float))]
        if not numeric:
            continue
        # Model load time is the max across pages; everything else is summed.
        timing[key] = round(max(numeric) if "model_load" in key else sum(numeric), 2)
    return timing


def _merge_invoice_group(
    page_results: list[dict],
    *,
    invoice_index: int,
    document_page_count: int,
) -> dict:
    """Create one canonical record from all pages of one logical invoice."""
    if not page_results:
        raise ValueError("Cannot merge an empty invoice page group.")

    merged = copy.deepcopy(page_results[0])
    fields = merged.setdefault("fields", {})
    sources = merged.setdefault("field_sources", {})

    # Merge header fields: take the first present value across all pages.
    for page_result in page_results[1:]:
        for key, value in (page_result.get("fields") or {}).items():
            if key in {"line_items", "bill_to", "ship_to"}:
                continue
            if not _is_present(fields.get(key)) and _is_present(value):
                fields[key] = copy.deepcopy(value)
                if key in (page_result.get("field_sources") or {}):
                    sources[key] = page_result["field_sources"][key]

    # Customer info: prefer a native Bill To block over generic customer fields.
    bill_to = next(
        (
            (page_result.get("fields") or {}).get("bill_to")
            for page_result in page_results
            if isinstance((page_result.get("fields") or {}).get("bill_to"), dict)
        ),
        None,
    )
    if bill_to:
        fields["bill_to"] = copy.deepcopy(bill_to)
        fields["customer_name"] = bill_to.get("name")
        fields["customer_address"] = bill_to.get("address")
        fields["customer_gstin"] = bill_to.get("gstin")
        for key in ("bill_to", "customer_name", "customer_address", "customer_gstin"):
            sources[key] = "native_bill_to_block"
    fields.pop("ship_to", None)
    sources.pop("ship_to", None)

    # Line items: deduplicate across pages by key, keeping the first occurrence.
    line_items = []
    seen_items = set()
    for page_result in page_results:
        for item in (page_result.get("fields") or {}).get("line_items") or []:
            if not _usable_item(item):
                continue
            key = _item_key(item)
            if key in seen_items:
                continue
            seen_items.add(key)
            line_items.append(copy.deepcopy(item))
    fields["line_items"] = line_items
    if line_items:
        sources["line_items"] = "merged_page_tables"

    # Terminal totals: prefer a page that has terminal summary values.
    terminal_totals = None
    for page_result in page_results:
        totals = (page_result.get("_router_context") or {}).get("totals") or {}
        if totals.get("has_terminal_total"):
            terminal_totals = totals
    if terminal_totals:
        for key in (
            "subtotal", "taxable_value", "cgst_amount", "sgst_amount",
            "igst_amount", "tax", "rounding", "total_amount",
            "payment_made", "balance_due",
        ):
            value = terminal_totals.get(key)
            if _is_present(value):
                fields[key] = value
                sources[key] = "native_terminal_summary"

    # Subtotal fallback: calculate from line items if not present in terminal.
    subtotal_from_items = _sum_item_values(line_items, "taxable_value")
    if subtotal_from_items is None:
        subtotal_from_items = _sum_item_values(line_items, "amount")
    if not isinstance(fields.get("subtotal"), (int, float)) and subtotal_from_items is not None:
        fields["subtotal"] = subtotal_from_items
        sources["subtotal"] = "calculated_from_merged_line_items"
    if not isinstance(fields.get("taxable_value"), (int, float)) and isinstance(fields.get("subtotal"), (int, float)):
        fields["taxable_value"] = fields["subtotal"]
        sources["taxable_value"] = "inferred_from_subtotal"

    # GST components: use terminal totals if present, else calculate from items.
    for component in ("cgst_amount", "sgst_amount", "igst_amount"):
        item_total = _sum_item_values(line_items, component)
        if item_total is not None and (
            not terminal_totals or not isinstance(terminal_totals.get(component), (int, float))
        ):
            fields[component] = item_total
            sources[component] = "calculated_from_merged_line_items"
        if not isinstance(fields.get(component), (int, float)):
            fields[component] = 0.0
            sources[component] = "not_applicable_zero"

    # Total tax and aliases.
    tax_total = round(sum(fields[key] for key in ("cgst_amount", "sgst_amount", "igst_amount")), 2)
    fields["tax"] = tax_total
    fields["total_tax_amount"] = tax_total
    sources["tax"] = "calculated_from_components"
    sources["total_tax_amount"] = "calculated_from_components"
    for component, alias in {
        "cgst_amount": "total_cgst_amount",
        "sgst_amount": "total_sgst_amount",
        "igst_amount": "total_igst_amount",
    }.items():
        fields[alias] = fields[component]
        sources[alias] = sources.get(component, "not_applicable_zero")

    # Gross total fallback: calculate if not present in terminal summary.
    if not isinstance(fields.get("total_amount"), (int, float)):
        total_values = _sum_item_values(line_items, "total_value")
        if total_values is not None:
            fields["total_amount"] = total_values
            sources["total_amount"] = "calculated_from_merged_line_items"
        elif isinstance(fields.get("subtotal"), (int, float)):
            rounding = fields.get("rounding") if isinstance(fields.get("rounding"), (int, float)) else 0.0
            fields["total_amount"] = round(fields["subtotal"] + tax_total + rounding, 2)
            sources["total_amount"] = "calculated_from_subtotal_and_tax"

    # Validation: check that the total arithmetic is consistent.
    combined_text = "\n".join(
        str((page_result.get("_router_context") or {}).get("text") or "")
        for page_result in page_results
    )
    validation_issues = [
        issue for issue in ocr._validation_results(fields, combined_text, sources)
        if issue.get("code") != "total_arithmetic_mismatch"
    ]
    subtotal = fields.get("subtotal")
    total_amount = fields.get("total_amount")
    if isinstance(subtotal, (int, float)) and isinstance(total_amount, (int, float)):
        rounding = fields.get("rounding") if isinstance(fields.get("rounding"), (int, float)) else 0.0
        expected = round(subtotal + tax_total + rounding, 2)
        difference = round(total_amount - expected, 2)
        if abs(difference) > 0.05:
            validation_issues.append({
                "code": "total_arithmetic_mismatch",
                "field": "total_amount",
                "expected": expected,
                "extracted": total_amount,
                "difference": difference,
                "message": "Printed total does not equal subtotal plus extracted GST amounts and rounding.",
            })

    # Missing required fields.
    missing_required = [
        field for field, spec in ocr.CANONICAL_SCHEMA.items()
        if spec.get("required") and not _is_present(fields.get(field))
    ]
    merged["missing_required"] = missing_required
    merged["validation_issues"] = validation_issues
    merged["unparsed_or_low_confidence"] = list(missing_required)
    merged["needs_review"] = bool(missing_required or validation_issues)

    # Merge metadata: page numbers, methods, roles, timing.
    page_numbers = [result["page_number"] for result in page_results]
    methods = [result["extraction_method"] for result in page_results]
    roles = [result.get("page_role", "unclassified_continuation") for result in page_results]
    merged["source_file"] = page_results[0]["source_file"]
    merged["invoice_index"] = invoice_index
    merged["page_numbers"] = page_numbers
    merged["first_page"] = min(page_numbers)
    merged["last_page"] = max(page_numbers)
    merged["invoice_page_count"] = len(page_numbers)
    merged["document_page_count"] = document_page_count
    merged["page_count"] = document_page_count
    merged.pop("page_number", None)
    merged["page_roles"] = roles
    merged.pop("page_role", None)
    merged["page_extraction_methods"] = methods
    merged["extraction_method"] = methods[0] if len(set(methods)) == 1 else "mixed_native_and_ocr"
    merged["multi_page_handling"] = MULTI_PAGE_HANDLING
    merged["timing"] = _aggregate_timing(page_results)

    # Prepare semantic context for potential Qwen fallback (removed later).
    merged["_semantic_context"] = {
        "raw_text": combined_text,
        "image_paths": [
            str((result.get("_router_context") or {}).get("image_path"))
            for result in page_results
            if (result.get("_router_context") or {}).get("image_path")
        ],
    }
    merged.pop("_router_context", None)
    return merged