"""Deterministic native-PDF invoice extraction helpers.

This module complements the existing generic OCR parser.  It uses the text
layer and table geometry already present in native PDFs, which is both faster
and more exact than asking a vision or language model to reproduce numbers.
"""

from __future__ import annotations

import re
from copy import deepcopy
from decimal import Decimal, InvalidOperation


GSTIN_RE = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b", re.I)
MONEY_RE = r"(?:₹|INR\s*|Rs\.?\s*)?(-?\s*\d[\d,]*(?:\.\d{1,2})?)"


def _clean(value) -> str:
    """Collapse runs of spaces/tabs and strip the ends of a cell value."""
    return re.sub(r"[ \t]+", " ", str(value or "")).strip()


def _norm(value) -> str:
    """Lowercase a label and reduce it to letters, digits and ``%`` for matching."""
    return re.sub(r"[^a-z0-9%]+", " ", _clean(value).lower()).strip()


def _money(value):
    """Parse a currency-ish string to a float, or None when it isn't a number.

    Tolerates the ``₹``/``Rs.``/``INR`` prefixes and thousands separators that
    appear in invoice cells.  Decimal is used rather than float() so values
    like ``1,00,000.50`` convert exactly instead of accumulating binary error.
    """
    text = _clean(value).replace("₹", "").replace(",", "")
    text = re.sub(r"(?i)^(?:inr|rs\.?)\s*", "", text)
    text = text.replace(" ", "")
    try:
        return float(Decimal(text))
    except (InvalidOperation, ValueError):
        return None


def _first_number(value):
    """Return the first number appearing in a cell, ignoring surrounding text."""
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", _clean(value))
    return _money(match.group(0)) if match else None


def _last_labeled_amount(text: str, label_pattern: str):
    """Return the amount following the *last* match of a labelled total.

    Invoices often repeat a label — a per-rate tax summary and then a grand
    total.  Taking the last occurrence gets the invoice-level figure rather
    than a per-line one.
    """
    matches = re.findall(
        rf"(?i){label_pattern}\s*[:|]?\s*{MONEY_RE}",
        text,
    )
    return _money(matches[-1]) if matches else None


def extract_invoice_number(text: str):
    """Read a header invoice identifier, including layouts that label it '#'."""
    patterns = [
        r"(?im)^\s*(?:invoice\s*(?:no\.?|number|#)|tax\s+invoice\s*(?:no\.?|number|#)|bill\s*(?:no\.?|number))\s*[:#-]?\s*([A-Z0-9][A-Z0-9./_-]{2,})",
        r"(?im)^\s*#\s*:\s*([A-Z0-9][A-Z0-9./_-]{2,})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match and re.search(r"\d", match.group(1)):
            return match.group(1).strip(".:- ").upper()
    return None


def page_profile(text: str, line_items: list[dict] | None = None) -> dict:
    """Classify a page's role in a multi-page invoice.

    The router uses this to decide whether a page starts a new logical invoice
    or continues the previous one.  An invoice number is the strongest signal;
    failing that, an invoice heading combined with both Bill To and an invoice
    date is treated as a new invoice.  Pages carrying items are continuations,
    and IRN/Ack pages are flagged as supporting pages.
    """
    invoice_number = extract_invoice_number(text)
    lowered = text.lower()
    has_invoice_heading = "tax invoice" in lowered or "invoice" in lowered[:800]
    has_bill_to = bool(re.search(r"\bbill\s+to\b", text, re.I))
    has_invoice_date = bool(re.search(r"\binvoice\s+date\b", text, re.I))
    has_items = bool(line_items) or bool(
        re.search(r"item\s*(?:&|and)?\s*description|goods/?service\s+description", text, re.I)
    )
    is_support = bool(re.search(r"\bIRN\b|\bAck\s+No\.?\b|e-Invoicing detail", text, re.I))
    starts_invoice = bool(invoice_number) or (
        has_invoice_heading and has_bill_to and has_invoice_date
    )
    if starts_invoice:
        role = "invoice_start"
    elif has_items:
        role = "table_continuation"
    elif is_support:
        role = "supporting_page"
    else:
        role = "unclassified_continuation"
    return {
        "invoice_number": invoice_number,
        "starts_invoice": starts_invoice,
        "has_items": has_items,
        "is_support": is_support,
        "role": role,
    }


def _supplier_block(tables: list, text: str) -> str:
    """Return the text block introducing the supplier, up to "Tax Invoice".

    Prefers a table cell that contains the "Tax Invoice" heading — on native
    PDFs the supplier name, address and GSTIN usually share that cell — and
    falls back to the page text when the layout has no table.
    """
    for table in tables or []:
        for row in table or []:
            for cell in row or []:
                # Preserve line breaks: the first line is normally the legal
                # supplier name and the remaining lines are its address/GSTIN.
                value = str(cell or "").strip()
                if value and re.search(r"\btax\s+invoice\b", value, re.I):
                    return value
    match = re.search(r"(?is)\A(.*?)\btax\s+invoice\b", text)
    return match.group(1).strip() if match else ""


def extract_supplier(tables: list, text: str) -> dict:
    """Read the supplier's name and GSTIN from the header block.

    The first non-empty line that isn't the "Tax Invoice" heading and doesn't
    look like a GSTIN, phone number or email is taken as the legal name.  A
    bare "GSTIN" label is resolved by position — the supplier's is the one
    appearing before any recipient block.
    """
    block = _supplier_block(tables, text)
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    name = None
    for line in lines:
        candidate = re.sub(r"(?i)\btax\s+invoice\b.*$", "", line).strip()
        if not candidate or re.search(r"\bgstin\b|@|\bphone\b|^\+?\d[\d -]{7,}$", candidate, re.I):
            continue
        name = candidate
        break
    gstins = GSTIN_RE.findall(block)
    if not gstins:
        before_bill = re.split(r"\bbill\s+to\b", text, maxsplit=1, flags=re.I)[0]
        gstins = GSTIN_RE.findall(before_bill)
    return {
        "name": name,
        "gstin": gstins[0].upper() if gstins else None,
    }


def _is_item_header_row(row: list) -> bool:
    """Return True when a table row looks like the line-item column header.

    Matches on the pairing of a description-like column with a tax or amount
    column, so vendors that word it "Particulars" or "Goods/Service
    Description" are both recognised.
    """
    joined = " | ".join(_norm(cell) for cell in row if cell)
    return (
        ("description" in joined or "particular" in joined)
        and ("hsn" in joined or "sac" in joined or "amount" in joined)
    )


def extract_bill_to(tables: list) -> dict | None:
    """Use table geometry to take only Bill To and never the Ship To column."""
    for table in tables or []:
        for row_index, row in enumerate(table or []):
            normalized = [_norm(cell) for cell in row]
            bill_indices = [i for i, value in enumerate(normalized) if value == "bill to"]
            if not bill_indices:
                continue
            bill_index = bill_indices[0]
            ship_indices = [
                i for i, value in enumerate(normalized)
                if i > bill_index and value == "ship to"
            ]
            bill_end = ship_indices[0] if ship_indices else len(row)
            chunks = []
            for following in table[row_index + 1:]:
                if _is_item_header_row(following):
                    break
                row_text = " | ".join(_clean(cell) for cell in following if cell)
                if re.search(r"(?i)^\s*subject\s*:|item\s*(?:&|and)?\s*description", row_text):
                    break
                for cell in following[bill_index:bill_end]:
                    value = str(cell or "").strip()
                    if value and _norm(value) not in {"bill to", "ship to"}:
                        chunks.append(value)
            if not chunks:
                continue

            lines = []
            for chunk in chunks:
                for line in chunk.splitlines():
                    line = line.strip()
                    if line and line not in lines:
                        lines.append(line)
            gstins = []
            address_lines = []
            name = None
            for line in lines:
                found = GSTIN_RE.findall(line)
                if found:
                    gstins.extend(value.upper() for value in found)
                    continue
                if name is None:
                    name = line
                else:
                    address_lines.append(line)
            if name:
                return {
                    "name": name,
                    "address": ", ".join(address_lines) or None,
                    "gstin": gstins[0] if gstins else None,
                }
    return None


def _find_header(table: list):
    """Return the row index of the line-item header, or None if absent."""
    for index, row in enumerate(table or []):
        if _is_item_header_row(row):
            return index
    return None


def _column_index(values: list[str], patterns: list[str], *, last=False):
    """Return the index of the header cell matching any pattern.

    With ``last=True`` the rightmost match wins, which matters for tax summary
    tables where a rate column precedes its amount column and only the
    rightmost "amount"-like header is the total.
    """
    matches = [
        index for index, value in enumerate(values)
        if any(re.search(pattern, value, re.I) for pattern in patterns)
    ]
    if not matches:
        return None
    return matches[-1] if last else matches[0]


def _parse_quantity(value):
    """Split a quantity cell into its number and trailing unit (e.g. 2, "NOS")."""
    text = _clean(value)
    quantity = _first_number(text)
    unit_match = re.search(r"[A-Za-z][A-Za-z ._-]*$", text)
    return quantity, _clean(unit_match.group(0)) if unit_match else None


def parse_line_items(tables: list) -> list[dict]:
    """Parse line items from every table that has an item header.

    Walks each table from its header row downward, reading cells by column
    name rather than by fixed position, so vendors that order their columns
    differently still parse.  Multi-row headers (a spanning "CGST" above
    "Rate"/"Amt") are combined into single column keys.
    """
    parsed_items = []
    for table in tables or []:
        header_start = _find_header(table)
        if header_start is None:
            continue
        raw_primary = [_clean(cell) for cell in table[header_start]]
        primary = [_norm(cell) for cell in table[header_start]]
        # ``_norm('#')`` is intentionally empty, so handle that common serial
        # header before using normalized-text patterns.
        serial_index = next(
            (index for index, value in enumerate(raw_primary) if value.strip() == "#"),
            None,
        )
        if serial_index is None:
            serial_index = _column_index(primary, [r"^sl\s*no", r"^sr\s*no"])
        if serial_index is None:
            serial_index = 0

        first_item_row = None
        for index in range(header_start + 1, len(table)):
            cells = table[index]
            serial = _clean(cells[serial_index] if serial_index < len(cells) else "")
            if re.fullmatch(r"\d+", serial):
                first_item_row = index
                break
        if first_item_row is None:
            continue

        header_rows = table[header_start:first_item_row]
        width = max(len(row) for row in header_rows)
        combined = []
        for column in range(width):
            pieces = [
                _norm(row[column]) for row in header_rows
                if column < len(row) and _clean(row[column])
            ]
            combined.append(" ".join(piece for piece in pieces if piece))

        description_index = _column_index(combined, [r"description", r"particular"])
        hsn_index = _column_index(combined, [r"\bhsn\b", r"\bsac\b"])
        quantity_index = _column_index(combined, [r"\bqty\b", r"quantity"])
        taxable_index = _column_index(primary, [r"taxable\s+value", r"taxable\s+amount"])
        total_value_index = _column_index(primary, [r"total\s+value"])

        rate_index = None
        for index, value in enumerate(primary):
            if value == "rate":
                rate_index = index
                break

        amount_candidates = [
            index for index, value in enumerate(primary)
            if value == "amount" or value.endswith(" amount")
        ]
        amount_index = amount_candidates[-1] if amount_candidates else None
        base_amount_index = taxable_index if taxable_index is not None else amount_index

        tax_starts = []
        for index, value in enumerate(primary):
            match = re.search(r"\b(cgst|sgst|utgst|igst)\b", value)
            if match:
                tax_starts.append((index, match.group(1).lower()))

        tax_columns = {}
        for position, (start, tax_name) in enumerate(tax_starts):
            stop = tax_starts[position + 1][0] if position + 1 < len(tax_starts) else width
            blockers = [
                value for value in (base_amount_index, total_value_index)
                if value is not None and value > start
            ]
            if blockers:
                stop = min(stop, min(blockers))
            rate_col = next(
                (i for i in range(start, stop) if "%" in combined[i] or "rate" in combined[i]),
                start,
            )
            amount_col = next(
                (i for i in range(start, stop) if re.search(r"\b(?:amt|amount)\b", combined[i])),
                None,
            )
            canonical = "sgst" if tax_name == "utgst" else tax_name
            tax_columns[canonical] = (rate_col, amount_col)

        last_item = None
        for row in table[first_item_row:]:
            cells = list(row) + [None] * max(0, width - len(row))
            row_text = " | ".join(_clean(cell) for cell in cells if cell)
            if re.search(r"\bsub\s*total\b|\bgrand\s+total\b|\bbalance\s+due\b", row_text, re.I):
                break
            serial = _clean(cells[serial_index])
            if not re.fullmatch(r"\d+", serial):
                if last_item is not None and description_index is not None:
                    continuation = _clean(cells[description_index])
                    if continuation:
                        last_item["description"] = _clean(
                            f"{last_item.get('description', '')} {continuation}"
                        )
                continue

            item = {"serial_number": int(serial)}
            if description_index is not None:
                item["description"] = _clean(cells[description_index].replace("\n", " ") if cells[description_index] else "")
            if hsn_index is not None:
                item["hsn_sac"] = re.sub(r"\s+", "", _clean(cells[hsn_index]))
            if quantity_index is not None:
                quantity, unit = _parse_quantity(cells[quantity_index])
                item["quantity"] = quantity
                if unit:
                    item["unit"] = unit
            if rate_index is not None:
                item["unit_rate"] = _first_number(cells[rate_index])
            if base_amount_index is not None:
                item["amount"] = _first_number(cells[base_amount_index])
            if taxable_index is not None:
                item["taxable_value"] = _first_number(cells[taxable_index])
            if total_value_index is not None:
                item["total_value"] = _first_number(cells[total_value_index])

            for tax_name, (tax_rate_index, tax_amount_index) in tax_columns.items():
                rate = _first_number(cells[tax_rate_index]) if tax_rate_index is not None else None
                amount = _first_number(cells[tax_amount_index]) if tax_amount_index is not None else None
                item[f"{tax_name}_rate"] = rate
                item[f"{tax_name}_amount"] = amount

            meaningful = item.get("description") or item.get("hsn_sac") or item.get("amount") is not None
            if meaningful:
                parsed_items.append(item)
                last_item = item
    return parsed_items


def extract_totals(text: str, line_items: list[dict]) -> dict:
    """Extract terminal table totals, keeping gross Total separate from Balance Due."""
    subtotal = _last_labeled_amount(text, r"\bsub\s*total\b")
    rounding = _last_labeled_amount(text, r"\bround(?:ing|\s*off)\b")
    payment_made = _last_labeled_amount(text, r"\bpayment\s+made(?:\s*\(-\))?")
    balance_due = _last_labeled_amount(text, r"\bbalance\s+due\b")

    total_candidates = []
    for line in text.splitlines():
        if re.search(r"\bsub\s*total\b|total\s+in\s+words|amount\s+in\s+words", line, re.I):
            continue
        match = re.search(
            rf"(?i)\b(?:grand\s+total|invoice\s+total|total)\b\s*[:|]?\s*{MONEY_RE}",
            line,
        )
        if match:
            value = _money(match.group(1))
            if value is not None:
                total_candidates.append(value)
    total_amount = total_candidates[-1] if total_candidates else None

    components = {}
    explicit_tax = False
    for canonical, names in {
        "cgst_amount": ("CGST",),
        "sgst_amount": ("SGST", "UTGST"),
        "igst_amount": ("IGST",),
    }.items():
        values = []
        for name in names:
            matches = re.findall(
                rf"(?i)\b{name}\s*[\d.]*\s*\(\s*[\d.]+\s*%\s*\)\s*[:|]?\s*{MONEY_RE}",
                text,
            )
            values.extend(value for value in (_money(match) for match in matches) if value is not None)
        if values:
            explicit_tax = True
            components[canonical] = round(sum(values), 2)

    for canonical in ("cgst_amount", "sgst_amount", "igst_amount"):
        if canonical in components:
            continue
        values = [
            item.get(canonical) for item in line_items
            if isinstance(item.get(canonical), (int, float))
        ]
        if values:
            components[canonical] = round(sum(values), 2)

    has_igst = components.get("igst_amount") is not None or bool(re.search(r"\bIGST\b", text))
    has_local_tax = any(components.get(key) is not None for key in ("cgst_amount", "sgst_amount")) or bool(
        re.search(r"\bCGST\b|\bSGST\b|\bUTGST\b", text)
    )
    if has_igst and not has_local_tax:
        components.setdefault("cgst_amount", 0.0)
        components.setdefault("sgst_amount", 0.0)
    if has_local_tax and not has_igst:
        components.setdefault("igst_amount", 0.0)

    if subtotal is None:
        item_amounts = [item.get("amount") for item in line_items]
        item_amounts = [value for value in item_amounts if isinstance(value, (int, float))]
        if item_amounts:
            subtotal = round(sum(item_amounts), 2)

    tax_values = [
        components.get(key) for key in ("cgst_amount", "sgst_amount", "igst_amount")
        if isinstance(components.get(key), (int, float))
    ]
    result = {
        "subtotal": subtotal,
        "taxable_value": subtotal,
        **components,
        "tax": round(sum(tax_values), 2) if tax_values else None,
        "rounding": rounding,
        "total_amount": total_amount,
        "payment_made": payment_made,
        "balance_due": balance_due,
        "has_terminal_total": total_amount is not None,
        "has_explicit_tax_summary": explicit_tax,
    }
    return result


def enhance_native_result(result: dict, text: str, tables: list) -> tuple[dict, dict]:
    """Overlay exact native-layout values on the existing generic result."""
    result = deepcopy(result)
    fields = result.setdefault("fields", {})
    sources = result.setdefault("field_sources", {})

    invoice_number = extract_invoice_number(text)
    if invoice_number:
        fields["invoice_number"] = invoice_number
        sources["invoice_number"] = "native_header"

    supplier = extract_supplier(tables, text)
    if supplier.get("name"):
        fields["vendor_name"] = supplier["name"]
        sources["vendor_name"] = "native_supplier_block"
    if supplier.get("gstin"):
        fields["vendor_gstin"] = supplier["gstin"]
        sources["vendor_gstin"] = "native_supplier_block"

    bill_to = extract_bill_to(tables)
    if bill_to:
        fields["bill_to"] = bill_to
        fields["customer_name"] = bill_to.get("name")
        fields["customer_address"] = bill_to.get("address")
        fields["customer_gstin"] = bill_to.get("gstin")
        sources.update({
            "bill_to": "native_bill_to_block",
            "customer_name": "native_bill_to_block",
            "customer_address": "native_bill_to_block",
            "customer_gstin": "native_bill_to_block",
        })
    fields.pop("ship_to", None)
    sources.pop("ship_to", None)

    line_items = parse_line_items(tables)
    if line_items:
        fields["line_items"] = line_items
        sources["line_items"] = "native_table_geometry"

    totals = extract_totals(text, line_items or fields.get("line_items") or [])
    for key, value in totals.items():
        if key.startswith("has_") or value is None:
            continue
        fields[key] = value
        sources[key] = "native_terminal_summary" if totals["has_terminal_total"] else "native_table"
    for tax_key, alias in {
        "cgst_amount": "total_cgst_amount",
        "sgst_amount": "total_sgst_amount",
        "igst_amount": "total_igst_amount",
        "tax": "total_tax_amount",
    }.items():
        if fields.get(tax_key) is not None:
            fields[alias] = fields[tax_key]
            sources[alias] = sources.get(tax_key, "native_table")

    missing = []
    for key in ("customer_name", "invoice_number", "invoice_date", "total_amount"):
        if fields.get(key) in (None, ""):
            missing.append(key)
    result["missing_required"] = missing
    result["needs_review"] = bool(missing or result.get("validation_issues"))

    context = {
        "text": text,
        "profile": page_profile(text, fields.get("line_items") or []),
        "totals": totals,
    }
    return result, context
