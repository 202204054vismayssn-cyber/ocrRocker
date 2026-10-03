"""
invoice_extraction_colab.py

Adapted from a drug-label text-extraction script -> repurposed for invoice
field extraction using PaddleOCR PP-OCRv6 (PP-OCRv6_small_det + PP-OCRv6_small_rec),
a fast, lightweight CPU-friendly OCR model pair.

Paste the cells below into Google Colab in order.
Output: JSON (and optional XML) keyed exactly to your DB/form field names,
so it can be loaded straight into form inputs without extra remapping.
"""

# =============================================================================
# CELL 1 — Install (Colab)
# =============================================================================
# PP-OCRv6_small_det and PP-OCRv6_small_rec are shipped with paddleocr>=2.9.0.
# Install paddlepaddle first (CPU-only build shown; swap for paddlepaddle-gpu
# if you have a CUDA-capable GPU):
"""
!python -m pip install paddlepaddle==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
!python -m pip install -U "paddleocr>=2.9.0"
"""
# GPU build (CUDA 12.6):
"""
!python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
!python -m pip install -U "paddleocr>=2.9.0"
"""
# The doc-parser extra (paddleocr[doc-parser]) is NOT required for
# PP-OCRv6_small_det / PP-OCRv6_small_rec — those are classic det+rec models,
# not the Vision-Language pipeline.


# =============================================================================
# CELL 2 — Imports
# =============================================================================
import re
import json
import time
import gc
from html import unescape
from html.parser import HTMLParser
from itertools import product
from pathlib import Path
from difflib import SequenceMatcher

# Keep the parser importable for unit/regression tests on machines where the
# heavy OCR runtime is not installed.  Colab installs PaddleOCR in CELL 1.
try:
    from paddleocr import PaddleOCR  # PP-OCRv6 classic det+rec API
except ImportError:  # pragma: no cover - exercised only outside OCR runtime
    PaddleOCR = None


# =============================================================================
# CELL 3 — Canonical schema + alias dictionary
# (same as field_mapper.py — keep these two files in sync, or import
#  field_mapper directly if it's in the same Colab working directory)
# =============================================================================

CANONICAL_SCHEMA = {
    "customer_name": {"required": True, "type": "str"},
    "customer_gstin": {"required": False, "type": "str"},
    "vendor_name": {"required": False, "type": "str"},
    "vendor_gstin": {"required": False, "type": "str"},
    "invoice_number": {"required": True, "type": "str"},
    "invoice_date": {"required": True, "type": "date"},
    "due_date": {"required": False, "type": "date"},
    "place_of_supply": {"required": False, "type": "str"},
    "taxable_value": {"required": False, "type": "float"},
    "cgst_amount": {"required": False, "type": "float"},
    "sgst_amount": {"required": False, "type": "float"},
    "igst_amount": {"required": False, "type": "float"},
    "subtotal": {"required": False, "type": "float"},
    "tax": {"required": False, "type": "float"},
    "total_amount": {"required": True, "type": "float"},
    "amount_in_words": {"required": False, "type": "str"},
    "line_items": {"required": False, "type": "list"},  # from markdown tables
}

ALIAS_DICTIONARY = {
    "customer_name": ["client name", "customer", "name", "bill to", "buyer", "billed to",
                       "details of recipient name", "recipient name"],
    "customer_gstin": ["customer gstin", "recipient gstin", "gstin of recipient", "buyer gstin"],
    "vendor_name": ["seller", "from", "company name", "billed by", "supplier name", "supplier"],
    # A bare "GSTIN" is deliberately not assigned here.  In many invoices it
    # appears once for the supplier and once for the recipient; assigning both
    # occurrences to vendor_gstin silently overwrites the correct value.  The
    # context-aware GSTIN resolver below handles bare labels safely.
    "vendor_gstin": ["gstin of supplier", "supplier gstin", "seller gstin"],
    "invoice_number": ["invoice no", "invoice #", "inv no", "bill number", "invoice id",
                        "invoice no.", "tax invoice no"],
    "invoice_date": ["date", "invoice dt", "bill date", "issued on", "date of invoice",
                      "invoice date"],
    "due_date": ["payment due", "due", "due on"],
    "place_of_supply": ["place of supply", "place of supply state code", "pos"],
    "taxable_value": ["taxable value", "taxable amount", "value"],
    "cgst_amount": ["cgst", "cgst amt", "cgst amount"],
    "sgst_amount": ["sgst", "sgst amt", "sgst amount", "sgst/utgst", "sgst/utgst amt"],
    "igst_amount": ["igst", "igst amt", "igst amount"],
    "subtotal": ["sub total", "amount before tax", "net amount"],
    "tax": ["gst", "vat", "tax amount", "sales tax"],
    "total_amount": ["total", "grand total", "amount due", "balance due", "total value"],
    "amount_in_words": ["amount in words"],
}


def _normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", label.lower()).strip()


def _fuzzy_match(label: str, candidates: list, threshold: float = 0.82) -> bool:
    label = _normalize_label(label)
    return any(
        SequenceMatcher(None, label, _normalize_label(c)).ratio() >= threshold
        for c in candidates
    )


def map_label_to_field(raw_label: str):
    norm = _normalize_label(raw_label)

    # Pass 1: exact alias match, checked across ALL fields first. This
    # matters because some invoice terms are only 1 character apart
    # (e.g. "cgst amt" vs "igst amt" vs "sgst amt") — fuzzy matching alone
    # can misfire between them, so an exact match anywhere must win before
    # fuzzy matching is even considered.
    for field, aliases in ALIAS_DICTIONARY.items():
        if norm in {_normalize_label(alias) for alias in aliases}:
            return field

    # Pass 2: fuzzy match, but pick the SINGLE BEST match across all
    # fields (not just the first field that clears the threshold) —
    # otherwise dictionary ordering can pick a worse match first.
    # Guard: "rate" and "amount"/"amt" are financially very different
    # things on an invoice ("IGST Rate" = 18%, "IGST Amt" = ₹517.32) but
    # score deceptively close on plain string similarity — never let a
    # "rate" label fuzzy-match an alias that doesn't also say "rate".
    norm_has_rate = "rate" in norm.split()

    best_field, best_score = None, 0.0
    for field, aliases in ALIAS_DICTIONARY.items():
        for alias in aliases:
            alias_norm = _normalize_label(alias)
            if norm_has_rate and "rate" not in alias_norm.split():
                continue
            score = SequenceMatcher(None, norm, alias_norm).ratio()
            if score > best_score:
                best_field, best_score = field, score

    return best_field if best_score >= 0.87 else None


# =============================================================================
# CELL 4 — Run PP-OCRv6 on an invoice image/PDF
# =============================================================================

def create_ocr_pipeline():
    """Create the PP-OCRv6 pipeline once and reuse it across the batch.

    Uses PP-OCRv6_small_det (text detection) and PP-OCRv6_small_rec (text
    recognition) — both are lightweight models tuned for CPU inference.

    paddleocr>=2.9.0 removed the old use_gpu/enable_mkldnn/cpu_threads
    constructor arguments in favour of a single ``device`` keyword.
    Device is auto-detected: GPU when paddle is compiled with CUDA and a
    CUDA device is available, otherwise CPU.
    """
    if PaddleOCR is None:
        raise ImportError(
            "PaddleOCR is not installed. Run CELL 1 in Colab before OCR inference."
        )

    # Detect device — default to CPU; promote to GPU only when paddle was
    # built with CUDA support and a GPU is actually present.
    device = "cpu"
    try:
        import paddle
        if (
            paddle.device.is_compiled_with_cuda()
            and str(paddle.device.get_device()).startswith("gpu")
        ):
            device = "gpu"
    except Exception:
        pass

    kwargs = {
        "text_detection_model_name": "PP-OCRv6_small_det",
        "text_recognition_model_name": "PP-OCRv6_small_rec",
        "device": device,
        # enable_mkldnn=False forces run_mode="paddle" on CPU via paddleocr's
        # _build_paddle_static_engine_config() logic. When True (the default),
        # paddlex sets run_mode="mkldnn" which crashes on certain Intel CPU +
        # paddlepaddle 3.x combinations with:
        #   ConvertPirAttribute2RuntimeAttribute not support
        #   [pir::ArrayAttribute<pir::DoubleAttribute>]
        # This is the correct way to disable it — run_mode is not a valid
        # PaddleOCR() constructor argument and must not be passed directly.
        "enable_mkldnn": False,
        "use_angle_cls": True,   # auto-correct rotated text lines
        "lang": "en",
        # Larger side length catches small text on dense invoices without
        # over-blowing memory.  960 is a good balance for A4 at 300 DPI.
        "det_limit_side_len": 960,
        "rec_batch_num": 16,     # recognise up to 16 text crops in one forward pass
    }

    return PaddleOCR(**kwargs)


def _ocr_result_to_text(ocr_result) -> str:
    """Convert PP-OCRv6 predict() output to a plain-text string.

    PaddleOCR returns a list of pages; each page is either:
      - A list of lines: [ [bbox, (text, confidence)], ... ]   (classic API)
      - A dict with a ``rec_texts`` / ``rec_res`` key           (newer SDK)

    We only need the recognised text — coordinates and confidence scores are
    not used by the downstream parser.  Lines are joined with newlines so the
    label:value regex in parse_markdown_output() can find them.
    """
    if not ocr_result:
        return ""

    lines: list[str] = []

    def _extract_page(page):
        if page is None:
            return
        # Newer paddleocr SDK wraps each page result in a dict.
        if isinstance(page, dict):
            # 'rec_texts' is a flat list of recognised strings.
            texts = page.get("rec_texts") or page.get("rec_res") or []
            for item in texts:
                text = item[0] if isinstance(item, (list, tuple)) else str(item)
                text = text.strip()
                if text:
                    lines.append(text)
            return
        # Classic API: page is a list of [bbox, (text, score)] entries.
        if isinstance(page, list):
            for item in page:
                if not item:
                    continue
                # item = [bbox, (text, score)]
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    text_part = item[1]
                    if isinstance(text_part, (list, tuple)) and text_part:
                        text = str(text_part[0]).strip()
                    else:
                        text = str(text_part).strip()
                    if text:
                        lines.append(text)

    # ocr_result may be a single page or a list of pages.
    if isinstance(ocr_result, list) and ocr_result:
        # Detect whether it's a list-of-pages or a single page's line list.
        first = ocr_result[0]
        if first is None or isinstance(first, dict):
            for page in ocr_result:
                _extract_page(page)
        elif isinstance(first, list) and first and isinstance(first[0], list):
            # Outer list = pages, inner list = lines (classic multi-page).
            for page in ocr_result:
                _extract_page(page)
        else:
            # Single-page result — the outer list IS the line list.
            _extract_page(ocr_result)
    else:
        _extract_page(ocr_result)

    return "\n".join(lines)


def run_paddleocr_vl(
    input_path: str,
    output_dir: str = "output",
    pipeline=None,
    print_result: bool = False,
):
    """Run PP-OCRv6_small_det + PP-OCRv6_small_rec on an invoice image or PDF.

    The function signature and return value are identical to the previous
    PaddleOCR-VL implementation so all callers (extract_invoice, pdf_router,
    run_batch) continue to work without modification.

    Returns
    -------
    results      : list  – raw OCR output objects (one per page/image)
    output_dir   : str   – directory where per-page text files were saved
    timing       : dict  – model_load_s / inference_s / save_output_s
    markdown_pages: list[str] – plain text per page, consumed by the parser
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Model load is a one-time cost when pipeline is passed from run_batch().
    load_start = time.perf_counter()
    if pipeline is None:
        pipeline = create_ocr_pipeline()
    load_time = time.perf_counter() - load_start

    inference_start = time.perf_counter()
    # paddleocr>=3.x uses predict() instead of ocr(); angle classification
    # is handled automatically by the pipeline (no cls= argument needed).
    # Fall back to the old ocr(cls=True) API for paddleocr<3.x installs.
    if hasattr(pipeline, "predict"):
        raw_result = pipeline.predict(input_path)
    else:
        raw_result = pipeline.ocr(input_path, cls=True)
    inference_time = round(time.perf_counter() - inference_start, 2)

    save_start = time.perf_counter()

    # Normalise output: always a list-of-pages so multi-page PDFs work.
    # paddleocr 3.x predict() returns a generator of result objects; consume
    # it into a list so we can iterate multiple times and get a length.
    if raw_result is None:
        raw_result = []
    elif hasattr(raw_result, "__next__") or hasattr(raw_result, "__iter__") and not isinstance(raw_result, list):
        raw_result = list(raw_result)

    # paddleocr 3.x: each item in raw_result is a result object with
    # .rec_texts / .rec_scores attributes (one object per input image/page).
    # paddleocr <3.x: raw_result is a list of pages, each page a list of lines.
    # Detect which API we got and normalise to a flat list of page-text strings.
    results = raw_result
    markdown_pages: list[str] = []

    def _extract_text_from_predict_result(result_obj) -> str:
        """Extract plain text from a paddleocr 3.x predict() result object."""
        # Result object may have rec_texts as a direct attribute or in a dict.
        if hasattr(result_obj, "rec_texts"):
            return "\n".join(t for t in result_obj.rec_texts if t and t.strip())
        if hasattr(result_obj, "__dict__"):
            data = result_obj.__dict__
            texts = data.get("rec_texts") or data.get("rec_res") or []
            return "\n".join(
                (t[0] if isinstance(t, (list, tuple)) else str(t)).strip()
                for t in texts if t
            )
        # Fallback: try the existing converter
        return _ocr_result_to_text(result_obj)

    if raw_result and hasattr(raw_result[0], "rec_texts"):
        # paddleocr 3.x path — one result object per page/image
        for page_idx, page_result in enumerate(raw_result):
            page_text = _extract_text_from_predict_result(page_result)
            if page_text:
                markdown_pages.append(page_text)
            if print_result:
                print(f"--- Page {page_idx + 1} ---\n{page_text}\n")
            page_txt_path = Path(output_dir) / f"page_{page_idx + 1}.md"
            page_txt_path.write_text(page_text, encoding="utf-8")
    else:
        # paddleocr <3.x path — list of pages, each a list of [bbox, (text, score)]
        if raw_result and not isinstance(raw_result[0], list):
            raw_result = [raw_result]
        for page_idx, page_result in enumerate(raw_result):
            page_text = _ocr_result_to_text(page_result)
            if page_text:
                markdown_pages.append(page_text)
            if print_result:
                print(f"--- Page {page_idx + 1} ---\n{page_text}\n")
            page_txt_path = Path(output_dir) / f"page_{page_idx + 1}.md"
            page_txt_path.write_text(page_text, encoding="utf-8")

    # Save a compact JSON summary alongside the text output.
    summary = {
        "source": input_path,
        "pages": len(raw_result),
        "ocr_model": {
            "det": "PP-OCRv6_small_det",
            "rec": "PP-OCRv6_small_rec",
        },
    }
    Path(output_dir, "ocr_raw_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    gc.collect()
    save_time = round(time.perf_counter() - save_start, 2)

    timing = {
        "model_load_s": round(load_time, 2),
        "inference_s": inference_time,
        "save_output_s": save_time,
    }
    print(f"  [timing] load={timing['model_load_s']}s  "
          f"inference={timing['inference_s']}s  save={timing['save_output_s']}s")

    return results, output_dir, timing, markdown_pages


# =============================================================================
# CELL 5 — Parse plain-text OCR output into label/value pairs + tables
# =============================================================================
# PP-OCRv6 returns plain text lines.  The parser below handles both
# key:value lines and pipe-style Markdown tables (e.g. pdfplumber output on
# native PDFs).  HTML table parsing is retained for native-PDF paths that
# use pdfplumber, which can emit HTML table markup.  The parser expands HTML
# spans into a rectangular grid, combines parent headers ("IGST" + "Amt" ->
# "IGST Amt"), and keeps unknown tables for audit rather than silently
# dropping them.

LABEL_VALUE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*\*|__)?([^:|\n]{1,60}?)(?:\*\*|__)?\s*[:：]\s*(.+?)\s*$"
)
TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
TABLE_SEP_RE = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")

LINE_ITEM_HEADER_HINTS = [
    "description", "item", "hsn", "sac", "qty", "quantity", "rate",
    "amount", "price", "value", "sl no", "particulars", "discount",
    "transaction", "account no",
]


def _clean_cell(value) -> str:
    value = unescape(str(value or ""))
    # PaddleOCR/PaddleX occasionally serializes line breaks literally as the
    # two characters ``\\n``.  Convert them before whitespace normalization so
    # the next-label boundary logic can separate merged fields correctly.
    value = value.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", " ")
    value = re.sub(r"<br\s*/?>", " ", value, flags=re.I)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" |\t\r\n")


class _TableHTMLParser(HTMLParser):
    """Small dependency-free HTML table reader with span metadata."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.tables = []
        self.table = None
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        attrs = dict(attrs)
        if tag == "table":
            if self.depth == 0:
                self.table = []
            self.depth += 1
        elif self.depth and tag == "tr":
            self.row = []
        elif self.depth and tag in {"td", "th"}:
            self.cell = {
                "parts": [],
                "rowspan": max(1, int(attrs.get("rowspan", "1") or 1)),
                "colspan": max(1, int(attrs.get("colspan", "1") or 1)),
            }
        elif self.depth and tag == "br" and self.cell is not None:
            self.cell["parts"].append(" ")

    def handle_data(self, data):
        if self.depth and self.cell is not None:
            self.cell["parts"].append(data)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in {"td", "th"} and self.depth and self.cell is not None:
            if self.row is not None:
                self.row.append({
                    "text": _clean_cell("".join(self.cell["parts"])),
                    "rowspan": self.cell["rowspan"],
                    "colspan": self.cell["colspan"],
                })
            self.cell = None
        elif tag == "tr" and self.depth:
            if self.table is not None and self.row:
                self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.depth:
            self.depth -= 1
            if self.depth == 0 and self.table:
                self.tables.append(self.table)
                self.table = None


def _expand_html_table(raw_rows: list) -> list[list[str]]:
    """Expand rowspan/colspan cells so multi-level headers stay aligned."""
    expanded = []
    pending = {}  # column -> [remaining_rows, text]

    for raw_row in raw_rows:
        row = []
        col = 0

        def emit_pending(column):
            remaining, text = pending[column]
            row.append(text)
            remaining -= 1
            if remaining:
                pending[column][0] = remaining
            else:
                del pending[column]

        for cell in raw_row:
            while col in pending:
                emit_pending(col)
                col += 1

            text = _clean_cell(cell["text"])
            for offset in range(cell["colspan"]):
                row.append(text)
                if cell["rowspan"] > 1:
                    pending[col + offset] = [cell["rowspan"] - 1, text]
            col += cell["colspan"]

        if pending:
            last_pending_col = max(pending)
            while col <= last_pending_col:
                if col in pending:
                    emit_pending(col)
                else:
                    row.append("")
                col += 1

        expanded.append(row)

    width = max((len(row) for row in expanded), default=0)
    return [row + [""] * (width - len(row)) for row in expanded]


def _extract_html_tables(markdown_text: str) -> list[list[list[str]]]:
    parser = _TableHTMLParser()
    parser.feed(markdown_text)
    return [_expand_html_table(table) for table in parser.tables]


def _extract_markdown_tables(markdown_text: str) -> list[list[list[str]]]:
    """Extract pipe tables with or without optional outer ``|`` markers."""
    tables = []
    current = []

    for line in markdown_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("```"):
            if current:
                tables.append(current)
                current = []
            continue

        # A Markdown separator may be ``|---|---|`` or ``---|---``.
        possible_separator = stripped.strip("|").replace(":", "").replace("-", "").replace("|", "")
        if not possible_separator.strip():
            continue

        if stripped.count("|") >= 2:
            row_text = stripped.strip("|")
            current.append([_clean_cell(cell) for cell in row_text.split("|")])
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def _plain_text_from_markup(markdown_text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", markdown_text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</(?:tr|p|div|h[1-6]|table)\s*>", "\n", text, flags=re.I)
    text = re.sub(r"</(?:td|th)\s*>", " | ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text).replace("**", "").replace("__", "")
    text = re.sub(r"(?m)^\s*\|?[\s:\-|]+\|?\s*$", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_line_item_header(row: list[str]) -> bool:
    text = " ".join(_normalize_label(cell) for cell in row)
    score = sum(hint in text for hint in LINE_ITEM_HEADER_HINTS)
    return score >= 2 or ("description" in text and ("amount" in text or "value" in text))


def _is_line_item_data_row(row: list[str], header: list[str]) -> bool:
    if not any(_clean_cell(cell) for cell in row):
        return False
    first = _clean_cell(row[0]) if row else ""
    if re.fullmatch(r"\d{1,4}", first):
        return True

    # Some suppliers omit serial number.  A description plus an HSN/SAC-like
    # numeric code is enough to identify the row without admitting totals rows.
    hsn_indexes = [
        index for index, name in enumerate(header)
        if _normalize_label(name) in {"hsn", "sac", "hsnsac", "hsncode", "saccode"}
    ]
    description_indexes = [
        index for index, name in enumerate(header)
        if "description" in _normalize_label(name) or "particular" in _normalize_label(name)
    ]
    has_hsn = any(
        index < len(row) and re.search(r"\b\d{4,8}\b", _clean_cell(row[index]))
        for index in hsn_indexes
    )
    has_description = any(
        index < len(row) and bool(_clean_cell(row[index]))
        for index in description_indexes
    )
    return has_hsn and has_description


def _combine_header_rows(parent: list[str], child: list[str]) -> list[str]:
    width = max(len(parent), len(child))
    parent = parent + [""] * (width - len(parent))
    child = child + [""] * (width - len(child))
    combined = []

    for upper, lower in zip(parent, child):
        upper, lower = _clean_cell(upper), _clean_cell(lower)
        # PaddleOCR occasionally reads the compact heading ``Amt`` as ``Ant``.
        # Repair it only in the child header row, where the intent is clear.
        if _normalize_label(lower) == "ant":
            lower = "Amt"
        if not lower or _normalize_label(upper) == _normalize_label(lower):
            combined.append(upper or lower)
        elif _normalize_label(lower) in {"rate", "amt", "amount", "value"}:
            combined.append(f"{upper} {lower}".strip())
        else:
            combined.append(lower or upper)
    return combined


def _unique_headers(header: list[str]) -> list[str]:
    seen = {}
    unique = []
    for index, value in enumerate(header, start=1):
        base = _clean_cell(value) or f"column_{index}"
        seen[base] = seen.get(base, 0) + 1
        unique.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    return unique


KV_LABEL_HINTS = {
    "name", "address", "state", "gstin", "invoice no", "invoice number",
    "date", "date of invoice", "place of supply", "pos", "branch code",
    "reverse charge", "phone", "email", "amount in words", "grand total",
    "taxable", "taxable value",
}


def _looks_like_kv_label(value: str) -> bool:
    norm = _normalize_label(value.rstrip(":"))
    if not norm or len(norm) > 45:
        return False
    return bool(map_label_to_field(norm)) or any(hint == norm for hint in KV_LABEL_HINTS)


def _collect_table_kv(rows: list[list[str]], kv_pairs: list):
    """Collect 2-col and alternating label/value table cells."""
    for row in rows:
        if len(row) < 2:
            continue
        for index in range(0, len(row) - 1, 2):
            label, value = _clean_cell(row[index]), _clean_cell(row[index + 1])
            if label and value and _looks_like_kv_label(label):
                kv_pairs.append({"label": label.rstrip(":"), "value": value})


def _collect_inline_cell_kv(rows: list[list[str]], kv_pairs: list):
    """Collect ``Label: Value`` pairs contained inside individual cells.

    A merged HTML cell is repeated once per expanded column. Deduplicating
    here avoids pairing one repeated merged cell with another as though they
    were separate label/value columns.
    """
    seen = set()
    for row in rows:
        for cell in row:
            value = _clean_cell(cell)
            if not value or value in seen:
                continue
            seen.add(value)
            match = LABEL_VALUE_RE.match(value)
            if not match:
                continue
            label = _clean_cell(match.group(1)).rstrip(":")
            cell_value = _clean_cell(match.group(2))
            if label and cell_value and _looks_like_kv_label(label):
                kv_pairs.append({"label": label, "value": cell_value})


def _collect_column_kv(rows: list[list[str]], kv_pairs: list):
    """Collect a header row followed by its value row (tax summary tables)."""
    for header_row, value_row in zip(rows, rows[1:]):
        if len(header_row) < 2 or len(value_row) < 2:
            continue
        width = min(len(header_row), len(value_row))
        recognized = sum(
            _looks_like_kv_label(_clean_cell(header_row[index]))
            for index in range(width)
        )
        # Requiring at least two known labels prevents a normal line-item
        # header/data pair from being mistaken for invoice-level totals.
        if recognized < 2:
            continue
        for index in range(width):
            label = _clean_cell(header_row[index])
            value = _clean_cell(value_row[index])
            if label and value and _looks_like_kv_label(label):
                kv_pairs.append({"label": label.rstrip(":"), "value": value})


def _classify_table(rows: list[list[str]], kv_pairs: list, line_items: list, other_tables: list):
    rows = [[_clean_cell(cell) for cell in row] for row in rows if any(_clean_cell(c) for c in row)]
    if not rows:
        return

    # Some parsers wrap the entire invoice in one HTML table. In the supplied
    # Canara layout, nine metadata rows precede the item header, so a fixed
    # eight-row window silently misses a perfectly readable table.
    header_index = next(
        (index for index, row in enumerate(rows) if _is_line_item_header(row)),
        None,
    )

    if header_index is not None:
        header = rows[header_index]
        data_start = header_index + 1
        if data_start < len(rows):
            candidate = rows[data_start]
            candidate_text = " ".join(_normalize_label(cell) for cell in candidate)
            if not _is_line_item_data_row(candidate, header) and re.search(
                r"\b(rate|amt|ant|amount)\b", candidate_text
            ):
                header = _combine_header_rows(header, candidate)
                data_start += 1

        header = _unique_headers(header)
        found_item = False
        data_end = data_start
        for row_index, row in enumerate(rows[data_start:], start=data_start):
            padded = row + [""] * (len(header) - len(row))
            padded = padded[:len(header)]
            if _is_line_item_data_row(padded, header):
                line_items.append(dict(zip(header, padded)))
                found_item = True
                data_end = row_index + 1
            elif found_item:
                # Item rows are contiguous. Stop before totals/footer rows so
                # a later GST summary cannot be mistaken for another item.
                break

        # Preserve metadata and footer summaries without applying vertical
        # pairing to the item header plus its first data row. That old pairing
        # made row-one values look like invoice-level totals.
        _collect_inline_cell_kv(rows, kv_pairs)
        _collect_table_kv(rows[:header_index], kv_pairs)
        post_item_rows = rows[data_end:] if found_item else rows[data_start:]
        _collect_table_kv(post_item_rows, kv_pairs)
        _collect_column_kv(post_item_rows, kv_pairs)
        if not found_item:
            other_tables.append(rows)
        return

    before = len(kv_pairs)
    _collect_table_kv(rows, kv_pairs)
    _collect_column_kv(rows, kv_pairs)
    if len(kv_pairs) == before:
        other_tables.append(rows)


def parse_markdown_output(markdown_text: str) -> dict:
    kv_pairs = []
    line_items = []
    other_tables = []

    tables = _extract_html_tables(markdown_text) + _extract_markdown_tables(markdown_text)
    for table in tables:
        _classify_table(table, kv_pairs, line_items, other_tables)

    # Structured and converted Markdown sources may contain the same table.
    # Keep the first copy so invoice totals are never doubled.
    unique_items = []
    seen_items = set()
    for item in line_items:
        signature = tuple(
            (_normalize_label(str(key)), _clean_cell(value))
            for key, value in item.items()
        )
        if signature not in seen_items:
            unique_items.append(item)
            seen_items.add(signature)
    line_items = unique_items

    plain_text = _plain_text_from_markup(markdown_text)
    for line in plain_text.splitlines():
        kv_match = LABEL_VALUE_RE.match(line.strip(" |"))
        if kv_match:
            kv_pairs.append({"label": kv_match.group(1), "value": kv_match.group(2).strip()})

    return {
        "kv_pairs": kv_pairs,
        "line_items": line_items,
        "other_tables": other_tables,
        "raw_text": plain_text,
    }


# =============================================================================
# CELL 6 — Map parsed content to canonical schema -> JSON ready for form/DB
# =============================================================================

DATE_FORMATS = [
    "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d.%m.%Y",
    "%d %b %Y", "%d %B %Y", "%d-%b-%Y",
]


def _parse_date(value: str):
    from datetime import datetime
    value = _clean_cell(value)
    candidates = [value]
    date_match = re.search(
        r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b|"
        r"\b\d{1,2}[- ](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[- ]\d{4}\b",
        value,
        flags=re.I,
    )
    if date_match:
        candidates.insert(0, date_match.group(0))
    for candidate in candidates:
        for fmt in DATE_FORMATS:
            try:
                parsed = datetime.strptime(candidate.strip(), fmt)
                if 1990 <= parsed.year <= 2100:
                    return parsed.date().isoformat()
            except ValueError:
                continue
    return None


NUMBER_RE = re.compile(r"(?<![A-Z0-9])[-+]?\(?(?:₹|Rs\.?\s*)?[\d,]+(?:\.\d+)?\)?", re.I)


def _parse_float(value: str, prefer_last: bool = False):
    matches = NUMBER_RE.findall(str(value or ""))
    if not matches:
        return None
    raw = matches[-1] if prefer_last else matches[0]
    negative = raw.strip().startswith("(") and raw.strip().endswith(")")
    cleaned = re.sub(r"[^\d.\-]", "", raw.replace(",", ""))
    try:
        number = float(cleaned)
        return -abs(number) if negative else number
    except ValueError:
        return None


NUMBER_WORD_VALUES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
NUMBER_WORD_SCALES = {"thousand": 1_000, "lakh": 100_000, "lac": 100_000,
                      "crore": 10_000_000}


def _integer_from_words(value: str):
    """Parse common Indian invoice-number words without external packages."""
    tokens = re.findall(r"[a-z]+", str(value or "").lower().replace("-", " "))
    if not tokens:
        return None
    current = total = 0
    recognized = False
    for token in tokens:
        if token in {"and", "only", "rupee", "rupees", "paisa", "paise"}:
            continue
        if token in NUMBER_WORD_VALUES:
            current += NUMBER_WORD_VALUES[token]
            recognized = True
        elif token == "hundred":
            current = max(1, current) * 100
            recognized = True
        elif token in NUMBER_WORD_SCALES:
            total += max(1, current) * NUMBER_WORD_SCALES[token]
            current = 0
            recognized = True
    return total + current if recognized else None


def _amount_from_words(value: str):
    """Recover a total only when a numeric total was not parsed from the page."""
    text = _clean_cell(value).lower()
    separator = re.search(r"\brupees?\b", text, flags=re.I)
    if not separator:
        return None
    rupee_part = text[:separator.start()]
    paisa_part = text[separator.end():]
    rupees = _integer_from_words(rupee_part)
    if rupees is None:
        return None
    paisa_match = re.search(r"(.+?)\s+pais(?:a|e)\b", paisa_part, flags=re.I)
    paisa = _integer_from_words(paisa_match.group(1)) if paisa_match else 0
    if paisa is None or not 0 <= paisa <= 99:
        paisa = 0
    return round(float(rupees) + paisa / 100.0, 2)


GSTIN_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GSTIN_PATTERN = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]Z[0-9A-Z]$")


def _gstin_check_character(first_14: str):
    if len(first_14) != 14 or any(char not in GSTIN_ALPHABET for char in first_14):
        return None
    total, factor = 0, 1
    for char in first_14:
        product_value = GSTIN_ALPHABET.index(char) * factor
        total += product_value // 36 + product_value % 36
        factor = 2 if factor == 1 else 1
    return GSTIN_ALPHABET[(36 - total % 36) % 36]


def _is_valid_gstin(value: str) -> bool:
    return bool(
        value
        and GSTIN_PATTERN.fullmatch(value)
        and _gstin_check_character(value[:14]) == value[-1]
    )


def _repair_gstin(value: str):
    """Repair only position-constrained OCR confusions and verify checksum."""
    raw = re.sub(r"[^0-9A-Z]", "", str(value or "").upper())
    if len(raw) != 15:
        embedded = re.search(r"(?<![0-9A-Z])[0-9A-Z]{15}(?![0-9A-Z])", str(value or "").upper())
        if not embedded:
            return None
        raw = embedded.group(0)

    digit_fix = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
    letter_fix = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}
    chars = list(raw)
    for index in {0, 1, 7, 8, 9, 10}:
        chars[index] = digit_fix.get(chars[index], chars[index])
    for index in {2, 3, 4, 5, 6, 11}:
        chars[index] = letter_fix.get(chars[index], chars[index])
    if chars[13] == "2":
        chars[13] = "Z"
    base = "".join(chars)
    if not GSTIN_PATTERN.fullmatch(base):
        return None
    if _is_valid_gstin(base):
        return base

    # Entity code (position 13 in human counting) is alphanumeric, where I/1
    # and O/0 are common OCR mistakes.  Try only those visually-confusable
    # alternatives and accept a repair only when the checksum proves it.
    confusable = {
        "I": ["I", "1"], "1": ["1", "I"],
        "O": ["O", "0"], "0": ["0", "O"],
        "Z": ["Z", "2"], "2": ["2", "Z"],
        "S": ["S", "5"], "5": ["5", "S"],
        "B": ["B", "8"], "8": ["8", "B"],
        "G": ["G", "6"], "6": ["6", "G"],
    }
    options = [confusable.get(base[12], [base[12]])]
    for (entity_code,) in product(*options):
        trial_first_14 = base[:12] + entity_code + base[13]
        expected = _gstin_check_character(trial_first_14)
        trial = trial_first_14 + (expected or base[14])
        if _is_valid_gstin(trial):
            observed_checksum_options = confusable.get(base[14], [base[14]])
            if expected in observed_checksum_options or base[14] == expected:
                return trial
    return base  # syntactically usable, but validation will flag its checksum


def _extract_gstins(raw_text: str) -> list[tuple[int, str]]:
    found = []
    for match in re.finditer(r"(?<![0-9A-Z])[0-9A-Z]{15}(?![0-9A-Z])", raw_text.upper()):
        repaired = _repair_gstin(match.group(0))
        if repaired and repaired not in [value for _, value in found]:
            found.append((match.start(), repaired))
    return found


def _canonical_line_key(raw_header: str) -> str:
    norm = _normalize_label(re.sub(r"_\d+$", "", raw_header))
    if norm in {"sl no", "s no", "sr no", "serial no", "serial number"}:
        return "serial_number"
    if "description" in norm or "particular" in norm or norm == "item":
        return "description"
    if "account" in norm:
        return "account_number"
    if "transaction" in norm or "unique" in norm:
        return "transaction_number"
    if norm in {"hsn", "sac", "hsnsac", "hsncode", "saccode"}:
        return "hsn_sac"
    if "quantity" in norm or norm == "qty":
        return "quantity"
    if "discount" in norm:
        return "discount"
    for tax_name in ("cgst", "sgst", "utgst", "igst"):
        if tax_name in norm:
            prefix = "sgst" if tax_name == "utgst" else tax_name
            if "rate" in norm or "%" in raw_header:
                return f"{prefix}_rate"
            if "amt" in norm or "ant" in norm or "amount" in norm:
                return f"{prefix}_amount"
    if "taxable" in norm:
        return "taxable_value"
    if "total" in norm and ("value" in norm or "amount" in norm):
        return "total_value"
    if norm in {"unit rate", "price", "rate"}:
        return "unit_rate"
    if norm in {"amount", "value"}:
        return "amount"
    return re.sub(r"[^a-z0-9]+", "_", norm).strip("_") or "column"


LINE_ITEM_NUMERIC_FIELDS = {
    "quantity", "discount", "taxable_value", "unit_rate", "amount",
    "cgst_rate", "cgst_amount", "sgst_rate", "sgst_amount",
    "igst_rate", "igst_amount", "total_value",
}


def _normalize_line_items(raw_items: list[dict]) -> list[dict]:
    normalized = []
    for raw_item in raw_items:
        item = {}
        for raw_header, raw_value in raw_item.items():
            key = _canonical_line_key(raw_header)
            if key in item:
                suffix = 2
                while f"{key}_{suffix}" in item:
                    suffix += 1
                key = f"{key}_{suffix}"
            value = _clean_cell(raw_value)
            item[key] = _parse_float(value) if key in LINE_ITEM_NUMERIC_FIELDS else value
        if any(value not in (None, "", 0.0) for value in item.values()):
            normalized.append(item)
    return normalized


def _fill_from_line_items(record: dict, line_items: list):
    """
    Some invoices (this one included) put summary totals inside the
    line-items table itself — either as a dedicated totals row (no Sl No /
    description, just numbers), or as a column that should be summed across
    all rows. This fills still-missing numeric fields from that table
    rather than leaving them null just because they weren't a separate
    "Label: Value" line.
    """
    if not line_items:
        return

    mapping = {
        "taxable_value": ["taxable_value", "amount"],
        "cgst_amount": ["cgst_amount"],
        "sgst_amount": ["sgst_amount"],
        "igst_amount": ["igst_amount"],
        "total_amount": ["total_value"],
    }
    for field, candidate_keys in mapping.items():
        if record.get(field) is not None:
            continue
        for key in candidate_keys:
            values = [item.get(key) for item in line_items]
            values = [value for value in values if isinstance(value, (int, float))]
            if values:
                record[field] = round(sum(values), 2)
                break


def _find_labeled_text(raw_text: str, labels: list[str]):
    label_group = "|".join(labels)
    match = re.search(
        rf"(?im)(?:{label_group})\s*[:：]\s*([^|\n]+)",
        raw_text,
    )
    return _clean_cell(match.group(1)) if match else None


def _find_labeled_amount(raw_text: str, labels: list[str], prefer_last: bool = False):
    label_group = "|".join(labels)
    # Permit cell separators between words (e.g. "Grand | Total | 870.84")
    label_group = label_group.replace(r"\s+", r"[\s|]+")
    match = re.search(
        rf"(?im)(?:{label_group})\s*[:：]?\s*\|?\s*"
        rf"((?:₹|Rs\.?\s*)?[\d,]+(?:\.\d+)?)",
        raw_text,
    )
    return _parse_float(match.group(1), prefer_last=prefer_last) if match else None


def _find_tax_amount(raw_text: str, tax_name: str):
    for line in raw_text.splitlines():
        if re.search(rf"\b{re.escape(tax_name)}\b", line, flags=re.I):
            numbers = [_parse_float(number) for number in NUMBER_RE.findall(line)]
            numbers = [number for number in numbers if number is not None]
            if len(numbers) >= 2:
                return numbers[-1]
    return None


def _invoice_number_from_filename(source_file: str):
    stem = Path(source_file).stem
    match = re.search(r"(?i)CBSInvoice-([^-_]+)", stem)
    if match:
        return match.group(1).upper()
    return None


GENERIC_INVOICE_VALUES = {
    "invoice", "tax invoice", "invoice no", "invoice number", "original",
    "original for recipient", "recipient copy",
}
GENERIC_NAME_VALUES = {
    "name", "customer", "recipient", "details of recipient",
    "recipient details", "customer details", "bill to", "buyer",
}


def _is_usable_invoice_number(value: str) -> bool:
    cleaned = _clean_cell(value).upper().strip(".:- ")
    if _normalize_label(cleaned) in {_normalize_label(item) for item in GENERIC_INVOICE_VALUES}:
        return False
    # A real invoice identifier must contain at least one digit.  This rejects
    # headings such as "INVOICE" while allowing numeric-only bill numbers.
    return len(cleaned) >= 4 and bool(re.search(r"\d", cleaned))


def _is_usable_customer_name(value: str) -> bool:
    cleaned = _clean_cell(value).strip(".:- ")
    norm = _normalize_label(cleaned)
    if not norm or norm in {_normalize_label(item) for item in GENERIC_NAME_VALUES}:
        return False
    if re.fullmatch(r"(?:details?|particulars?)\s+of\s+(?:recipient|customer|buyer)", norm):
        return False
    if re.search(r"\b(?:gstin|address|state|invoice|date|phone|email)\b\s*[:：]", cleaned, re.I):
        return False
    return len(cleaned) >= 3


def _find_customer_name(raw_text: str):
    """Find a real recipient name while rejecting section headings."""
    label_pattern = re.compile(
        r"(?im)(?:recipient\s+name|customer\s+name|\bname)\s*[:：|]\s*([^|\n]+)"
    )
    for match in label_pattern.finditer(raw_text):
        candidate = _clean_string_field("customer_name", match.group(1))
        if _is_usable_customer_name(candidate):
            return candidate

    # Layout engines occasionally place "Details of Recipient" between the
    # label and its value.  Search the recipient section for a company-like
    # line rather than accepting that heading as the value.
    section_match = re.search(r"details?\s+of\s+recipient", raw_text, flags=re.I)
    section = raw_text[section_match.end():] if section_match else raw_text
    section = re.split(
        r"(?im)^\s*(?:sl\s*no|goods/service|description|amount\s+in\s+words)\b",
        section,
        maxsplit=1,
    )[0]
    for line in section.splitlines():
        candidate = _clean_cell(line.strip("#* |\""))
        candidate = re.sub(r"^(?:name\s*[:：|]\s*)", "", candidate, flags=re.I)
        if (
            _is_usable_customer_name(candidate)
            and re.search(r"\b(?:limited|ltd|pvt|llp|company|corporation|bank)\b", candidate, re.I)
        ):
            return candidate
    return None


def _guess_vendor_name(raw_text: str):
    # High-confidence common heading in the supplied regression invoices.
    if re.search(r"\bCanara\s+Bank\b", raw_text, flags=re.I):
        return "Canara Bank"

    before_invoice = re.split(r"\btax\s+invoice\b", raw_text, maxsplit=1, flags=re.I)[0]
    candidates = []
    for line in before_invoice.splitlines():
        value = _clean_cell(line.strip("# |"))
        if not value or len(value) > 100 or re.search(r"original for|gstin|invoice", value, re.I):
            continue
        if re.search(r"\b(bank|limited|ltd|pvt|llp|company|corporation|enterprise)\b", value, re.I):
            candidates.append(value)
    return min(candidates, key=len) if candidates else None


def _validation_results(record: dict, raw_text: str = "", field_sources: dict = None) -> list[dict]:
    issues = []
    field_sources = field_sources or {}
    for field in ("vendor_gstin", "customer_gstin"):
        value = record.get(field)
        if value and not _is_valid_gstin(value):
            issues.append({
                "code": "invalid_gstin_checksum",
                "field": field,
                "message": f"{field} failed GSTIN format/checksum validation.",
            })

    taxable = record.get("taxable_value")
    total = record.get("total_amount")
    tax_values = [record.get(key) for key in ("cgst_amount", "sgst_amount", "igst_amount")]
    present_tax_values = [value for value in tax_values if isinstance(value, (int, float))]
    if isinstance(taxable, (int, float)) and isinstance(total, (int, float)) and present_tax_values:
        expected = round(taxable + sum(present_tax_values), 2)
        difference = round(total - expected, 2)
        if abs(difference) > 0.05:
            issues.append({
                "code": "total_arithmetic_mismatch",
                "field": "total_amount",
                "expected": expected,
                "extracted": total,
                "difference": difference,
                "message": "Printed total does not equal taxable value plus extracted GST amounts.",
            })

    if (
        not record.get("line_items")
        and re.search(r"\b(?:hsn\s*/?\s*sac|goods\s*/?\s*service|taxable\s+value)\b", raw_text, re.I)
    ):
        issues.append({
            "code": "line_item_table_not_parsed",
            "field": "line_items",
            "message": "An item-table heading is visible in OCR text, but no line-item row was parsed.",
        })

    if field_sources.get("total_amount") == "amount_in_words_derived":
        issues.append({
            "code": "total_derived_from_words",
            "field": "total_amount",
            "message": "Numeric total was not parsed; total_amount was recovered from Amount In Words.",
        })
    return issues


def _clean_string_field(field: str, value: str):
    value = _clean_cell(value)
    if field in {"vendor_gstin", "customer_gstin"}:
        return _repair_gstin(value)
    if field == "invoice_number":
        match = re.search(r"[A-Z0-9][A-Z0-9/-]{5,}", value.upper())
        return match.group(0) if match else None

    # Defensive cleanup for a value that arrives as ``Name: Company`` rather
    # than just ``Company``.
    if field == "customer_name":
        value = re.sub(
            r"^(?:recipient\s+name|customer\s+name|name)\s*[:：|]\s*",
            "",
            value,
            flags=re.I,
        )

    # Plain OCR sometimes merges the left and right columns into one line,
    # e.g. "Name: Bajaj ... State: 26".  Stop at the next known label.
    if field in {"customer_name", "vendor_name", "place_of_supply"}:
        value = re.split(
            r"\s+(?=(?:state|invoice\s*(?:no|number)|date|branch\s+code|"
            r"reverse\s+charge(?:\s+applicable)?|gstin|pos|phone|email)\s*[:：])",
            value,
            maxsplit=1,
            flags=re.I,
        )[0]
    return _clean_cell(value)


def build_form_ready_json(parsed: dict, source_file: str) -> dict:
    """
    Produces JSON keyed EXACTLY to canonical form/DB field names, so it can
    be loaded directly into form inputs (form_field_name -> value) without
    any further remapping downstream.
    """
    record = {field: None for field in CANONICAL_SCHEMA}
    field_sources = {}

    for pair in parsed["kv_pairs"]:
        field = map_label_to_field(pair["label"])
        if not field:
            continue
        spec = CANONICAL_SCHEMA[field]
        if spec["type"] == "date":
            value = _parse_date(pair["value"])
        elif spec["type"] == "float":
            value = _parse_float(
                pair["value"],
                prefer_last=field in {"cgst_amount", "sgst_amount", "igst_amount"},
            )
        else:
            value = _clean_string_field(field, pair["value"])
        if field == "invoice_number" and value and not _is_usable_invoice_number(value):
            value = None
        if field == "customer_name" and value and not _is_usable_customer_name(value):
            value = None
        if value not in (None, "") and record.get(field) in (None, ""):
            record[field] = value
            field_sources[field] = "label_value"

    raw_text = parsed.get("raw_text", "")

    # Filename is a strong fallback for bank-generated batches.  Use it when
    # OCR missed the number, or when it is an obvious near-match with an OCR
    # substitution (e.g. O/0).
    filename_invoice_number = _invoice_number_from_filename(source_file)
    if filename_invoice_number:
        current = str(record.get("invoice_number") or "")
        if (
            not _is_usable_invoice_number(current)
            or SequenceMatcher(None, current, filename_invoice_number).ratio() >= 0.70
        ):
            record["invoice_number"] = filename_invoice_number
            field_sources["invoice_number"] = "source_filename"

    if not record.get("invoice_number"):
        value = _find_labeled_text(raw_text, [r"invoice\s*(?:no\.?|number|#)"])
        if value:
            match = re.search(r"[A-Z0-9][A-Z0-9/-]{5,}", value.upper())
            if match:
                record["invoice_number"] = match.group(0)
                field_sources["invoice_number"] = "text_fallback"

    if not record.get("invoice_date"):
        value = _find_labeled_text(
            raw_text,
            [r"date\s+of\s+invoice", r"invoice\s+date", r"date"],
        )
        parsed_date = _parse_date(value or "")
        if parsed_date:
            record["invoice_date"] = parsed_date
            field_sources["invoice_date"] = "text_fallback"

    if not _is_usable_customer_name(record.get("customer_name") or ""):
        record["customer_name"] = None
        value = _find_customer_name(raw_text)
        if value:
            record["customer_name"] = value
            field_sources["customer_name"] = "text_fallback"

    if not record.get("vendor_name"):
        value = _guess_vendor_name(raw_text)
        if value:
            record["vendor_name"] = value
            field_sources["vendor_name"] = "heading_fallback"

    if not record.get("place_of_supply"):
        value = _find_labeled_text(raw_text, [r"place\s+of\s+supply(?:\s*\(state\s*code\))?", r"pos"])
        if value:
            record["place_of_supply"] = _clean_string_field("place_of_supply", value)
            field_sources["place_of_supply"] = "text_fallback"

    if not record.get("amount_in_words"):
        value = _find_labeled_text(raw_text, [r"amount\s+in\s+words"])
        if value:
            record["amount_in_words"] = value
            field_sources["amount_in_words"] = "text_fallback"

    # Resolve GSTINs by explicit labels first, then by reading order.  This is
    # what distinguishes the generic header GSTIN and recipient GSTIN in the
    # Silvassa (26BAG...) layout.
    for field, label_patterns in {
        "vendor_gstin": [r"gstin\s+of\s+supplier", r"supplier\s+gstin"],
        "customer_gstin": [r"customer\s+gstin", r"recipient\s+gstin", r"gstin\s+of\s+recipient"],
    }.items():
        if not record.get(field):
            value = _find_labeled_text(raw_text, label_patterns)
            repaired = _repair_gstin(value or "")
            if repaired:
                record[field] = repaired
                field_sources[field] = "explicit_gstin_label"

    gstins = _extract_gstins(raw_text)
    recipient_position_match = re.search(r"details\s+of\s+recipient", raw_text, flags=re.I)
    recipient_position = recipient_position_match.start() if recipient_position_match else None
    if not record.get("vendor_gstin"):
        before_recipient = [value for position, value in gstins if recipient_position is None or position < recipient_position]
        if before_recipient:
            record["vendor_gstin"] = before_recipient[0]
            field_sources["vendor_gstin"] = "gstin_reading_order"
    if not record.get("customer_gstin"):
        candidates = [
            value for position, value in gstins
            if value != record.get("vendor_gstin")
            and (recipient_position is None or position > recipient_position)
        ]
        if not candidates:
            candidates = [value for _, value in gstins if value != record.get("vendor_gstin")]
        if candidates:
            record["customer_gstin"] = candidates[0]
            field_sources["customer_gstin"] = "gstin_reading_order"

    # Normalize any GSTIN already obtained from a labelled table.
    for field in ("vendor_gstin", "customer_gstin"):
        repaired = _repair_gstin(record.get(field) or "")
        if repaired:
            record[field] = repaired

    normalized_items = _normalize_line_items(parsed["line_items"])
    record["line_items"] = normalized_items
    before_line_item_fill = {
        field: record.get(field)
        for field in ("taxable_value", "cgst_amount", "sgst_amount", "igst_amount", "total_amount")
    }
    _fill_from_line_items(record, normalized_items)
    for field, old_value in before_line_item_fill.items():
        if old_value is None and record.get(field) is not None:
            field_sources[field] = "line_items"

    numeric_fallbacks = {
        "taxable_value": ([r"taxable(?:\s+value)?"], False),
        "total_amount": ([r"grand\s+total", r"amount\s+due"], False),
    }
    for field, (labels, prefer_last) in numeric_fallbacks.items():
        if record.get(field) is None:
            value = _find_labeled_amount(raw_text, labels, prefer_last=prefer_last)
            if value is not None:
                record[field] = value
                field_sources[field] = "text_fallback"

    for field, tax_name in {
        "cgst_amount": "cgst",
        "sgst_amount": "sgst",
        "igst_amount": "igst",
    }.items():
        if record.get(field) is None:
            value = _find_tax_amount(raw_text, tax_name)
            if value is not None:
                record[field] = value
                field_sources[field] = "text_fallback"

    # A readable Amount In Words line is a useful final fallback for the grand
    # total.  Keep its provenance explicit and require review; it must not be
    # confused with a directly printed numeric total.
    if record.get("total_amount") is None and record.get("amount_in_words"):
        value = _amount_from_words(record["amount_in_words"])
        if value is not None:
            record["total_amount"] = value
            field_sources["total_amount"] = "amount_in_words_derived"

    # GST invoices normally use either IGST or the CGST+SGST/UTGST pair for a
    # line.  Filling the mutually-exclusive absent component with zero keeps
    # JSON/DB output numeric without inventing a non-zero tax value.
    if isinstance(record.get("igst_amount"), (int, float)) and record["igst_amount"] > 0:
        for field in ("cgst_amount", "sgst_amount"):
            if record.get(field) is None:
                record[field] = 0.0
                field_sources[field] = "inferred_tax_mode"
    elif any(isinstance(record.get(field), (int, float)) for field in ("cgst_amount", "sgst_amount")):
        if record.get("igst_amount") is None:
            record["igst_amount"] = 0.0
            field_sources["igst_amount"] = "inferred_tax_mode"

    present_tax_values = [
        record.get(field) for field in ("cgst_amount", "sgst_amount", "igst_amount")
        if isinstance(record.get(field), (int, float))
    ]
    if record.get("tax") is None and present_tax_values:
        record["tax"] = round(sum(present_tax_values), 2)
        field_sources["tax"] = "calculated_from_components"

    if record.get("subtotal") is None and isinstance(record.get("taxable_value"), (int, float)):
        record["subtotal"] = record["taxable_value"]
        field_sources["subtotal"] = "inferred_from_taxable_value"

    validation_issues = _validation_results(record, raw_text, field_sources)

    # Recompute which required fields are still genuinely missing/unparsed
    # AFTER the line-items fallback has had a chance to fill numeric totals
    # A missing optional due date is normal when the invoice never prints one.
    # Flag only required fields plus expected GST/table values that OCR saw but
    # the parser failed to map.
    confidence_flags = [
        field for field, spec in CANONICAL_SCHEMA.items()
        if spec["required"] and not record.get(field)
    ]
    table_visible = bool(re.search(
        r"\b(?:hsn\s*/?\s*sac|goods\s*/?\s*service|taxable\s+value)\b",
        raw_text,
        flags=re.I,
    ))
    if table_visible:
        for field in ("taxable_value", "cgst_amount", "sgst_amount", "igst_amount"):
            if record.get(field) is None and field not in confidence_flags:
                confidence_flags.append(field)
        if not record.get("line_items"):
            confidence_flags.append("line_items")

    missing_required = [
        f for f, spec in CANONICAL_SCHEMA.items()
        if spec["required"] and not record.get(f)
    ]

    return {
        "source_file": source_file,
        "fields": record,               # <- load this dict straight into your form
        "field_sources": field_sources,
        "needs_review": bool(missing_required or validation_issues),
        "missing_required": missing_required,
        "unparsed_or_low_confidence": confidence_flags,
        "validation_issues": validation_issues,
    }


# =============================================================================
# CELL 7 — Optional XML export (only if your system needs XML instead of JSON)
# =============================================================================

def to_xml(form_ready: dict) -> str:
    import xml.etree.ElementTree as ET

    root = ET.Element("Invoice", source_file=form_ready["source_file"])
    fields_el = ET.SubElement(root, "Fields")
    for key, value in form_ready["fields"].items():
        if key == "line_items":
            continue
        el = ET.SubElement(fields_el, key)
        el.text = "" if value is None else str(value)

    items_el = ET.SubElement(root, "LineItems")
    for item in form_ready["fields"].get("line_items", []):
        item_el = ET.SubElement(items_el, "Item")
        for k, v in item.items():
            cell_el = ET.SubElement(item_el, re.sub(r"\W+", "_", k) or "col")
            cell_el.text = "" if v is None else str(v)

    return ET.tostring(root, encoding="unicode")


# =============================================================================
# CELL 8 — Full pipeline, end to end
# =============================================================================

def extract_invoice(
    input_path: str,
    output_dir: str = "output",
    export_xml: bool = False,
    pipeline=None,
    shared_model_load_s: float = None,
):
    pipeline_start = time.perf_counter()

    _, out_dir, ocr_timing, markdown_pages = run_paddleocr_vl(
        input_path,
        output_dir,
        pipeline=pipeline,
    )

    # Prefer text directly attached to this prediction result.  This avoids
    # reading a stale .md from an earlier run.  The file fallback remains for
    # debugging or callers that construct text files externally.
    if markdown_pages:
        markdown_text = "\n\n".join(markdown_pages)
    else:
        md_files = sorted(Path(out_dir).glob("*.md"))
        if not md_files:
            raise FileNotFoundError(f"No Markdown output found in {out_dir}")
        markdown_text = "\n\n".join(
            path.read_text(encoding="utf-8") for path in md_files
        )

    parse_start = time.perf_counter()
    parsed = parse_markdown_output(markdown_text)
    form_ready = build_form_ready_json(parsed, source_file=input_path)
    parse_time = time.perf_counter() - parse_start

    # Keep the exact parser input and a compact diagnostic beside the final
    # JSON.  If a future invoice layout fails, these two small files are enough
    # to improve the parser without rerunning the GPU model.
    Path(out_dir, "parser_input.md").write_text(markdown_text, encoding="utf-8")
    diagnostics = {
        "kv_pair_count": len(parsed.get("kv_pairs", [])),
        "line_item_count": len(parsed.get("line_items", [])),
        "unclassified_table_count": len(parsed.get("other_tables", [])),
        "unclassified_tables": parsed.get("other_tables", []),
    }
    Path(out_dir, "parser_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )

    total_time = time.perf_counter() - pipeline_start

    # Attach timing to the result — this is what tells you where the time
    # is actually going: almost always dominated by "inference_s", with
    # parsing/mapping (your own code) taking a tiny fraction by comparison.
    form_ready["timing"] = {
        **ocr_timing,
        "parse_and_map_s": round(parse_time, 2),
        "total_s": round(total_time, 2),
    }
    if shared_model_load_s is not None:
        form_ready["timing"]["shared_batch_model_load_s"] = shared_model_load_s

    # Write only after timing/validation metadata has been attached so the
    # downloaded per-invoice JSON matches the in-memory batch summary.
    json_out_path = Path(out_dir) / "invoice_extracted.json"
    json_out_path.write_text(json.dumps(form_ready, indent=2), encoding="utf-8")

    if export_xml:
        xml_out_path = Path(out_dir) / "invoice_extracted.xml"
        xml_out_path.write_text(to_xml(form_ready), encoding="utf-8")

    return form_ready


# =============================================================================
# CELL 9 — Run it (Colab-ready: upload, run, summarize, download)
# =============================================================================
# Fixes / additions vs the draft version:
#  - `extract_invoice` used Path(out_dir).glob("*.md") and grabbed the FIRST
#    match — fine for one file, but breaks once you process more than one
#    invoice into the same output dir (results get mixed up / overwritten).
#    Fixed below by giving each invoice its own output subfolder.
#  - Added an actual file-upload step (Colab's file picker), since the
#    original script assumed a file already sitting in the working directory.
#  - Added batch handling for multiple uploaded invoices in one run.
#  - Added a clean printed summary per invoice instead of a raw JSON dump.
#  - Added a download step so you actually get the JSON/XML files back out
#    of the Colab VM (files inside Colab disappear when the runtime resets).

def _get_output_subdir(input_path: str, base_dir: str = "output") -> str:
    """Give each invoice its own output folder, keyed by filename, so
    batch runs don't overwrite each other's JSON/Markdown output."""
    stem = Path(input_path).stem
    return str(Path(base_dir) / stem)


def _release_unused_gpu_cache():
    """Release only unreferenced Paddle GPU buffers; installed packages stay untouched."""
    gc.collect()
    try:
        import paddle

        if (
            paddle.device.is_compiled_with_cuda()
            and str(paddle.device.get_device()).startswith("gpu")
        ):
            paddle.device.cuda.empty_cache()
    except Exception:
        # Cleanup must never turn a successfully extracted invoice into an
        # error, and CPU-only environments do not need this operation.
        pass


def run_batch(input_paths: list[str], export_xml: bool = False) -> list[dict]:
    """Runs extract_invoice() over multiple files and returns all results."""
    all_results = []
    batch_start = time.perf_counter()

    if not input_paths:
        return all_results

    model_load_start = time.perf_counter()
    pipeline = create_ocr_pipeline()
    shared_model_load_s = round(time.perf_counter() - model_load_start, 2)
    print(f"OCR model loaded once in {shared_model_load_s}s for {len(input_paths)} invoice(s).")

    for path in input_paths:
        print(f"\n{'='*60}\nProcessing: {path}\n{'='*60}")
        out_dir = _get_output_subdir(path)
        try:
            result = extract_invoice(
                path,
                output_dir=out_dir,
                export_xml=export_xml,
                pipeline=pipeline,
                shared_model_load_s=shared_model_load_s,
            )
        except Exception as e:
            print(f"[ERROR] Failed to process {path}: {e}")
            all_results.append({
                "source_file": path,
                "fields": None,
                "needs_review": True,
                "error": str(e),
            })
            _release_unused_gpu_cache()
            continue

        all_results.append(result)
        _print_summary(result, out_dir=out_dir)
        _release_unused_gpu_cache()

    batch_total = time.perf_counter() - batch_start
    succeeded = [r for r in all_results if r.get("fields") is not None]
    avg_time = batch_total / len(succeeded) if succeeded else 0

    print(f"\n{'='*60}")
    print(f"Batch complete: {len(succeeded)}/{len(input_paths)} succeeded")
    print(f"Total time: {batch_total:.2f}s   Avg per invoice: {avg_time:.2f}s")
    print(f"{'='*60}")

    return all_results


def _print_summary(result: dict, out_dir: str = None):
    fields = result["fields"]
    print(f"\nExtracted fields for: {result['source_file']}")
    if out_dir:
        print(f"  (output saved to: {out_dir}/)")
    for key, value in fields.items():
        if key == "line_items":
            continue
        flag = "  <-- MISSING" if value is None and CANONICAL_SCHEMA[key]["required"] else ""
        print(f"  {key:16s}: {value}{flag}")

    if fields.get("line_items"):
        print(f"  line_items      : {len(fields['line_items'])} row(s) detected")

    if result["needs_review"]:
        print(f"\n  ⚠ NEEDS REVIEW — missing: {result['missing_required']}, "
              f"unparsed: {result['unparsed_or_low_confidence']}")
        for issue in result.get("validation_issues", []):
            print(f"    - {issue['code']}: {issue['message']}")
    else:
        print("\n  ✓ Ready to auto-fill form / insert into DB directly")


if __name__ == "__main__":
    # --- Option A: running in Google Colab -> upload invoice file(s) ---
    try:
        from google.colab import files as colab_files  # noqa: F401
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    if IN_COLAB:
        print("Upload one or more invoice images/PDFs:")
        uploaded = colab_files.upload()          # opens Colab's file picker
        input_paths = list(uploaded.keys())      # uploaded filenames, as-is
    else:
        # --- Option B: running locally -> point at real file(s) yourself ---
        input_paths = ["invoice1.jpg"]            # replace with your file(s)

    all_results = run_batch(input_paths, export_xml=True)

    # Save a combined summary JSON of every invoice processed this run
    combined_path = Path("output") / "batch_summary.json"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved combined summary: {combined_path}")

    # In Colab, download one archive containing final JSON/XML plus raw
    # Markdown/result blocks and diagnostics.  One archive avoids a browser
    # download prompt for every file and retains everything needed to debug a
    # new invoice layout without rerunning the GPU model.
    if IN_COLAB:
        import shutil

        archive_path = shutil.make_archive(
            "invoice_ocr_results",
            "zip",
            root_dir="output",
        )
        print(f"Saved complete result archive: {archive_path}")
        colab_files.download(archive_path)
