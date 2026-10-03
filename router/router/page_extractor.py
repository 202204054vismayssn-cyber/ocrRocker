"""Per-page extraction: runs the PaddleOCR path and attaches routing metadata.

``extract_native_page`` lives in ``pdf_router.py`` rather than here because it
resolves ``pdfplumber`` from that module's globals, which is what callers and
tests patch.  The functions here take their dependencies from ``_modules``,
so patching ``ocr.<name>`` works from anywhere.
"""

from __future__ import annotations

from pathlib import Path

from router._modules import native_parser, ocr
from router.batch import MULTI_PAGE_HANDLING
from router.page_detector import OCRPipelineManager


def _run_scanned_page(
    image_path: str,
    page_output_dir: Path,
    pipeline_manager: OCRPipelineManager,
) -> dict:
    """Run PaddleOCR on a rendered page image, reusing the shared pipeline."""
    pipeline = pipeline_manager.get()
    try:
        return ocr.extract_invoice(
            image_path,
            output_dir=str(page_output_dir),
            export_xml=False,
            pipeline=pipeline,
            shared_model_load_s=pipeline_manager.load_seconds,
        )
    finally:
        # Releases only unused GPU buffers; does not uninstall packages or
        # unload the shared model instance.
        ocr._release_unused_gpu_cache()


def _read_ocr_text(page_output_dir: Path) -> str:
    """Read PaddleOCR markdown output for the cheap Qwen text tier."""
    preferred = page_output_dir / "parser_input.md"
    candidates = [preferred] if preferred.is_file() else sorted(page_output_dir.glob("*.md"))
    for candidate in candidates:
        try:
            text = candidate.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            continue
        if text:
            return text
    return ""


def _attach_router_metadata(
    result: dict,
    pdf_path: str,
    page_number: int,
    page_count: int,
    extraction_method: str,
    page_text: str = "",
) -> dict:
    """Attach routing provenance fields to a single-page result dict."""
    context = result.get("_router_context")
    if not isinstance(context, dict):
        # Scanned-page results don't carry a router context yet — build one now
        # from the available fields and text so grouping can still work.
        fields = result.get("fields") or {}
        profile = native_parser.page_profile(page_text, fields.get("line_items") or [])
        candidate_number = fields.get("invoice_number")
        if candidate_number and not profile.get("invoice_number"):
            profile["invoice_number"] = str(candidate_number)
            profile["starts_invoice"] = True
            profile["role"] = "invoice_start"
        context = {
            "text": page_text,
            "profile": profile,
            "totals": {
                "has_terminal_total": isinstance(fields.get("total_amount"), (int, float)),
            },
        }
        result["_router_context"] = context

    result["source_file"] = str(pdf_path)
    result["page_number"] = page_number + 1
    result["page_count"] = page_count
    result["multi_page_handling"] = MULTI_PAGE_HANDLING
    result["extraction_method"] = extraction_method
    result["page_role"] = context.get("profile", {}).get(
        "role",
        "unclassified_continuation",
    )
    return result
