"""Per-page PDF router with logical multi-page invoice grouping.

Native PDF pages use pdfplumber. Scanned pages are rendered at 300 DPI or
higher and passed to the existing PP-OCRv6 ``extract_invoice`` function.
Both paths reuse the tested parser/mapping logic in
``invoice_extraction_colab.py`` and converge on one canonical result/invoice.

The lower-level building blocks live in the ``router/`` package:
``page_detector`` (native vs scanned detection, rendering, pipeline),
``page_extractor`` (per-page extraction), and ``invoice_grouper``
(grouping and merging pages into logical invoices).  The batch orchestration
stays here because this module is the one callers and tests patch against.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import tempfile
import time
import sys
import zipfile
from pathlib import Path

# The ``router`` package is a sibling of this file.  This module is normally
# imported by name (so the project root is already on sys.path), but it is also
# loaded by path via importlib in tests and in ``presentation_ocr.py``, where
# sys.path may not contain the project root yet.  Add it defensively so the
# imports below always resolve.
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Imports from the router package ──────────────────────────────────────────

from router._modules import ocr, native_parser, qwen_fallback          # noqa: F401
from router.page_detector import (                                       # noqa: F401
    fitz,
    pdfplumber,
    PDFRouterDependencyError,
    MIN_OCR_DPI,
    DEFAULT_NATIVE_MIN_CHARS,
    OCRPipelineManager,
    page_is_native,
    render_page_to_image,
    _require_fitz,
    _require_pdfplumber,
)
from router.page_extractor import (                                      # noqa: F401
    _run_scanned_page,
    _read_ocr_text,
    _attach_router_metadata,
)
from router.invoice_grouper import (                                     # noqa: F401
    _is_present,
    _normal_identifier,
    _group_page_results,
    _item_key,
    _usable_item,
    _sum_item_values,
    _aggregate_timing,
    _merge_invoice_group,
)
from router.batch import (                                               # noqa: F401
    DEFAULT_NATIVE_MISSING_THRESHOLD,
    MULTI_PAGE_HANDLING,
)

# ── Native page extraction ───────────────────────────────────────────────────

def extract_native_page(pdf_path: str, page_number: int) -> dict:
    """Extract one zero-based native page through the existing parser and mapper.

    This function is defined here rather than in ``router.page_extractor`` so
    that it resolves ``pdfplumber`` from this module's globals — which is what
    callers and tests patch.
    """
    _require_pdfplumber()
    start = time.perf_counter()

    with pdfplumber.open(str(pdf_path)) as pdf:
        if not 0 <= page_number < len(pdf.pages):
            raise IndexError(
                f"Page index {page_number} is outside PDF page range 0..{len(pdf.pages) - 1}."
            )
        page = pdf.pages[page_number]
        page_text = page.extract_text() or ""
        tables = page.extract_tables() or []

    if tables:
        kv_pairs = []
        line_items = []
        other_tables = []
        for table_rows in tables:
            if table_rows:
                # Reuse the existing classifier as-is — it understands the
                # Canara Bank table layout and must not be duplicated here.
                ocr._classify_table(table_rows, kv_pairs, line_items, other_tables)
        parsed = {
            "kv_pairs": kv_pairs,
            "line_items": line_items,
            "other_tables": other_tables,
            "raw_text": page_text,
        }
    else:
        # Rare native invoices contain positioned text but no detectable table.
        # The existing text parser already handles Label: Value lines safely.
        parsed = ocr.parse_markdown_output(page_text)

    result = ocr.build_form_ready_json(parsed, source_file=str(pdf_path))
    result, router_context = native_parser.enhance_native_result(
        result,
        page_text,
        tables,
    )
    # Private routing evidence is removed before JSON/XML is written.  Keeping
    # it beside the page result makes logical grouping deterministic without
    # exposing the entire PDF text layer in the public result.
    result["_router_context"] = router_context
    elapsed = time.perf_counter() - start
    result["timing"] = {
        "model_load_s": 0.0,
        "inference_s": 0.0,
        "save_output_s": 0.0,
        "parse_and_map_s": round(elapsed, 2),
        "total_s": round(elapsed, 2),
    }
    return result


# ── Batch orchestration ──────────────────────────────────────────────────────

def _safe_output_component(value: str) -> str:
    """Sanitize a string for use as a file or directory name component."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "invoice"


def _public_result(result: dict) -> dict:
    """Return a deep copy of a result with all private routing keys removed."""
    public = copy.deepcopy(result)
    public.pop("_router_context", None)
    public.pop("_semantic_context", None)
    return public


def _write_invoice_result(result: dict, invoice_output_dir: Path, export_xml: bool):
    """Write the canonical JSON (and optionally XML) for one invoice to disk."""
    invoice_output_dir.mkdir(parents=True, exist_ok=True)
    (invoice_output_dir / "invoice_extracted.json").write_text(
        json.dumps(_public_result(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if export_xml and result.get("fields") is not None:
        (invoice_output_dir / "invoice_extracted.xml").write_text(
            ocr.to_xml(result),
            encoding="utf-8",
        )


def process_pdf(
    pdf_path: str,
    *,
    output_dir: str = "pdf_router_output",
    export_xml: bool = False,
    min_chars: int = DEFAULT_NATIVE_MIN_CHARS,
    native_missing_threshold: int = DEFAULT_NATIVE_MISSING_THRESHOLD,
    dpi: int = MIN_OCR_DPI,
    pipeline_manager: OCRPipelineManager | None = None,
    semantic_fallback=None,
) -> list[dict]:
    """Route each page of a PDF, then return one canonical result per logical invoice."""
    _require_fitz()
    _require_pdfplumber()
    if native_missing_threshold < 0:
        raise ValueError("native_missing_threshold cannot be negative")
    if dpi < MIN_OCR_DPI:
        raise ValueError(f"dpi must be at least {MIN_OCR_DPI}")

    source_path = Path(pdf_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"PDF not found: {source_path}")
    if source_path.suffix.lower() != ".pdf":
        raise ValueError(f"PDF router accepts only .pdf files: {source_path}")

    manager = pipeline_manager or OCRPipelineManager()
    pdf_output_dir = Path(output_dir) / _safe_output_component(source_path.stem)
    page_results = []

    # Phase 1: detect native vs scanned for each page.
    with fitz.open(str(source_path)) as document:
        page_count = len(document)
        detected_native = []
        page_texts = []
        for page in document:
            try:
                page_texts.append(str(page.get_text() or ""))
                detected_native.append(page_is_native(page, min_chars=min_chars))
            except Exception:
                # A broken or unreadable text layer is safest on the OCR path.
                page_texts.append("")
                detected_native.append(False)

    # Phase 2: extract each page via the appropriate path.
    with tempfile.TemporaryDirectory(prefix="invoice_pdf_router_") as temporary_dir:
        for page_number, is_native in enumerate(detected_native):
            page_output_dir = pdf_output_dir / "page_diagnostics" / f"page_{page_number + 1:04d}"
            image_path = None

            if is_native:
                try:
                    result = extract_native_page(str(source_path), page_number)
                except Exception:
                    # Native extraction can fail even when text-layer detection
                    # succeeds. The OCR path is the accuracy fallback.
                    image_path = render_page_to_image(
                        str(source_path), page_number, dpi=dpi, output_dir=temporary_dir
                    )
                    result = _run_scanned_page(image_path, page_output_dir, manager)
                    extraction_method = "native_pdf_fallback_to_ocr"
                else:
                    missing_required = result.get("missing_required") or []
                    context = result.get("_router_context") or {}
                    profile = context.get("profile") or native_parser.page_profile(
                        page_texts[page_number],
                        (result.get("fields") or {}).get("line_items") or [],
                    )
                    # Only pages that start a new invoice are candidates for OCR
                    # fallback; continuation pages with missing fields are expected
                    # and should stay on the native path.
                    eligible_for_fallback = bool(
                        profile.get("starts_invoice") or page_number == 0
                    )
                    if (
                        len(missing_required) > native_missing_threshold
                        and eligible_for_fallback
                    ):
                        image_path = render_page_to_image(
                            str(source_path), page_number, dpi=dpi, output_dir=temporary_dir
                        )
                        result = _run_scanned_page(image_path, page_output_dir, manager)
                        extraction_method = "native_pdf_fallback_to_ocr"
                    else:
                        extraction_method = "native_pdf"
            else:
                image_path = render_page_to_image(
                    str(source_path), page_number, dpi=dpi, output_dir=temporary_dir
                )
                result = _run_scanned_page(image_path, page_output_dir, manager)
                extraction_method = "scanned_ocr"

            # For the Qwen text tier, prefer the richer PaddleOCR markdown over
            # the raw fitz text layer when the page went through OCR.
            semantic_text = page_texts[page_number]
            if image_path:
                semantic_text = _read_ocr_text(page_output_dir) or semantic_text

            result = _attach_router_metadata(
                result,
                str(source_path),
                page_number,
                page_count,
                extraction_method,
                semantic_text,
            )
            if image_path:
                result["_router_context"]["image_path"] = image_path
            page_results.append(result)

        # Phase 3: group pages into logical invoices and merge. Rendered page
        # images must remain alive through optional vision fallback, which is
        # why this phase runs inside the TemporaryDirectory context.
        invoice_results = []
        for invoice_index, group in enumerate(_group_page_results(page_results), start=1):
            merged = _merge_invoice_group(
                group,
                invoice_index=invoice_index,
                document_page_count=page_count,
            )
            if semantic_fallback is not None:
                context = merged.get("_semantic_context") or {}
                merged = semantic_fallback.apply(
                    merged,
                    raw_text=context.get("raw_text") or "",
                    image_paths=context.get("image_paths") or [],
                )
            merged.pop("_semantic_context", None)
            invoice_number = (merged.get("fields") or {}).get("invoice_number")
            suffix = _safe_output_component(str(invoice_number or "unknown"))
            invoice_output_dir = pdf_output_dir / f"invoice_{invoice_index:04d}_{suffix}"
            _write_invoice_result(merged, invoice_output_dir, export_xml=export_xml)
            invoice_results.append(merged)

    return invoice_results


def _extract_pdf_zip(zip_path: Path, destination: Path) -> list[Path]:
    """Safely extract only PDF members, flattening untrusted ZIP paths."""
    destination.mkdir(parents=True, exist_ok=True)
    extracted = []
    used_names = set()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir() or Path(member.filename).suffix.lower() != ".pdf":
                continue
            base_name = _safe_output_component(Path(member.filename).stem) + ".pdf"
            candidate_name = base_name
            duplicate_index = 2
            while candidate_name.lower() in used_names:
                candidate_name = f"{Path(base_name).stem}_{duplicate_index}.pdf"
                duplicate_index += 1
            used_names.add(candidate_name.lower())
            output_path = destination / candidate_name
            with archive.open(member) as source, output_path.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted.append(output_path)
    return extracted


def collect_pdf_inputs(
    arguments: list[str],
    *,
    zip_extract_dir: str | None = None,
) -> list[str]:
    """Collect PDFs from explicit files, folders, and uploaded ZIP batches."""
    files = []
    seen = set()
    zip_root = Path(zip_extract_dir) if zip_extract_dir else None
    for argument_index, argument in enumerate(arguments, start=1):
        path = Path(argument)
        if path.is_dir():
            candidates = sorted(path.rglob("*.pdf"))
        elif path.is_file() and path.suffix.lower() == ".zip":
            if zip_root is None:
                zip_root = Path(tempfile.mkdtemp(prefix="invoice_pdf_zip_"))
            zip_destination = zip_root / f"zip_{argument_index:04d}_{_safe_output_component(path.stem)}"
            candidates = _extract_pdf_zip(path, zip_destination)
        else:
            candidates = [path]
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix.lower() != ".pdf":
                continue
            resolved = str(candidate.resolve())
            if resolved not in seen:
                files.append(resolved)
                seen.add(resolved)
    return files


def run_pdf_batch(
    pdf_paths: list[str],
    *,
    output_dir: str = "pdf_router_output",
    export_xml: bool = True,
    min_chars: int = DEFAULT_NATIVE_MIN_CHARS,
    native_missing_threshold: int = DEFAULT_NATIVE_MISSING_THRESHOLD,
    dpi: int = MIN_OCR_DPI,
    pipeline_manager: OCRPipelineManager | None = None,
    semantic_fallback=None,
) -> list[dict]:
    """Process multiple PDFs while sharing one lazily loaded OCR pipeline."""
    manager = pipeline_manager or OCRPipelineManager()
    all_results = []

    for pdf_path in pdf_paths:
        print(f"Processing PDF: {pdf_path}")
        try:
            invoice_results = process_pdf(
                pdf_path,
                output_dir=output_dir,
                export_xml=export_xml,
                min_chars=min_chars,
                native_missing_threshold=native_missing_threshold,
                dpi=dpi,
                pipeline_manager=manager,
                semantic_fallback=semantic_fallback,
            )
            all_results.extend(invoice_results)
            method_counts = {}
            for result in invoice_results:
                method = result["extraction_method"]
                method_counts[method] = method_counts.get(method, 0) + 1
            methods = ", ".join(
                f"{method}={count}" for method, count in sorted(method_counts.items())
            )
            print(f"  Completed {len(invoice_results)} invoice(s): {methods}")
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            all_results.append({
                "source_file": str(pdf_path),
                "fields": None,
                "extraction_method": "router_error",
                "needs_review": True,
                "missing_required": [],
                "unparsed_or_low_confidence": [],
                "error": str(exc),
            })

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "pdf_router_batch_summary.json").write_text(
        json.dumps([_public_result(result) for result in all_results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return all_results


# ── CLI entry points (Colab and local) ───────────────────────────────────────

def _run_colab():
    """Upload PDFs or a ZIP via Google Colab, process, and download results."""
    from google.colab import files as colab_files

    print("Upload one or more PDF invoices, or one ZIP containing PDFs:")
    uploaded = colab_files.upload()
    with tempfile.TemporaryDirectory(prefix="invoice_uploaded_zip_") as zip_directory:
        pdf_paths = collect_pdf_inputs(
            list(uploaded.keys()),
            zip_extract_dir=zip_directory,
        )
        if not pdf_paths:
            raise SystemExit("No PDF files were found in the upload.")
        run_pdf_batch(pdf_paths, export_xml=True)
    archive_path = shutil.make_archive(
        "invoice_pdf_router_results",
        "zip",
        root_dir="pdf_router_output",
    )
    print(f"Saved result archive: {archive_path}")
    colab_files.download(archive_path)


def _run_local():
    """Parse CLI arguments and run the router locally."""
    parser = argparse.ArgumentParser(
        description="Route native/scanned PDF invoice pages to the correct extractor."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="PDF file(s), ZIP batch(es), or folders containing PDFs",
    )
    parser.add_argument("--output-dir", default="pdf_router_output")
    parser.add_argument("--dpi", type=int, default=MIN_OCR_DPI)
    parser.add_argument(
        "--native-missing-threshold",
        type=int,
        default=DEFAULT_NATIVE_MISSING_THRESHOLD,
    )
    parser.add_argument("--no-xml", action="store_true")
    parser.add_argument(
        "--enable-qwen",
        action="store_true",
        help="Use local Ollama/Qwen only when required fields remain missing",
    )
    parser.add_argument(
        "--ollama-model",
        default=os.environ.get("INVOICE_QWEN_MODEL", "qwen2.5vl:3b"),
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
    )
    parser.add_argument("--ollama-timeout", type=int, default=180)
    parser.add_argument("--no-qwen-image-fallback", action="store_true")
    arguments = parser.parse_args()

    semantic_fallback = None
    if arguments.enable_qwen:
        base_url = arguments.ollama_url
        if not re.match(r"^https?://", base_url, re.I):
            base_url = "http://" + base_url
        config = qwen_fallback.QwenFallbackConfig(
            model=arguments.ollama_model,
            base_url=base_url,
            timeout_seconds=arguments.ollama_timeout,
            image_escalation=not arguments.no_qwen_image_fallback,
        )
        semantic_fallback = qwen_fallback.QwenSemanticFallback(
            ocr.CANONICAL_SCHEMA,
            config=config,
        )

    with tempfile.TemporaryDirectory(prefix="invoice_local_zip_") as zip_directory:
        pdf_paths = collect_pdf_inputs(
            arguments.paths,
            zip_extract_dir=zip_directory,
        )
        if not pdf_paths:
            raise SystemExit("No PDF files were found.")
        run_pdf_batch(
            pdf_paths,
            output_dir=arguments.output_dir,
            export_xml=not arguments.no_xml,
            native_missing_threshold=arguments.native_missing_threshold,
            dpi=arguments.dpi,
            semantic_fallback=semantic_fallback,
        )


if __name__ == "__main__":
    try:
        import google.colab  # noqa: F401
    except ImportError:
        _run_local()
    else:
        _run_colab()
