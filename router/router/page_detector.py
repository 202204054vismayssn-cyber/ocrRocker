"""Page-level detection: decides native vs scanned, renders pages to PNG, manages the OCR pipeline.

A *native* PDF page has a sufficient embedded text layer that can be read
directly with pdfplumber — no OCR needed.  A *scanned* page (or one whose
text layer is too sparse) must be rendered to an image and passed through
PaddleOCR.
"""

from __future__ import annotations

import re
import tempfile
import time
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover - compatibility with older PyMuPDF
    try:
        import fitz  # type: ignore[no-redef]
    except ImportError as exc:  # pragma: no cover - depends on local environment
        fitz = None
        _FITZ_IMPORT_ERROR = exc
    else:
        _FITZ_IMPORT_ERROR = None
else:
    _FITZ_IMPORT_ERROR = None

try:
    import pdfplumber
except ImportError as exc:  # pragma: no cover - depends on local environment
    pdfplumber = None
    _PDFPLUMBER_IMPORT_ERROR = exc
else:
    _PDFPLUMBER_IMPORT_ERROR = None


MIN_OCR_DPI = 300
DEFAULT_NATIVE_MIN_CHARS = 20


class PDFRouterDependencyError(ImportError):
    """Raised when one of the router's two lightweight PDF packages is absent."""


def _require_fitz():
    """Raise PDFRouterDependencyError when PyMuPDF is not installed."""
    if fitz is None:
        raise PDFRouterDependencyError(
            "PyMuPDF is required for PDF detection/rendering. "
            "Install it with: python -m pip install PyMuPDF"
        ) from _FITZ_IMPORT_ERROR


def _require_pdfplumber():
    """Raise PDFRouterDependencyError when pdfplumber is not installed."""
    if pdfplumber is None:
        raise PDFRouterDependencyError(
            "pdfplumber is required for native-PDF extraction. "
            "Install it with: python -m pip install pdfplumber"
        ) from _PDFPLUMBER_IMPORT_ERROR


def page_is_native(page, min_chars: int = DEFAULT_NATIVE_MIN_CHARS) -> bool:
    """Return True only when a page has a sufficiently usable embedded text layer."""
    if min_chars < 1:
        raise ValueError("min_chars must be at least 1")

    text = str(page.get_text() or "").strip()
    if len(text) < min_chars:
        return False

    # Reject pages where most characters are non-printable (e.g. binary blobs).
    printable_ratio = sum(character.isprintable() for character in text) / len(text)
    return printable_ratio > 0.85


def render_page_to_image(
    pdf_path: str,
    page_number: int,
    dpi: int = MIN_OCR_DPI,
    output_dir: str | None = None,
) -> str:
    """Render one zero-based PDF page to a PNG at a safe OCR resolution."""
    _require_fitz()
    if dpi < MIN_OCR_DPI:
        raise ValueError(f"OCR rendering must use at least {MIN_OCR_DPI} DPI.")

    destination = Path(output_dir or tempfile.gettempdir())
    destination.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(pdf_path).stem).strip("._") or "invoice"
    output_path = destination / f"{stem}_p{page_number + 1:04d}.png"

    with fitz.open(str(pdf_path)) as document:
        if not 0 <= page_number < len(document):
            raise IndexError(
                f"Page index {page_number} is outside PDF page range 0..{len(document) - 1}."
            )
        page = document[page_number]
        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
        pixmap.save(str(output_path))

    return str(output_path)


class OCRPipelineManager:
    """Load PaddleOCR lazily and reuse one model instance across all routed pages.

    Pass a pre-built pipeline in tests to skip the real model load entirely.
    """

    def __init__(self, pipeline=None):
        self.pipeline = pipeline
        self.load_seconds = 0.0

    def get(self):
        """Return the shared pipeline, loading it on first call if needed."""
        if self.pipeline is None:
            from router._modules import ocr
            started = time.perf_counter()
            self.pipeline = ocr.create_ocr_pipeline()
            self.load_seconds = round(time.perf_counter() - started, 2)
        return self.pipeline
