"""FastAPI REST API for the invoice OCR pipeline.

Single responsibility: accept file uploads, run them through the existing
extraction pipeline, and return structured JSON.  All heavy logic lives in
``pdf_router.py`` and ``invoice_extraction_colab.py`` — this module is thin
HTTP glue.

Start with:
    uvicorn api:app --host 0.0.0.0 --port 8000

Environment variables:
    INVOICE_QWEN_MODEL   Ollama model name (default: qwen2.5vl:3b)
    OLLAMA_HOST          Ollama base URL  (default: http://127.0.0.1:11434)
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class InvoiceSummary(BaseModel):
    total_invoices: int = Field(..., description="Total logical invoices extracted")
    needs_review: int = Field(..., description="Number of invoices needing manual review")
    extraction_methods: Dict[str, int] = Field(..., description="Counts of extraction methods used")

class ExtractResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(..., description="List of extracted invoices")
    summary: InvoiceSummary = Field(..., description="Aggregated statistics")
    errors: Optional[List[Dict[str, Any]]] = Field(None, description="Any file-level errors")

# The router package and sibling modules are loaded through pdf_router's
# sys.path guard, so importing pdf_router is enough to bring everything in.
import pdf_router as _router

# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------

_VERSION = "1.0.0"
_DESCRIPTION = (
    "Invoice OCR extraction API.  Upload one or more PDF/image invoices "
    "and receive structured field data."
)

app = FastAPI(
    title="Invoice OCR API",
    description=_DESCRIPTION,
    version=_VERSION,
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,
        "displayRequestDuration": True,
        "filter": True,
        "syntaxHighlight.theme": "monokai"
    }
)

# Add CORS middleware to allow requests from browsers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Shared pipeline — loaded lazily on first request, reused for all subsequent
# ---------------------------------------------------------------------------

_pipeline_manager = _router.OCRPipelineManager()

# ---------------------------------------------------------------------------
# Supported file types (mirrors presentation_ocr.py)
# ---------------------------------------------------------------------------

_SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Helper: build semantic-fallback when enable_qwen=True
# ---------------------------------------------------------------------------

def _make_fallback(enable_qwen: bool):
    """Return a QwenSemanticFallback instance, or None when Qwen is disabled."""
    if not enable_qwen:
        return None
    base_url = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    if not re.match(r"^https?://", base_url, re.I):
        base_url = "http://" + base_url
    config = _router.qwen_fallback.QwenFallbackConfig(
        model=os.environ.get("INVOICE_QWEN_MODEL", "qwen2.5vl:3b"),
        base_url=base_url,
        timeout_seconds=int(os.environ.get("OLLAMA_TIMEOUT", "180")),
    )
    return _router.qwen_fallback.QwenSemanticFallback(
        _router.ocr.CANONICAL_SCHEMA,
        config=config,
    )


# ---------------------------------------------------------------------------
# Helper: extract one file, returning a list of invoice dicts
# ---------------------------------------------------------------------------

def _extract_file(
    file_path: Path,
    work_dir: Path,
    semantic_fallback,
) -> list[dict]:
    """Route one uploaded file through the appropriate extraction path."""
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        return _router.process_pdf(
            str(file_path),
            output_dir=str(work_dir / "pdf_output"),
            export_xml=False,
            pipeline_manager=_pipeline_manager,
            semantic_fallback=semantic_fallback,
        )

    if ext in _SUPPORTED_EXTENSIONS:
        # Single-image invoices go through extract_invoice directly.
        result = _router.ocr.extract_invoice(
            str(file_path),
            output_dir=str(work_dir / "img_output"),
            export_xml=False,
            pipeline=_pipeline_manager.get(),
            shared_model_load_s=_pipeline_manager.load_seconds,
        )
        try:
            _router.ocr._release_unused_gpu_cache()
        except Exception:
            pass
        if semantic_fallback is not None:
            context = result.get("_semantic_context") or {}
            result = semantic_fallback.apply(
                result,
                raw_text=context.get("raw_text") or "",
                image_paths=[str(file_path)],
            )
        return [result]

    raise ValueError(f"Unsupported file type: {ext!r}")


# ---------------------------------------------------------------------------
# Helper: build the summary block from a list of invoice results
# ---------------------------------------------------------------------------

def _build_summary(results: list[dict]) -> dict:
    """Aggregate extraction statistics across all logical invoices."""
    needs_review_count = sum(1 for r in results if r.get("needs_review"))
    method_counts: dict[str, int] = {}
    for result in results:
        method = result.get("extraction_method", "unknown")
        method_counts[method] = method_counts.get(method, 0) + 1
    return {
        "total_invoices": len(results),
        "needs_review": needs_review_count,
        "extraction_methods": method_counts,
    }


# ---------------------------------------------------------------------------
# Helper: clean a single result for the API response
# ---------------------------------------------------------------------------

def _public_result(result: dict) -> dict:
    """Strip private routing keys and add a top-level needs_review flag."""
    import copy
    public = copy.deepcopy(result)
    public.pop("_router_context", None)
    public.pop("_semantic_context", None)
    # Ensure needs_review is always present and prominent for API callers.
    public["needs_review"] = bool(result.get("needs_review"))
    public["missing_required"] = list(result.get("missing_required") or [])
    return public


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", summary="API info")
def root():
    """Return API version and available endpoints."""
    return {
        "name": "Invoice OCR API",
        "version": _VERSION,
        "endpoints": {
            "POST /extract": "Upload invoice file(s) and extract structured data",
            "GET  /upload":  "Web interface for testing file uploads",
            "GET  /health":  "Health check — returns {status: ok}",
            "GET  /":        "This endpoint",
        },
    }


@app.get("/upload", summary="Upload test page")
def upload_page():
    """Serve the HTML upload test page."""
    return FileResponse("test_upload.html", media_type="text/html")


@app.get("/health", summary="Health check")
def health():
    """Return 200 OK when the API is running."""
    return {"status": "ok"}


@app.post("/extract", summary="Extract invoice fields", response_model=ExtractResponse)
async def extract(
    files: List[UploadFile] = File(..., description="One or more PDF/image/ZIP files"),
    enable_qwen: bool = Form(False, description="Enable Qwen semantic fallback via local Ollama"),
):
    """Upload invoice files and receive structured extraction results.

    Accepts PDF, JPEG, PNG, TIFF, and ZIP (containing any of the above).
    Multi-page PDFs are grouped into logical invoices automatically.

    Returns a JSON object with:
    - ``results``: list of extracted invoices, each with ``fields``,
      ``needs_review``, ``missing_required``, ``extraction_method``, etc.
    - ``summary``: aggregated statistics over all invoices in the upload.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files uploaded.",
        )

    semantic_fallback = _make_fallback(enable_qwen)
    all_results: list[dict] = []
    errors: list[dict] = []

    # Each request gets a private temp directory — cleaned up unconditionally.
    with tempfile.TemporaryDirectory(prefix="invoice_api_") as tmp:
        work_dir = Path(tmp)

        for upload in files:
            original_name = Path(upload.filename or "upload").name
            ext = Path(original_name).suffix.lower()

            # Write the upload to disk.
            dest = work_dir / original_name
            with dest.open("wb") as fh:
                shutil.copyfileobj(upload.file, fh)

            # Expand ZIPs; reject anything unsupported.
            if ext == ".zip":
                pdf_paths = _router.collect_pdf_inputs(
                    [str(dest)],
                    zip_extract_dir=str(work_dir / "zip_extract"),
                )
                if not pdf_paths:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"No PDF files found inside ZIP: {original_name}",
                    )
                input_paths = [Path(p) for p in pdf_paths]
            elif ext in _SUPPORTED_EXTENSIONS:
                input_paths = [dest]
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Unsupported file type: {ext!r}. "
                        f"Accepted: {', '.join(sorted(_SUPPORTED_EXTENSIONS))} and .zip"
                    ),
                )

            # Extract each file.
            for file_path in input_paths:
                try:
                    results = _extract_file(file_path, work_dir, semantic_fallback)
                    all_results.extend(results)
                except Exception as exc:
                    errors.append({
                        "file": file_path.name,
                        "error": str(exc),
                    })

    if not all_results and errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "All files failed extraction.", "errors": errors},
        )

    public = [_public_result(r) for r in all_results]
    response: dict = {
        "results": public,
        "summary": _build_summary(all_results),
    }
    if errors:
        response["errors"] = errors

    return JSONResponse(content=response)
