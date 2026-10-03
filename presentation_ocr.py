"""Minimal presentation wrapper for the tested invoice OCR package.

The extraction and validation pipelines still run internally, but the only
user-facing JSON keys are ``fields`` and ``missing_required``. The saved file
is always named ``extracted_invoice.json`` unless ``--output`` is supplied.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path


def _load_sibling_module(filename: str, module_name: str):
    module_path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load required module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


router = _load_sibling_module("pdf_router.py", "invoice_pdf_router")
ocr = router.ocr

SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DEFAULT_OUTPUT_FILE = "extracted_invoice.json"


def presentation_result(full_result: dict) -> dict:
    """Return the exact two-key JSON contract requested for presentation."""
    fields = full_result.get("fields")
    if not isinstance(fields, dict):
        raise ValueError("Invoice extraction did not return a valid fields object.")
    return {
        "fields": copy.deepcopy(fields),
        "missing_required": list(full_result.get("missing_required") or []),
    }


def _validate_input(input_path: str) -> Path:
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Invoice file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported invoice format. Supported formats: {supported}")
    return path


def _extract_full_results(
    input_path: str,
    pipeline_manager: router.OCRPipelineManager,
) -> list[dict]:
    path = _validate_input(input_path)

    # All detailed timing/validation output remains internal. Redirecting the
    # underlying status prints keeps the presentation console focused on the
    # same minimal JSON that is written to disk.
    quiet_console = io.StringIO()
    with tempfile.TemporaryDirectory(prefix="presentation_ocr_") as temporary_dir:
        if path.suffix.lower() == ".pdf":
            with redirect_stdout(quiet_console):
                return router.process_pdf(
                    str(path),
                    output_dir=temporary_dir,
                    export_xml=False,
                    pipeline_manager=pipeline_manager,
                )

        with redirect_stdout(quiet_console):
            pipeline = pipeline_manager.get()
            try:
                result = ocr.extract_invoice(
                    str(path),
                    output_dir=temporary_dir,
                    export_xml=False,
                    pipeline=pipeline,
                    shared_model_load_s=pipeline_manager.load_seconds,
                )
            finally:
                ocr._release_unused_gpu_cache()
        return [result]


def extract_presentation_invoice(
    input_path: str,
    *,
    pipeline_manager: router.OCRPipelineManager | None = None,
) -> dict | list[dict]:
    """Extract one image/PDF and return only fields plus missing_required.

    One logical invoice returns one object even when it spans multiple pages.
    A batch PDF containing multiple invoices returns an ordered list with one
    two-key object per invoice.
    """
    manager = pipeline_manager or router.OCRPipelineManager()
    full_results = _extract_full_results(input_path, manager)
    if not full_results:
        raise ValueError("The invoice produced no page results.")

    simplified = [presentation_result(result) for result in full_results]
    return simplified[0] if len(simplified) == 1 else simplified


def run_presentation(
    input_path: str,
    *,
    output_path: str = DEFAULT_OUTPUT_FILE,
    pipeline_manager: router.OCRPipelineManager | None = None,
) -> dict | list[dict]:
    """Extract, save and print the minimal presentation JSON."""
    payload = extract_presentation_invoice(
        input_path,
        pipeline_manager=pipeline_manager,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered_json = json.dumps(payload, indent=2, ensure_ascii=False)
    destination.write_text(rendered_json, encoding="utf-8")
    print(rendered_json)
    print(f"\nSaved: {destination}")
    return payload


def _run_colab():
    from google.colab import files as colab_files

    print("Upload one invoice image or PDF:")
    uploaded = colab_files.upload()
    inputs = [
        name for name in uploaded
        if Path(name).suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if len(inputs) != 1:
        raise SystemExit("Please upload exactly one supported invoice file.")

    run_presentation(inputs[0], output_path=DEFAULT_OUTPUT_FILE)
    colab_files.download(DEFAULT_OUTPUT_FILE)


def _run_local():
    parser = argparse.ArgumentParser(
        description="Extract an invoice into presentation-friendly minimal JSON."
    )
    parser.add_argument("invoice", help="Invoice image or PDF path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE)
    arguments = parser.parse_args()
    run_presentation(arguments.invoice, output_path=arguments.output)


if __name__ == "__main__":
    try:
        import google.colab  # noqa: F401
    except ImportError:
        _run_local()
    else:
        _run_colab()
