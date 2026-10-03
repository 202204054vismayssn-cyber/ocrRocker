import copy
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "presentation_ocr.py"
SPEC = importlib.util.spec_from_file_location("presentation_ocr", MODULE_PATH)
presentation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(presentation)


def detailed_result(total=3391.32):
    return {
        "source_file": "invoice.jpg",
        "fields": {
            "customer_name": "Bajaj General Insurance Limited",
            "invoice_number": "13BAG05260000001",
            "invoice_date": "2026-05-31",
            "taxable_value": 2874.0,
            "total_amount": total,
            "line_items": [],
        },
        "field_sources": {"customer_name": "label_value"},
        "needs_review": True,
        "missing_required": [],
        "unparsed_or_low_confidence": [],
        "validation_issues": [{
            "code": "total_arithmetic_mismatch",
            "field": "total_amount",
        }],
        "timing": {
            "model_load_s": 75.44,
            "inference_s": 33.44,
            "total_s": 33.88,
        },
        "extraction_method": "scanned_ocr",
    }


class PresentationOCRTests(unittest.TestCase):
    def test_presentation_result_has_exactly_two_keys(self):
        simplified = presentation.presentation_result(detailed_result())

        self.assertEqual(list(simplified), ["fields", "missing_required"])
        self.assertEqual(simplified["missing_required"], [])
        self.assertNotIn("needs_review", simplified)
        self.assertNotIn("validation_issues", simplified)
        self.assertNotIn("timing", simplified)
        self.assertNotIn("field_sources", simplified)

    def test_image_run_writes_only_minimal_json(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            image_path = root / "invoice.jpg"
            image_path.write_bytes(b"placeholder")
            output_path = root / "extracted_invoice.json"
            manager = presentation.router.OCRPipelineManager(pipeline=object())

            with (
                mock.patch.object(
                    presentation.ocr,
                    "extract_invoice",
                    return_value=copy.deepcopy(detailed_result()),
                ),
                mock.patch.object(presentation.ocr, "_release_unused_gpu_cache"),
                redirect_stdout(io.StringIO()),
            ):
                payload = presentation.run_presentation(
                    str(image_path),
                    output_path=str(output_path),
                    pipeline_manager=manager,
                )

            saved = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(saved, payload)
        self.assertEqual(set(saved), {"fields", "missing_required"})
        self.assertEqual(saved["fields"]["total_amount"], 3391.32)

    def test_batch_pdf_returns_two_key_object_per_logical_invoice(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            pdf_path = Path(temporary_dir) / "invoice.pdf"
            pdf_path.write_bytes(b"placeholder")
            page_results = [detailed_result(100.0), detailed_result(200.0)]

            with mock.patch.object(
                presentation.router,
                "process_pdf",
                return_value=copy.deepcopy(page_results),
            ):
                payload = presentation.extract_presentation_invoice(str(pdf_path))

        self.assertIsInstance(payload, list)
        self.assertEqual(len(payload), 2)
        self.assertTrue(all(set(page) == {"fields", "missing_required"} for page in payload))
        self.assertEqual(
            [page["fields"]["total_amount"] for page in payload],
            [100.0, 200.0],
        )


if __name__ == "__main__":
    unittest.main()
