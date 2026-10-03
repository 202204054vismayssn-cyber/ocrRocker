import copy
import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "pdf_router.py"
SPEC = importlib.util.spec_from_file_location("pdf_router", MODULE_PATH)
router = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(router)


NATIVE_METADATA_TABLE = [
    ["GSTIN Of Supplier", "13AAACC6106G2Z9"],
    ["Invoice No.", "13BAG05260000001"],
    ["Date of Invoice", "31-05-2026"],
    ["Place of Supply (State Code)", "MAHARASHTRA-(27)"],
    ["Name", "Bajaj General Insurance Limited"],
    ["Customer GSTIN", "27AABCB5730G1ZX"],
    ["Amount In Words", "three thousand three hundred ninety one rupees and thirty two paisa Only"],
]

NATIVE_ITEM_TABLE = [
    [
        "Sl No", "Goods/Service Description", "HSN/SAC", "Discount",
        "Taxable Value", "CGST Amt", "SGST Amt", "IGST Amt", "Total Value",
    ],
    ["1", "COMM - BAGIC GL", "997161", "0.00", "2874.00", "0.00", "0.00", "517.32", "3391.32"],
]


class FakePlumberPage:
    def __init__(self, text, tables):
        self._text = text
        self._tables = tables

    def extract_text(self):
        return self._text

    def extract_tables(self):
        return self._tables


class FakePlumberPDF:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakeFitzPage:
    def __init__(self, text):
        self._text = text

    def get_text(self):
        return self._text


class FakeFitzDocument:
    def __init__(self, page_texts):
        self.pages = [FakeFitzPage(text) for text in page_texts]

    def __iter__(self):
        return iter(self.pages)

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, index):
        return self.pages[index]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def complete_result():
    return {
        "source_file": "temporary-page.png",
        "fields": {
            "customer_name": "Bajaj General Insurance Limited",
            "invoice_number": "TEST-001",
            "invoice_date": "2026-05-31",
            "total_amount": 118.0,
            "line_items": [],
        },
        "field_sources": {},
        "needs_review": False,
        "missing_required": [],
        "unparsed_or_low_confidence": [],
        "validation_issues": [],
    }


class PDFRouterTests(unittest.TestCase):
    def test_semantic_fallback_runs_after_native_canonical_merge(self):
        incomplete = complete_result()
        incomplete["fields"]["customer_name"] = None
        incomplete["missing_required"] = ["customer_name"]
        incomplete["needs_review"] = True

        class CapturingFallback:
            def __init__(self):
                self.calls = []

            def apply(self, result, *, raw_text, image_paths):
                self.calls.append((raw_text, list(image_paths)))
                result["fields"]["customer_name"] = "Recovered Buyer"
                result["missing_required"] = []
                return result

        fallback = CapturingFallback()
        with tempfile.TemporaryDirectory() as temporary_dir:
            pdf_path = Path(temporary_dir) / "native.pdf"
            pdf_path.write_bytes(b"placeholder")
            native_text = "Tax Invoice Buyer New Customer Invoice No TEST-001"
            with (
                mock.patch.object(router.fitz, "open", return_value=FakeFitzDocument([native_text])),
                mock.patch.object(router, "extract_native_page", return_value=incomplete),
            ):
                results = router.process_pdf(
                    str(pdf_path),
                    output_dir=str(Path(temporary_dir) / "output"),
                    native_missing_threshold=10,
                    semantic_fallback=fallback,
                )

        self.assertEqual(results[0]["fields"]["customer_name"], "Recovered Buyer")
        self.assertEqual(fallback.calls, [(native_text, [])])

    def test_page_is_native_checks_text_length_and_printable_ratio(self):
        self.assertTrue(router.page_is_native(FakeFitzPage("Readable invoice text with enough characters")))
        self.assertFalse(router.page_is_native(FakeFitzPage("short")))
        self.assertFalse(router.page_is_native(FakeFitzPage("\x00" * 30)))

    def test_native_tables_reuse_existing_classifier_and_schema(self):
        plumber_page = FakePlumberPage(
            "Canara Bank\nTax Invoice\nBajaj General Insurance Limited",
            [NATIVE_METADATA_TABLE, NATIVE_ITEM_TABLE],
        )
        fake_pdfplumber = SimpleNamespace(
            open=lambda _path: FakePlumberPDF([plumber_page])
        )

        with mock.patch.object(router, "pdfplumber", fake_pdfplumber):
            result = router.extract_native_page("CBSInvoice-13BAG05260000001.pdf", 0)

        fields = result["fields"]
        self.assertEqual(fields["vendor_name"], "Canara Bank")
        self.assertEqual(fields["invoice_number"], "13BAG05260000001")
        self.assertEqual(fields["customer_name"], "Bajaj General Insurance Limited")
        self.assertEqual(fields["taxable_value"], 2874.0)
        self.assertEqual(fields["igst_amount"], 517.32)
        self.assertEqual(fields["total_amount"], 3391.32)
        self.assertEqual(len(fields["line_items"]), 1)

    def test_native_page_without_tables_uses_existing_text_parser(self):
        page_text = """Canara Bank
GSTIN Of Supplier: 13AAACC6106G2Z9
Invoice No.: 13BAG05260000001
Date of Invoice: 31-05-2026
Name: Bajaj General Insurance Limited
Customer GSTIN: 27AABCB5730G1ZX
Grand Total: 3391.32
"""
        fake_pdfplumber = SimpleNamespace(
            open=lambda _path: FakePlumberPDF([FakePlumberPage(page_text, [])])
        )

        with mock.patch.object(router, "pdfplumber", fake_pdfplumber):
            result = router.extract_native_page("native.pdf", 0)

        self.assertEqual(result["fields"]["invoice_number"], "13BAG05260000001")
        self.assertEqual(result["fields"]["total_amount"], 3391.32)

    def test_mixed_pdf_groups_pages_and_reuses_preloaded_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            pdf_path = Path(temporary_dir) / "mixed.pdf"
            pdf_path.write_bytes(b"test placeholder")
            output_dir = Path(temporary_dir) / "output"
            manager = router.OCRPipelineManager(pipeline=object())

            fake_document = FakeFitzDocument([
                "Native invoice page with plenty of readable characters",
                "",
            ])
            with (
                mock.patch.object(router.fitz, "open", return_value=fake_document),
                mock.patch.object(router, "extract_native_page", return_value=complete_result()),
                mock.patch.object(router, "render_page_to_image", return_value="page.png"),
                mock.patch.object(
                    router.ocr,
                    "extract_invoice",
                    side_effect=lambda *_args, **_kwargs: copy.deepcopy(complete_result()),
                ) as extract_mock,
                mock.patch.object(router.ocr, "_release_unused_gpu_cache"),
            ):
                results = router.process_pdf(
                    str(pdf_path),
                    output_dir=str(output_dir),
                    pipeline_manager=manager,
                )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["extraction_method"], "mixed_native_and_ocr")
        self.assertEqual(results[0]["page_extraction_methods"], ["native_pdf", "scanned_ocr"])
        self.assertEqual(results[0]["page_numbers"], [1, 2])
        self.assertEqual(results[0]["page_count"], 2)
        self.assertEqual(results[0]["multi_page_handling"], "grouped_by_invoice")
        self.assertEqual(extract_mock.call_count, 1)

    def test_incomplete_native_result_falls_back_to_ocr(self):
        incomplete = complete_result()
        incomplete["missing_required"] = ["customer_name", "invoice_date", "total_amount"]
        incomplete["needs_review"] = True

        with tempfile.TemporaryDirectory() as temporary_dir:
            pdf_path = Path(temporary_dir) / "fallback.pdf"
            pdf_path.write_bytes(b"test placeholder")
            manager = router.OCRPipelineManager(pipeline=object())

            with (
                mock.patch.object(
                    router.fitz,
                    "open",
                    return_value=FakeFitzDocument([
                        "Native-looking invoice text with plenty of readable characters"
                    ]),
                ),
                mock.patch.object(router, "extract_native_page", return_value=incomplete),
                mock.patch.object(router, "render_page_to_image", return_value="fallback.png"),
                mock.patch.object(
                    router.ocr,
                    "extract_invoice",
                    return_value=copy.deepcopy(complete_result()),
                ),
                mock.patch.object(router.ocr, "_release_unused_gpu_cache"),
            ):
                results = router.process_pdf(
                    str(pdf_path),
                    output_dir=str(Path(temporary_dir) / "output"),
                    pipeline_manager=manager,
                )

        self.assertEqual(results[0]["extraction_method"], "native_pdf_fallback_to_ocr")
        self.assertEqual(results[0]["missing_required"], [])

    def test_missing_fields_on_continuation_page_do_not_trigger_ocr(self):
        start = complete_result()
        start["fields"]["invoice_number"] = "INV-001"
        start["_router_context"] = {
            "text": "Tax Invoice Bill To Invoice Date INV-001",
            "profile": {
                "invoice_number": "INV-001",
                "starts_invoice": True,
                "role": "invoice_start",
            },
            "totals": {"has_terminal_total": False},
        }
        continuation = complete_result()
        continuation["fields"] = {"line_items": []}
        continuation["missing_required"] = [
            "customer_name", "invoice_number", "invoice_date", "total_amount"
        ]
        continuation["needs_review"] = True
        continuation["_router_context"] = {
            "text": "Item & Description HSN Qty Rate Amount",
            "profile": {
                "invoice_number": None,
                "starts_invoice": False,
                "role": "table_continuation",
            },
            "totals": {"has_terminal_total": False},
        }

        with tempfile.TemporaryDirectory() as temporary_dir:
            pdf_path = Path(temporary_dir) / "continuation.pdf"
            pdf_path.write_bytes(b"test placeholder")
            with (
                mock.patch.object(
                    router.fitz,
                    "open",
                    return_value=FakeFitzDocument([
                        "Tax Invoice Bill To Invoice Date readable native first page",
                        "Item Description continuation page with readable native text",
                    ]),
                ),
                mock.patch.object(
                    router,
                    "extract_native_page",
                    side_effect=[start, continuation],
                ),
                mock.patch.object(router, "render_page_to_image") as render_mock,
            ):
                results = router.process_pdf(
                    str(pdf_path),
                    output_dir=str(Path(temporary_dir) / "output"),
                )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["page_numbers"], [1, 2])
        self.assertEqual(
            results[0]["page_extraction_methods"],
            ["native_pdf", "native_pdf"],
        )
        render_mock.assert_not_called()

    def test_different_invoice_numbers_start_separate_groups(self):
        first = router._attach_router_metadata(
            complete_result(), "batch.pdf", 0, 2, "native_pdf"
        )
        second_result = complete_result()
        second_result["fields"]["invoice_number"] = "TEST-002"
        second = router._attach_router_metadata(
            second_result, "batch.pdf", 1, 2, "native_pdf"
        )

        groups = router._group_page_results([first, second])

        self.assertEqual(len(groups), 2)
        self.assertEqual([len(group) for group in groups], [1, 1])

    def test_zip_batch_extracts_only_pdfs_without_path_traversal(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            archive_path = root / "batch.zip"
            extraction_root = root / "extracted"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("folder/one.pdf", b"pdf-one")
                archive.writestr("../../two.pdf", b"pdf-two")
                archive.writestr("ignore.txt", b"not a pdf")

            paths = router.collect_pdf_inputs(
                [str(archive_path)],
                zip_extract_dir=str(extraction_root),
            )

            self.assertEqual(len(paths), 2)
            self.assertTrue(all(Path(path).is_relative_to(extraction_root) for path in paths))
            self.assertFalse((root / "two.pdf").exists())

    def test_render_rejects_dpi_below_accuracy_floor(self):
        with self.assertRaisesRegex(ValueError, "at least 300 DPI"):
            router.render_page_to_image("invoice.pdf", 0, dpi=299)


if __name__ == "__main__":
    unittest.main()
