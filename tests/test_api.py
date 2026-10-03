"""Integration test for the FastAPI invoice OCR API.

This test spins up the real FastAPI application using httpx's async test
client — no network, no real OCR model.  The OCR pipeline and router
functions are patched so the test runs in milliseconds without any GPU or
file-system dependencies.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

# ---------------------------------------------------------------------------
# Load api.py by path, patching FastAPI's dependency imports so the test
# does not require uvicorn/httpx to be installed just for discovery.
# ---------------------------------------------------------------------------

_API_PATH = Path(__file__).resolve().parents[1] / "api.py"


def _load_api():
    spec = importlib.util.spec_from_file_location("api", _API_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Minimal stub results that look like real extraction output
# ---------------------------------------------------------------------------

def _stub_result(invoice_number="TEST-001"):
    return {
        "fields": {
            "invoice_number": invoice_number,
            "invoice_date": "2026-05-31",
            "customer_name": "Test Buyer Pvt Ltd",
            "vendor_name": "Test Supplier",
            "total_amount": 3391.32,
            "line_items": [],
        },
        "extraction_method": "scanned_ocr",
        "needs_review": False,
        "missing_required": [],
        "validation_issues": [],
        "source_file": "test.jpg",
        "page_count": 1,
        "timing": {"total_s": 0.1},
    }


class APIHealthTest(unittest.TestCase):
    """Smoke-test /health and / without touching the OCR pipeline."""

    def setUp(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi not installed — skipping API integration tests")
        self.TestClient = TestClient

    def test_health_returns_ok(self):
        api = _load_api()
        client = self.TestClient(api.app)
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_root_returns_version_and_endpoints(self):
        api = _load_api()
        client = self.TestClient(api.app)
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("version", data)
        self.assertIn("endpoints", data)
        self.assertIn("POST /extract", data["endpoints"])


class APIExtractTest(unittest.TestCase):
    """Test /extract with mocked OCR to avoid real inference."""

    def setUp(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi not installed — skipping API integration tests")
        self.TestClient = TestClient

    def _client_with_stubbed_extraction(self, stub_return):
        """Return a test client where _extract_file is patched to return stub_return."""
        api = _load_api()
        # Patch at the module level inside api so all code paths see the stub.
        patcher = mock.patch.object(
            api,
            "_extract_file",
            return_value=stub_return,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return self.TestClient(api.app)

    def test_extract_image_returns_results_and_summary(self):
        stub = [_stub_result()]
        client = self._client_with_stubbed_extraction(stub)
        sample = Path(__file__).resolve().parents[1] / "samples"
        jpg = next(sample.glob("*.jpg"), None)
        if jpg is None:
            self.skipTest("No sample image found in samples/")
        with open(jpg, "rb") as f:
            response = client.post(
                "/extract",
                files=[("files", (jpg.name, f, "image/jpeg"))],
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("results", data)
        self.assertIn("summary", data)
        self.assertGreaterEqual(data["summary"]["total_invoices"], 1)
        self.assertGreaterEqual(len(data["results"]), 1)
        first = data["results"][0]
        self.assertIn("needs_review", first)
        self.assertIn("missing_required", first)
        self.assertIn("fields", first)

    def test_extract_unsupported_type_returns_400(self):
        api = _load_api()
        client = self.TestClient(api.app)
        response = client.post(
            "/extract",
            files=[("files", ("doc.docx", b"fake content", "application/octet-stream"))],
        )
        self.assertEqual(response.status_code, 400)

    def test_extract_multi_invoice_summary_counts_correctly(self):
        stubs = [
            _stub_result("INV-001"),
            {**_stub_result("INV-002"), "needs_review": True, "missing_required": ["invoice_date"]},
        ]
        client = self._client_with_stubbed_extraction(stubs)
        sample = Path(__file__).resolve().parents[1] / "samples"
        jpg = next(sample.glob("*.jpg"), None)
        if jpg is None:
            self.skipTest("No sample image found in samples/")
        with open(jpg, "rb") as f:
            response = client.post(
                "/extract",
                files=[("files", (jpg.name, f, "image/jpeg"))],
            )
        self.assertEqual(response.status_code, 200)
        summary = response.json()["summary"]
        self.assertEqual(summary["total_invoices"], 2)
        self.assertEqual(summary["needs_review"], 1)

    def test_needs_review_is_always_present_in_result(self):
        stub = [_stub_result()]
        client = self._client_with_stubbed_extraction(stub)
        sample = Path(__file__).resolve().parents[1] / "samples"
        jpg = next(sample.glob("*.jpg"), None)
        if jpg is None:
            self.skipTest("No sample image found in samples/")
        with open(jpg, "rb") as f:
            response = client.post(
                "/extract",
                files=[("files", (jpg.name, f, "image/jpeg"))],
            )
        for result in response.json()["results"]:
            self.assertIn("needs_review", result)
            self.assertIsInstance(result["needs_review"], bool)


if __name__ == "__main__":
    unittest.main()
