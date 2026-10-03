import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "qwen_semantic_fallback.py"
SPEC = importlib.util.spec_from_file_location("qwen_semantic_fallback_test", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


SCHEMA = {
    "customer_name": {"required": True, "type": "str"},
    "invoice_number": {"required": True, "type": "str"},
    "invoice_date": {"required": True, "type": "date"},
    "total_amount": {"required": True, "type": "float"},
}


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def chat(self, prompt, response_schema, image_paths=None):
        self.calls.append({"prompt": prompt, "images": list(image_paths or [])})
        return next(self.responses)


class FailingClient:
    def chat(self, prompt, response_schema, image_paths=None):
        raise RuntimeError("Ollama is offline")


def result_with_missing(*names):
    fields = {
        "customer_name": "Existing Customer",
        "invoice_number": "INV-1",
        "invoice_date": "2026-05-31",
        "total_amount": 118.0,
    }
    for name in names:
        fields[name] = None
    return {
        "fields": fields,
        "field_sources": {},
        "missing_required": list(names),
        "validation_issues": [],
        "needs_review": bool(names),
    }


class QwenFallbackTests(unittest.TestCase):
    def test_does_not_call_model_when_nothing_is_missing(self):
        client = FakeClient([])
        fallback = module.QwenSemanticFallback(SCHEMA, client=client)
        output = fallback.apply(result_with_missing(), raw_text="complete")
        self.assertEqual(client.calls, [])
        self.assertNotIn("semantic_fallback", output)

    def test_text_tier_fills_only_missing_and_preserves_existing_values(self):
        client = FakeClient([{"fields": {"customer_name": "Bajaj", "total_amount": 999}}])
        fallback = module.QwenSemanticFallback(SCHEMA, client=client)
        output = fallback.apply(result_with_missing("customer_name"), raw_text="Buyer Bajaj")
        self.assertEqual(output["fields"]["customer_name"], "Bajaj")
        self.assertEqual(output["fields"]["total_amount"], 118.0)
        self.assertEqual(output["field_sources"]["customer_name"], "qwen_text_fallback")
        self.assertEqual(output["missing_required"], [])

    def test_scanned_result_escalates_to_vision_after_text(self):
        client = FakeClient([
            {"fields": {"invoice_number": None}},
            {"fields": {"invoice_number": "SCAN-42"}},
        ])
        fallback = module.QwenSemanticFallback(SCHEMA, client=client)
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "page.png"
            image.write_bytes(b"png")
            output = fallback.apply(
                result_with_missing("invoice_number"),
                raw_text="unclear OCR",
                image_paths=[str(image)],
            )
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[0]["images"], [])
        self.assertEqual(client.calls[1]["images"], [str(image)])
        self.assertEqual(output["field_sources"]["invoice_number"], "qwen_vision_fallback")

    def test_native_result_never_escalates_without_images(self):
        client = FakeClient([{"fields": {"invoice_number": None}}])
        fallback = module.QwenSemanticFallback(SCHEMA, client=client)
        output = fallback.apply(result_with_missing("invoice_number"), raw_text="native text")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(output["missing_required"], ["invoice_number"])

    def test_failure_is_non_fatal_and_ship_to_customer_is_rejected(self):
        client = FakeClient([{"fields": {"customer_name": "Ship To: Warehouse"}}])
        fallback = module.QwenSemanticFallback(SCHEMA, client=client)
        output = fallback.apply(result_with_missing("customer_name"), raw_text="Ship To Warehouse")
        self.assertIsNone(output["fields"]["customer_name"])
        self.assertEqual(output["missing_required"], ["customer_name"])

    def test_offline_ollama_preserves_result_instead_of_crashing(self):
        fallback = module.QwenSemanticFallback(SCHEMA, client=FailingClient())
        output = fallback.apply(result_with_missing("invoice_number"), raw_text="invoice")
        self.assertIsNone(output["fields"]["invoice_number"])
        self.assertEqual(output["missing_required"], ["invoice_number"])
        self.assertIn("Ollama is offline", output["semantic_fallback"]["errors"][0])


if __name__ == "__main__":
    unittest.main()
