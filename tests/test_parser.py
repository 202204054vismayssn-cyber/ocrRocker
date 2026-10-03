import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "invoice_extraction_colab.py"
SPEC = importlib.util.spec_from_file_location("invoice_extraction_colab", MODULE_PATH)
ocr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ocr)


CANARA_18_TABLE = """
<table>
<tr><th rowspan="2">Sl No</th><th rowspan="2">Goods/Service Description</th>
<th rowspan="2">HSN/SAC</th><th rowspan="2">Discount</th>
<th rowspan="2">Taxable Value</th><th colspan="2">CGST</th>
<th colspan="2">SGST/UTGST</th><th colspan="2">IGST</th>
<th rowspan="2">Total Value</th></tr>
<tr><th>Rate</th><th>Amt</th><th>Rate</th><th>Amt</th>
<th>Rate</th><th>Amt</th></tr>
<tr><td>1</td><td>COMM - BAGIC GL</td><td>997161</td><td>0.00</td>
<td>286778.00</td><td>9.00</td><td>25810.02</td><td>9.00</td>
<td>25810.02</td><td>0.00</td><td>0.00</td><td>338398.04</td></tr>
<tr><td>2</td><td>COMM-HEALTH INSURANCE-BAGIC</td><td>997161</td><td>0.00</td>
<td>28771.00</td><td>9.00</td><td>2589.50</td><td>9.00</td>
<td>2589.50</td><td>0.00</td><td>0.00</td><td>33949.78</td></tr>
</table>
"""


CANARA_21_FULL_PAGE_TABLE = """
<table>
<tr><td colspan="12">Tax Invoice</td></tr>
<tr><td colspan="4">GSTIN Of Supplier: 21AAACC6106G3ZB</td>
<td colspan="8">Place of Supply (State Code): ODISHA-(21)</td></tr>
<tr><td colspan="4">Invoice No.: 21BAG0526000002</td>
<td colspan="8">Reverse Charge Applicable: NO</td></tr>
<tr><td colspan="12">Date of Invoice: 31-05-2026</td></tr>
<tr><td colspan="12">Details of Recipient</td></tr>
<tr><td colspan="12">Name: Bajaj General Insurance Limited</td></tr>
<tr><td colspan="12">Address: Bhubaneswar, Odisha</td></tr>
<tr><td colspan="12">State Name (State Code) ODISHA-(21)</td></tr>
<tr><td colspan="12">Customer GSTIN: 21AABCB5730GIZ9</td></tr>
<tr><th rowspan="2">SI No</th><th rowspan="2">Goods/Service Description</th>
<th rowspan="2">HSN/SAC</th><th rowspan="2">Discount</th>
<th rowspan="2">Taxable Value</th><th colspan="2">CGST</th>
<th colspan="2">SGST/UTGST</th><th colspan="2">IGST</th>
<th rowspan="2">Total Value</th></tr>
<tr><th>Rate</th><th>Ant</th><th>Rate</th><th>Ant</th>
<th>Rate</th><th>Ant</th></tr>
<tr><td>1</td><td>COMM- HEALTH- INSURANCE- BAGIC RETAIL</td><td>997161</td>
<td>0.00</td><td>83234.00</td><td>9.00</td><td>7491.00</td>
<td>9.00</td><td>7491.00</td><td>0.00</td><td>0.00</td><td>98216.12</td></tr>
<tr><td>2</td><td>COMM-GENERAL INSURANCE-BAGIC RETAIL</td><td>997161</td>
<td>0.00</td><td>808.00</td><td>9.00</td><td>72.50</td>
<td>9.00</td><td>72.50</td><td>0.00</td><td>0.00</td><td>953.44</td></tr>
<tr><td colspan="4"></td><td>0.00</td><td>84042.00</td><td></td>
<td>7563.50</td><td></td><td>7563.50</td><td></td><td>0.00</td></tr>
<tr><td colspan="12">Amount In Words : ninety nine thousand one hundred sixty nine rupees and fifty six paisa Only</td></tr>
<tr><td colspan="12">Refer Annexure for detailed Transaction Summary</td></tr>
<tr><td colspan="12">For Canara Bank</td></tr>
<tr><td colspan="12">DECLARATION:</td></tr>
<tr><td colspan="12">Declaration text</td></tr>
<tr><td colspan="12">E&amp;O.</td></tr>
<tr><td colspan="12">Authorised Signatory</td></tr>
<tr><td colspan="12">DISCLAIMER: computer generated invoice</td></tr>
<tr><th colspan="2">Rate</th><th colspan="2">SGST/UTGST</th>
<th colspan="6">CGST</th><th colspan="2">IGST</th></tr>
<tr><td colspan="2">18.00 %</td><td colspan="2">7563.50</td>
<td colspan="6">7563.50</td><td colspan="2">0.00</td></tr>
</table>
"""


# A structured table block sitting alongside the page's plain text — the shape
# a table-aware OCR model emits.  The current PP-OCRv6 pipeline returns plain
# text only, so this fixture exercises parse_markdown_output's HTML-table
# branch directly rather than through a PaddleOCR-VL result adapter.
CANARA_18_MARKDOWN = """Canara Bank
Tax Invoice
GSTIN Of Supplier: 18AAACC6106G2ZZ
Invoice No.: INVOICE
Date of Invoice: 31-05-2026
Place of Supply (State Code): ASSAM-(18)
Details of Recipient
Name: Details of Recipient
Bajaj General Insurance Limited
Customer GSTIN: 18AABCB5730G1ZW
Amount In Words: three lakh seventy two thousand three hundred forty seven rupees and eighty two paisa Only
""" + CANARA_18_TABLE


class ParserRegressionTests(unittest.TestCase):
    def test_full_page_table_with_late_item_header(self):
        parsed = ocr.parse_markdown_output(CANARA_21_FULL_PAGE_TABLE)
        result = ocr.build_form_ready_json(
            parsed,
            "CBSInvoice-21BAG05260000002-any-uuid.jpg",
        )
        fields = result["fields"]

        self.assertEqual(fields["customer_name"], "Bajaj General Insurance Limited")
        self.assertEqual(fields["customer_gstin"], "21AABCB5730G1Z9")
        self.assertEqual(fields["invoice_number"], "21BAG05260000002")
        self.assertEqual(len(fields["line_items"]), 2)
        self.assertEqual(fields["taxable_value"], 84042.0)
        self.assertEqual(fields["cgst_amount"], 7563.5)
        self.assertEqual(fields["sgst_amount"], 7563.5)
        self.assertEqual(fields["igst_amount"], 0.0)
        self.assertEqual(fields["total_amount"], 99169.56)
        self.assertEqual(fields["line_items"][0]["cgst_amount"], 7491.0)
        self.assertNotIn(
            "line_item_table_not_parsed",
            {issue["code"] for issue in result["validation_issues"]},
        )

    def test_structured_table_block_recovers_canara_items(self):
        parsed = ocr.parse_markdown_output(CANARA_18_MARKDOWN)
        result = ocr.build_form_ready_json(
            parsed,
            "CBSInvoice-18BAG05260000001-any-uuid.jpg",
        )
        fields = result["fields"]

        self.assertEqual(fields["invoice_number"], "18BAG05260000001")
        self.assertEqual(fields["customer_name"], "Bajaj General Insurance Limited")
        self.assertEqual(len(fields["line_items"]), 2)
        self.assertEqual(fields["taxable_value"], 315549.0)
        self.assertEqual(fields["cgst_amount"], 28399.52)
        self.assertEqual(fields["sgst_amount"], 28399.52)
        self.assertEqual(fields["total_amount"], 372347.82)
        self.assertIn(
            "total_arithmetic_mismatch",
            {issue["code"] for issue in result["validation_issues"]},
        )

    def test_amount_in_words_is_safe_total_fallback(self):
        self.assertEqual(
            ocr._amount_from_words(
                "sixty one thousand six hundred twenty one rupees and ninety six paisa Only"
            ),
            61621.96,
        )

    def test_literal_newline_does_not_contaminate_place_of_supply(self):
        parsed = ocr.parse_markdown_output(
            "Place of Supply (State Code): TRIPURA-(16)\\nReverse Charge Applicable: NO"
        )
        result = ocr.build_form_ready_json(parsed, "invoice.jpg")
        self.assertEqual(result["fields"]["place_of_supply"], "TRIPURA-(16)")

    def test_pipe_table_without_outer_pipes_is_supported(self):
        markdown = """Sl No | Description | HSN/SAC | Taxable Value | Total Value
--- | --- | --- | --- | ---
1 | Service fee | 997161 | 100.00 | 118.00
"""
        parsed = ocr.parse_markdown_output(markdown)
        items = ocr._normalize_line_items(parsed["line_items"])
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["taxable_value"], 100.0)
        self.assertEqual(items[0]["total_value"], 118.0)


if __name__ == "__main__":
    unittest.main()
