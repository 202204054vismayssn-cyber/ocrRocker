import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "native_invoice_parser.py"
SPEC = importlib.util.spec_from_file_location("native_invoice_parser", MODULE_PATH)
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


ZOHO_TABLE = [
    ["", "SACHIN SHUKLA\nMaharashtra\nIndia\nGSTIN 27IDSPS9135M1ZA TAX INVOICE", None, None, None, None, None, None, None, None, None, None, None, None, ""],
    [None, "# : INV-000129\nInvoice Date : 13/04/2026", None, None, None, None, "Place Of Supply : Maharashtra (27)", None, None, None, None, None, None, None, None],
    [None, "Bill To", None, None, None, None, "Ship To", None, None, None, None, None, None, None, None],
    [None, "M-Fins Services Pvt Ltd\nBill address\nGSTIN 27AAHCM3839P2ZQ", None, None, None, None, "Different shipping address", None, None, None, None, None, None, None, None],
    [None, "#", "Item & Description", "HSN/SAC", "Qty", "Rate", None, "CGST", None, None, "SGST", None, None, "Amount", None],
    [None, None, None, None, None, None, None, "%", None, "Amt", "%", "Amt", None, None, None],
    [None, "1", "Solar system", "8541", "1.00", "1,08,600.00", None, "2.5%", None, "2,715.00", "2.5%", "2,715.00", None, "1,08,600.00", None],
    [None, "2", "Installation", "995461", "1.00", "46,900.00", None, "9%", None, "4,221.00", "9%", "4,221.00", None, "46,900.00", None],
]


class NativeInvoiceParserTests(unittest.TestCase):
    def test_bare_hash_invoice_number_and_personal_supplier(self):
        text = """SACHIN SHUKLA
GSTIN 27IDSPS9135M1ZA TAX INVOICE
# : INV-000129 Place Of Supply : Maharashtra (27)
Invoice Date : 13/04/2026
Bill To Ship To
"""

        self.assertEqual(native.extract_invoice_number(text), "INV-000129")
        self.assertEqual(
            native.extract_supplier([ZOHO_TABLE], text)["name"],
            "SACHIN SHUKLA",
        )

    def test_bill_to_geometry_never_merges_ship_to(self):
        bill_to = native.extract_bill_to([ZOHO_TABLE])

        self.assertEqual(bill_to["name"], "M-Fins Services Pvt Ltd")
        self.assertEqual(bill_to["address"], "Bill address")
        self.assertEqual(bill_to["gstin"], "27AAHCM3839P2ZQ")
        self.assertNotIn("shipping", str(bill_to).lower())

    def test_line_items_and_multi_rate_tax_summary(self):
        text = """Sub Total 1,55,500.00
CGST2.5 (2.5%) 2,715.00
SGST2.5 (2.5%) 2,715.00
CGST9 (9%) 4,221.00
Notes SGST9 (9%) 4,221.00
Thanks for your business.
Total ₹1,69,372.00
Balance Due ₹1,69,372.00
"""
        items = native.parse_line_items([ZOHO_TABLE])
        totals = native.extract_totals(text, items)

        self.assertEqual(len(items), 2)
        self.assertEqual(sum(item["amount"] for item in items), 155500.0)
        self.assertEqual(totals["subtotal"], 155500.0)
        self.assertEqual(totals["cgst_amount"], 6936.0)
        self.assertEqual(totals["sgst_amount"], 6936.0)
        self.assertEqual(totals["igst_amount"], 0.0)
        self.assertEqual(totals["total_amount"], 169372.0)
        self.assertEqual(totals["balance_due"], 169372.0)


if __name__ == "__main__":
    unittest.main()
