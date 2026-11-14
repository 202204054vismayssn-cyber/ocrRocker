# Multi-Country Invoice Reader Architecture

## 1. Purpose

This service is a high-throughput, asynchronous invoice processing REST API supporting **UAE**, **USA**, and **India** invoices. 

All document OCR is processed locally using **PaddleOCR** (on GPU or CPU). Extracted text is sent asynchronously to a local or hosted **Ollama** server running the **Gemma** model (e.g. Gemma 2B or 9B, or Gemma 4 variant). Gemma directly generates structured, schema-validated JSON with descriptive, human-readable field names (such as `invoice_number`, `invoice_date`, `vendor_name`, etc.), completely eliminating compressed alias keys (`i`, `id`, `sn`, etc.).

The system is optimized for deployment both in production environments and on **Google Colab with GPU acceleration**, processing **4 to 5 invoices concurrently** using non-blocking asynchronous workflows.

---

## 2. Technology Stack

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Framework** | FastAPI (Python 3.13) | Native `async`/`await` support for high concurrency. |
| **OCR Engine** | PaddleOCR (`PP-OCRv5`) | Single, unified OCR engine for scanned pages and images. GPU-accelerated. |
| **PDF Rendering** | PyMuPDF (`fitz`) | Renders PDF pages into images before feeding to PaddleOCR. |
| **LLM Provider** | Ollama | Local model runner supporting CUDA GPU acceleration and parallel request processing. |
| **LLM Model** | Gemma (e.g. `gemma2:2b`, `gemma2:9b`) | Fast, lightweight, instruction-tuned model for structured extraction. |
| **Structured Output** | Native Ollama JSON Schema | Directly constrains Gemma output to match Pydantic schemas. |
| **Flow Control** | `asyncio.Semaphore` | Dynamically limits active GPU extractions (default 4–5 concurrent). |
| **Configuration** | Pydantic Settings | Environment-driven settings with `.env` support. |

---

## 3. Asynchronous Request Flow

```text
Client (Postman / Web App / Script)
  |
  |  POST /api/v1/{country}/invoices/extract
  |  (File: PDF/Image, Optional: Custom Prompt)
  v
Authentication & Rate/Concurrency Gate
  `-- asyncio.Semaphore (allows 4-5 simultaneous extractions)
  v
File Validation & Temporary Workspace
  `-- Validates signature (PDF, JPG, PNG) and dimensions
  v
Unified PaddleOCR Extraction (in threadpool via asyncio.to_thread)
  |-- PDF  -> PyMuPDF renders pages to pixmaps -> PaddleOCR
  `-- JPG/PNG -> PaddleOCR directly
  v
Text Formatting & Compaction
  `-- Reconstructs natural line reading order
  v
Asynchronous Ollama Request (via httpx / ollama-python)
  `-- Calls Ollama /api/chat with model=gemma, format=PydanticSchema
  `-- Injects Country System Prompt or Caller's Custom Company Prompt
  v
Descriptive JSON Parsing & Pydantic Validation
  `-- Validates types, dates, currency, and country tax rules
  v
Consistent JSON API Response (with token & execution metrics)
```

---

## 4. Descriptive Schema (No Compressed Aliases)

The previous cryptic alias mapping (`i`, `id`, `sn`, `sa`, etc.) is completely removed. Gemma directly outputs clean, readable JSON keys matching the API contract:

### Example Output Structure
```json
{
  "success": true,
  "country": "uae",
  "data": {
    "invoice_number": "INV-2026-0042",
    "invoice_date": "2026-03-15",
    "due_date": "2026-04-15",
    "currency": "AED",
    "vendor_name": "Gulf Tech Supplies LLC",
    "vendor_address": "Downtown Dubai, UAE",
    "vendor_trn_no": "100234567800003",
    "buyer_name": "Al Amal Trading Co.",
    "buyer_address": "Business Bay, Dubai, UAE",
    "buyer_trn_no": "100987654300003",
    "net_amount": 1000.00,
    "vat_rate": 5.0,
    "total_tax": 50.00,
    "grand_total": 1050.00,
    "items": [
      {
        "item": "Network Switch 24-Port",
        "qty": 2.0,
        "unit_price": 500.00,
        "net_amount": 1000.00,
        "vat_rate": 5.0,
        "line_total": 1050.00
      }
    ]
  },
  "extracted_text": "...",
  "execution_time_ms": 1420
}
```

---

## 5. Country Modules

Each country retains an isolated package under `src/uae_invoice_reader/countries/`:

1. **UAE (`/api/v1/invoices`)**:
   - Focus: 15-digit TRNs (Tax Registration Numbers) for vendor and buyer.
   - Standard 5% VAT rate, net amount, total tax, and grand total.
   - Default currency: AED.

2. **USA (`/api/v1/usa/invoices`)**:
   - Focus: Vendor EIN/FEIN, Buyer Tax ID.
   - State Tax Permits, Resale Certificates.
   - Sales tax rates in percentage points, amount due vs grand total.

3. **India (`/api/v1/india/invoices`)**:
   - Focus: 15-character GSTIN, auto-derived PAN from GSTIN (chars 3–12).
   - Breakdown of CGST, SGST, IGST.
   - HSN/SAC codes, item quantities, and IRN numbers.

---

## 6. PaddleOCR-Only Pipeline

- To ensure consistent, robust text extraction across all invoice formats, the heuristic selectable-text routing is replaced by a **PaddleOCR-only pipeline**.
- **PDFs**: Each page is rendered at 200 DPI to an image in memory via PyMuPDF. Each rendered page image is passed directly into PaddleOCR.
- **Images**: PNG, JPG, and JPEG files are passed directly to PaddleOCR.
- Synchronous PaddleOCR inference runs within Python's `asyncio.to_thread` executor, preventing any blocking of concurrent network traffic or other requests.
- When `PADDLE_DEVICE=gpu` is configured, PaddleOCR utilizes CUDA on the GPU for sub-second page recognition.

---

## 7. Concurrency & Google Colab GPU Optimization

On a Google Colab GPU (e.g. **NVIDIA T4 16GB VRAM** or **A100**):
- **Ollama Parallelism**: Ollama is launched with `OLLAMA_NUM_PARALLEL=4` (or 5) to serve multiple concurrent inferences on the GPU.
- **VRAM Budget**:
  - Gemma 2B/9B model: ~3 GB to 6 GB VRAM.
  - PaddleOCR GPU: ~1.5 GB to 2 GB VRAM.
  - 4x parallel context slots: ~3 GB to 4 GB VRAM.
  - Total requirement: ~8 GB to 12 GB, fitting within a single 16 GB T4 GPU.
- **FastAPI Concurrency**: `MAX_CONCURRENT_EXTRACTIONS=4` or `5` controls simultaneous active extractions. Additional incoming requests queue safely in the `asyncio.Semaphore` without failing.

---

## 8. Cleaned-Up & Streamlined Codebase

The following bloated components have been pruned:
- Obsolete Google GenAI Batch API endpoints (`/batch/prepare`, `/batch/submit`, `/batch/status`, `/batch/result`).
- Cryptic short-key alias mapping dictionaries and expansion services.
- Legacy comparison and benchmark endpoints (`/test/tesseract-text`, etc.).
- Complex duplicated flow controllers and redundant wrappers.
