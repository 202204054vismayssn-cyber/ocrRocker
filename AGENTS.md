# Multi-Country Invoice Reader - Agent Instructions

## Project objective

Build a lightweight, production-ready, asynchronous FastAPI service that accepts invoices (PDF or image) from UAE, USA, and India, extracts text using local PaddleOCR on GPU/CPU, and sends the extracted text directly to an Ollama-hosted Gemma model for structured JSON extraction.

The architecture is streamlined for execution locally or on Google Colab with GPU acceleration, processing 4 to 5 invoices concurrently using asynchronous pipelines.

## Required reading order

Before changing code, read these files completely:

1. `AGENTS.md`
2. `ARCHITECTURE.md`
3. `pyproject.toml`
4. The code involved in the requested change

## Implementation reporting

When finishing a change, report:

- Exact files changed.
- API fields or environment variables added.
- Validation performed and its result.
- Anything that could not be validated at runtime.

## Application layering

- Put country-specific code under `countries/<country>/` (`uae`, `usa`, `india`). Each country owns its system prompt, direct Pydantic response schema, business validation, and routes.
- Put only country-neutral infrastructure under `common/`: configuration, security, file validation, PaddleOCR rendering, Ollama async client, async semaphore flow control, and system endpoints.
- Code under `common/` must never import a country module.
- Follow a streamlined `dto / routes / services` structure without bloated helper abstractions.
- DTOs define clean, descriptive public API contracts and Pydantic models directly matching the LLM output (no alias compression).
- Services implement file validation, PDF rendering, PaddleOCR extraction, and the async Ollama client.

## Ollama and Gemma rules

- Connect to Ollama via its asynchronous REST API (default `http://127.0.0.1:11434`) using `OLLAMA_BASE_URL` and `OLLAMA_MODEL`.
- Default target model is Gemma (e.g. `gemma2:2b`, `gemma2:9b`, or user-specified Gemma light variant).
- Request structured JSON directly from Ollama by passing the target Pydantic JSON schema in the API request format (`format=Schema.model_json_schema()`).
- Do NOT use cryptic single-letter alias keys (e.g. `i` for invoice number, `id` for invoice date, `sn`, `sa`). Use descriptive, readable field names (`invoice_number`, `invoice_date`, `vendor_name`, etc.) directly in the prompt and schema.
- Allow callers to pass a custom prompt or company-specific prompt override per request to accommodate different organizational invoice patterns.
- Keep Ollama temperature at `0` for reproducible invoice field extraction.
- Support parallel execution: set `OLLAMA_NUM_PARALLEL=4` (or 5) in Ollama's environment to allow multiple model evaluations simultaneously.

## OCR rules (PaddleOCR only)

- Use **PaddleOCR exclusively** as the single OCR engine for all documents.
- For PDF documents: Render each page to an image using PyMuPDF (`page.get_pixmap()`), then feed each rendered page image directly to PaddleOCR.
- For images (JPG, JPEG, PNG): Feed directly into PaddleOCR without intermediary conversion.
- Support GPU acceleration (`PADDLE_DEVICE=gpu` with CUDA) when running on Google Colab or GPU instances; fall back gracefully to `cpu`.
- Run synchronous PaddleOCR operations in worker threads using `asyncio.to_thread` to prevent blocking the FastAPI event loop.
- Group detected text blocks into natural, reading-order text lines before feeding to Gemma.

## Concurrency and Async execution

- FastAPI endpoints must be fully `async`.
- Concurrency limit is controlled by `MAX_CONCURRENT_EXTRACTIONS` (default `4` or `5`).
- Use an `asyncio.Semaphore` to manage concurrent requests, allowing 4 to 5 invoices to be processed simultaneously on GPU.
- Long-running OCR and LLM calls run concurrently across multiple requests.

## Country invoice rules

### UAE
- Primary identifiers: Supplier TRN and Customer TRN (15 digits).
- Currency defaults to AED if not explicitly printed.
- VAT breakdown (net amount, VAT rate 5%, total VAT, grand total).
- Validate item lines and totals without fabricating missing values.

### USA
- Vendor EIN/FEIN separate from Customer Tax ID/EIN.
- State Tax Permit / Resale Certificate.
- Sales tax rates in percentage points.
- Grand total vs. Amount due / balance payable.

### India
- Vendor GSTIN and Customer GSTIN (15 characters).
- Customer PAN derived from GSTIN (characters 3-12) if PAN is missing.
- Separate CGST, SGST, and IGST components.
- HSN/SAC codes, item quantities, and IRN numbers.

## API and Security

- Primary endpoints:
  - `POST /api/v1/invoices/extract` (UAE)
  - `POST /api/v1/usa/invoices/extract` (USA)
  - `POST /api/v1/india/invoices/extract` (India)
- Accept `multipart/form-data` with:
  - `file`: PDF, JPEG, or PNG invoice.
  - `prompt` (optional): Custom company prompt override.
- Optional API key authentication via `SAVVY_API_KEY` header.
- `GET /health` endpoint reporting service health, current Ollama model, and device status.
- Return consistent JSON responses and clear error codes (400 for invalid inputs, 422 for unreadable text, 502/503 for Ollama connection issues).
