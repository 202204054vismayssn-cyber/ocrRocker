# 03: Build the FastAPI REST API (`api.py`)

**What to build:** A new `api.py` module that wraps the invoice extraction pipeline behind an HTTP interface. Users upload a PDF, ZIP, or multiple PDFs to `POST /extract` and receive a structured JSON response. A single `OCRPipelineManager` instance is shared across all requests so the PaddleOCR model is loaded once and reused. A `GET /health` endpoint returns `{"status": "ok"}` for container health checks. A `GET /` endpoint returns the API version and available endpoints.

Uploaded files are written to a per-request temporary directory and cleaned up automatically on completion. The API does not write to the `/output` volume — results are returned in the HTTP response body only.

The `enable_qwen` request parameter activates the Qwen semantic fallback; when `false` (the default) Ollama is never contacted.

**Blocked by:** 01 (API imports from the refactored `router/` package)

**Status:** ready-for-agent

- [ ] `POST /extract` accepts `multipart/form-data` with `file`, `export_xml` (bool, default false), `enable_qwen` (bool, default false)
- [ ] Response shape: `{"results": [...], "summary": {"total_invoices": N, "needs_review": N, "extraction_methods": {...}}}`
- [ ] `GET /health` returns `{"status": "ok"}` with HTTP 200
- [ ] `GET /` returns API version and endpoint list
- [ ] Unsupported file type returns HTTP 400
- [ ] No PDFs found in ZIP returns HTTP 400
- [ ] One shared `OCRPipelineManager` instance across all requests (not recreated per request)
- [ ] Temporary upload directory cleaned up after each request
- [ ] Lightweight integration test: upload the sample invoice image from `samples/`, assert response contains `results` with at least one entry and `summary.total_invoices >= 1`
- [ ] `uvicorn api:app` starts the server without errors
