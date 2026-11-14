# Multi-country Invoice Reader

FastAPI service for UAE, USA, and India invoices. Selectable PDF pages are read locally with PyMuPDF. Scanned PDF pages and JPG/JPEG/PNG images use the memory-efficient PaddleOCR models on GPU/CPU. Compacted extracted text is sent directly to a local or remote Ollama-hosted Gemma model (`gemma2:2b`) with native JSON Schema output.

## Architecture

The application follows country modules under `countries/uae`, `countries/usa`,
and `countries/india`, each with `models / dto / controllers / routes / services`.
Reusable upload validation, PaddleOCR, security, Ollama client, middleware, and operational services
live under `common`. See `ARCHITECTURE.md` for the dependency boundaries.

## Configuration

Copy `.env.example` values into `.env` and set:

```text
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=gemma2:2b
OLLAMA_TIMEOUT_SECONDS=120.0
SAVVY_API_KEY=your-private-client-key
ADMIN_API_KEY=your-separate-private-admin-key
ENABLE_SELF_RESTART=false
RESTART_DELAY_SECONDS=2
MAX_CONCURRENT_EXTRACTIONS=4
LOG_LEVEL=INFO
PADDLE_DEVICE=gpu
PADDLE_LANG=en
```

The default extraction model is `gemma2:2b` served via Ollama.

## Install and run

```powershell
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

Binding to `0.0.0.0` makes the API reachable through the server's network
address. Internet access still depends on the host firewall, router/cloud port,
DNS, and HTTPS reverse-proxy configuration.

The first scanned document may download Paddle model files into `.model_cache/paddle`. Subsequent inference uses the local cache.

## Developer logs

Sanitized operational logs are written to the container console and never need
a writable application directory. They contain request IDs, HTTP methods,
endpoint paths, status codes, durations, and safe error source/code fields. They
never record uploaded filenames, invoice/OCR text, prompts, form values,
company/invoice IDs, financial values, API keys, or raw LLM responses. Follow
them with:

```powershell
docker compose logs -f api
```

For Docker, build and start the API:

```powershell
docker compose up -d --build
docker compose ps
```

## API

Health:

```powershell
Invoke-RestMethod http://127.0.0.1:4207/health
```

Docker publishes host port `4207` to container port `8000`. When running Uvicorn
directly with the development command above, use port `8000` unless you also
change that command to `--port 4207`.

Compose development mode bind-mounts `src/` and `main.py`. With
`AUTO_RELOAD=true`, Uvicorn checks for Python changes every 5 seconds and reloads
the application automatically. Change `RELOAD_DELAY_SECONDS` in `.env` to choose
a delay between 5 and 10 seconds. Changes to dependencies, `Dockerfile`, Compose,
or `.env` still require rebuilding or recreating the container.

Extraction endpoint:

```text
POST /api/v1/invoices/extract
SAVVY_API_KEY: <your-private-client-key>
multipart field: file
```

The request waits until local extraction and Ollama/Gemma processing return a result.
For normal bulk uploads, clients call this endpoint sequentially or concurrently.

Extract a completed invoice with lightweight PaddleOCR and Gemma:

```text
POST /api/v1/invoices/complete
SAVVY_API_KEY: <your-private-client-key>
multipart fields:
  file: PDF/JPG/JPEG/PNG
  invoice_id: invoice row ID
```

This UAE-only endpoint renders every PDF page and runs local
PaddleOCR. Extracted text is sent to Ollama (Gemma 2). It returns the same structured response
envelope as `/extract`, plus the unchanged top-level `invoice_id`.

Example form values:

```text
invoice_id=123
```

Live processing status:

```text
GET /api/v1/system/processing-status
SAVVY_API_KEY: <your-private-client-key>
```

The response reports active count, capacity, request IDs, safe processing
phases, start times, elapsed seconds, and up to 20 recent sanitized errors.
Errors identify `client`, `docker_api`, `paddle_ocr`, or `ollama` as
their source. It does not return invoice contents or raw provider responses.

Safe container restart:

```text
POST /api/v1/system/restart
ADMIN_API_KEY: <your-separate-private-admin-key>
```

Set `ENABLE_SELF_RESTART=true` only in Docker production. The endpoint returns
HTTP 409 while an invoice is active and HTTP 202 when an idle restart is
scheduled. Docker must retain `restart: unless-stopped`. The API does not mount
or use the Docker socket.

Batch workflow:

```text
POST /api/v1/invoices/batch/prepare
POST /api/v1/invoices/batch/submit
GET  /api/v1/invoices/batch/status/{batch_job_name}
GET  /api/v1/invoices/batch/result/{batch_job_name}
```

`batch/prepare` accepts one invoice file and performs local PaddleOCR extraction only.
Store its `character_stored`, `total_pages`, and `character_count` values.
`batch/submit` accepts JSON containing 1 to 100 prepared texts. It immediately returns a
`batch_job_name`, such as `batches/loc_20261019_164800_a1b2c3d4`, and schedules items for
asynchronous parallel GPU extraction with Ollama Gemma. Poll `batch/status/{batch_job_name}`
to track progress, and call `batch/result/{batch_job_name}` to retrieve completed structured extractions.
Every endpoint requires the `SAVVY_API_KEY` header.

USA workflows use the independent `/api/v1/usa/invoices` prefix. They provide
the same extract, complete, OCR comparison, prepare, submit, status, and result
operations shown above. USA responses use USA sales-tax, EIN, resale-certificate,
ABA routing, freight, amount-due, and item fields and intentionally omit the
top-level `validation` object.

India invoice extraction uses the independent endpoint:

```text
POST /api/v1/india/invoices/extract
SAVVY_API_KEY: <your-private-client-key>
multipart field: file
```

It reuses the shared PyMuPDF and PaddleOCR mobile pipeline, sends compacted text
directly to Ollama (Gemma 2), and returns India GST/PAN, CGST/SGST/IGST, HSN/SAC, IRN, bank,
item, and HSN-summary fields. India also provides `/complete` routes under the same India prefix.
`/complete` forces lightweight PaddleOCR for every PDF page. India success responses intentionally omit
the top-level `validation` object.

When taxable amount is not printed, the India prompt requires a bounded
calculation from printed item amounts or from the printed total and available
tax, charge, and discount components. Multiple HSN/SAC codes are requested as a
unique comma-separated string. A null item quantity is returned as `1`.
When customer PAN is missing, it is derived from a structurally valid customer
GSTIN; an explicitly extracted customer PAN is preserved.
India item `sr_no` values are assigned automatically in response order as
`"1"`, `"2"`, `"3"`, and so on.
Missing India `igst_amount`, `discount`, `state_cess`, `other_charges`, and
`irn_number` fields serialize cleanly. A missing top-level HSN/SAC value
is rebuilt from extracted item or HSN-summary codes; nullable summary amounts
remain unchanged.

## Tests

```powershell
uv run pytest
```

## Postman

Import `postman/UAE_Invoice_Reader.postman_collection.json`, set its `apiKey`
variable to your `SAVVY_API_KEY`, set `adminApiKey` only when testing restart,
and choose an invoice for the `file` form-data field.

For USA invoices, import `postman/USA_Invoice_Reader.postman_collection.json`.

For India invoices, import
`postman/India_Invoice_Reader.postman_collection.json`, set `apiKey`, and select
an invoice file in the `invoiceFile` collection variable or directly in the
request's `file` form-data row. The collection contains extract, complete,
PaddleOCR and Tesseract comparisons, plus batch prepare, submit, status,
and result requests.
