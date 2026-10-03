# Spec: Docker Image + REST API + Code Refactor
**Project:** invoice_ocr_qwen_ollama_v3  
**Date:** 2026-09-29  
**Status:** Approved for implementation

---

## 1. Goal

Package the existing invoice OCR pipeline into a production-ready Docker image with a REST API entry point, while refactoring the codebase for clarity and maintainability. The Qwen/Ollama fallback is kept as an optional future-ready capability, disabled by default.

---

## 2. Docker Image

### Base image
- `python:3.11-slim` (glibc required by PaddleOCR; alpine is not compatible)
- CPU-only runtime; no CUDA drivers in this image

### PaddleOCR model weights
- **Not baked in.** Mounted as a Docker volume at `/models` inside the container.
- The environment variable `PADDLE_HOME` is set to `/models` so PaddleOCR reads/writes model files there.
- On first run against an empty volume, PaddleOCR downloads its model automatically. Subsequent runs reuse the cached weights from the volume.
- Users who want fully-offline operation download the model once, then keep the volume.

### Output
- Results written to `/output` inside the container, mounted as a volume by the caller.

### Entry point
- The container starts the REST API server (see §3). It does **not** run `pdf_router.py` directly.

### Environment variables (all optional)
| Variable | Default | Purpose |
|---|---|---|
| `PADDLE_HOME` | `/models` | PaddleOCR model cache directory |
| `OUTPUT_DIR` | `/output` | Where extracted JSON/XML is written |
| `OLLAMA_HOST` | `http://ollama:11434` | Ollama sidecar URL (only used when `--enable-qwen` flag is active) |
| `INVOICE_QWEN_MODEL` | `qwen2.5vl:3b` | Qwen model name |
| `API_HOST` | `0.0.0.0` | API bind host |
| `API_PORT` | `8000` | API bind port |

---

## 3. REST API (`api.py`)

Built with **FastAPI** + **Uvicorn**.

### Endpoints

#### `POST /extract`
Upload one or more PDF files (or one ZIP of PDFs). Returns structured extraction results.

**Request:** `multipart/form-data`
- `file` — one PDF, ZIP, or multiple PDFs
- `export_xml` (bool, default `false`) — include XML alongside JSON
- `enable_qwen` (bool, default `false`) — activate Qwen semantic fallback (requires Ollama sidecar)

**Response:** `application/json`
```json
{
  "results": [ /* one object per logical invoice */ ],
  "summary": {
    "total_invoices": 3,
    "needs_review": 1,
    "extraction_methods": { "native_pdf": 2, "scanned_ocr": 1 }
  }
}
```

**Error responses:**
- `400` — unsupported file type or no PDFs found in ZIP
- `422` — validation error on request params
- `500` — internal extraction error (deterministic result preserved where possible)

#### `GET /health`
Returns `{"status": "ok"}`. Used by Docker health check and load balancers.

#### `GET /`
Returns API version and available endpoints (simple discovery response).

### Design constraints
- One `OCRPipelineManager` instance is shared across requests (lazy-loaded on first scanned page, reused thereafter).
- Uploaded files are written to a `tempfile.TemporaryDirectory` per request and cleaned up on completion.
- The API never writes to the mounted `/output` volume directly — that is the CLI path. API responses are returned as JSON in the HTTP response body.

---

## 4. docker-compose.yml

Two services defined, one profile:

### Default (no profile)
```
invoice-ocr:
  build: .
  ports: ["8000:8000"]
  volumes:
    - ./models:/models
    - ./output:/output
  environment:
    PADDLE_HOME: /models
    OUTPUT_DIR: /output
```

### `ollama` profile (opt-in)
```
ollama:
  image: ollama/ollama
  profiles: ["ollama"]
  volumes:
    - ollama_data:/root/.ollama
  ports: ["11434:11434"]
```

When the `ollama` profile is active, `invoice-ocr` gains `OLLAMA_HOST: http://ollama:11434` and depends on the `ollama` service. Activate with:
```
docker compose --profile ollama up
```

---

## 5. Code Refactor

### Principles
- No logic changes that alter extraction behaviour.
- All existing test imports must continue to work (backward-compatible re-exports).
- Every public function gets a one-line docstring if it is missing one.
- Inline comments on non-obvious logic blocks.

### `pdf_router.py` split

The file is ~900 lines covering four distinct responsibilities. Split into:

| New file | Responsibility | Key contents |
|---|---|---|
| `router/page_detector.py` | Decide native vs scanned for each page | `page_is_native`, `render_page_to_image`, `OCRPipelineManager` |
| `router/page_extractor.py` | Extract one page via either path | `extract_native_page`, `_run_scanned_page`, `_attach_router_metadata` |
| `router/invoice_grouper.py` | Group pages into logical invoices, merge | `_group_page_results`, `_merge_invoice_group`, helpers |
| `router/batch.py` | Orchestrate multi-PDF batches, write outputs | `process_pdf`, `run_pdf_batch`, `collect_pdf_inputs`, `_write_invoice_result` |
| `pdf_router.py` (thin) | Re-export everything for backward compat | `from router.batch import *` etc. |

### Other files
- `native_invoice_parser.py` — already well-structured; add missing docstrings only
- `qwen_semantic_fallback.py` — already clean; add missing docstrings only
- `presentation_ocr.py` — already clean; no changes needed

### New files added
| File | Purpose |
|---|---|
| `api.py` | FastAPI REST API (see §3) |
| `Dockerfile` | Container build instructions (see §2) |
| `docker-compose.yml` | Service orchestration (see §4) |
| `.dockerignore` | Exclude `__pycache__`, `output/`, `*.pyc`, sample files from build context |
| `router/__init__.py` | Package marker |

---

## 6. Out of scope (this iteration)

- GPU/CUDA Docker variant
- Authentication or rate limiting on the API
- Async extraction (requests are handled synchronously; PaddleOCR is not thread-safe)
- Baking Ollama or model weights into the image
- Changes to `invoice_extraction_colab.py` (untouched, as per existing design)

---

## 7. Acceptance criteria

- [ ] `docker compose up` starts the API on port 8000
- [ ] `POST /extract` with a sample PDF returns valid JSON with `results` and `summary`
- [ ] `GET /health` returns `{"status": "ok"}`
- [ ] `docker compose --profile ollama up` starts both services; `enable_qwen=true` routes through Ollama
- [ ] All existing tests pass: `python -m unittest discover -s tests -v`
- [ ] `pdf_router.py` direct CLI still works: `python pdf_router.py invoice.pdf`
- [ ] No `__pycache__` or output files in the Docker build context
