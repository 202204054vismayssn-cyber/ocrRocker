# 04: Write `Dockerfile` and `.dockerignore`

**What to build:** A `Dockerfile` that produces a working CPU-only container image for the invoice OCR API. The image uses `python:3.11-slim` as its base, installs CPU PaddleOCR and all other dependencies from `requirements.txt`, and starts the FastAPI server via Uvicorn as its entry point.

PaddleOCR model weights are not baked in — they are expected on a volume mounted at `/models` (controlled by `PADDLE_HOME=/models`). On first run against an empty volume PaddleOCR downloads the model automatically; subsequent runs reuse the cached weights.

A `.dockerignore` file excludes `__pycache__`, `*.pyc`, `output/`, `pdf_router_output/`, `samples/`, `.scratch/`, and test fixtures from the build context so the image stays lean.

**Blocked by:** 01, 03

**Status:** ready-for-agent

- [ ] `docker build -t invoice-ocr .` completes without errors
- [ ] `docker run -p 8000:8000 -v ./models:/models invoice-ocr` starts and `GET /health` returns `{"status": "ok"}`
- [ ] `PADDLE_HOME` is set to `/models` in the image
- [ ] `OUTPUT_DIR`, `OLLAMA_HOST`, `INVOICE_QWEN_MODEL`, `API_HOST`, `API_PORT` are defined as ENV defaults with the values from the spec
- [ ] No `__pycache__`, `*.pyc`, `output/`, `pdf_router_output/`, `samples/` or `.scratch/` in the build context (verified via `.dockerignore`)
- [ ] Docker `HEALTHCHECK` calls `GET /health`
- [ ] Image size is reasonable for a CPU Python image (no unnecessary build tools left in final layer)
