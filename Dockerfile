# syntax=docker/dockerfile:1
# ──────────────────────────────────────────────────────────────────────────────
# Invoice OCR API — CPU-only image
#
# Build:
#   docker build -t invoice-ocr .
#
# Run:
#   docker run -p 8000:8000 \
#     -v ./models:/models \
#     -v ./output:/output \
#     invoice-ocr
#
# The /models volume is where PaddleOCR caches model weights.  On the first
# run against an empty volume the weights are downloaded automatically (~200 MB).
# Subsequent runs reuse the cache — no re-download needed.
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
# libgomp1    — OpenMP, required by PaddlePaddle CPU kernels
# libglib2.0-0 — required by OpenCV (imported transitively by PaddleOCR)
# libgl1      — required by OpenCV for image I/O
# libpoppler-cpp-dev — required by pdf2image (optional but avoids noisy warnings)
# We install then clean in a single RUN layer to keep the layer count low.

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
# Install before copying application code so Docker layer caching works:
# a code change does not invalidate the (slow) pip install layer.

WORKDIR /app

COPY requirements.txt .

# paddlepaddle CPU wheel — must match the version expected by paddleocr>=2.9.
# PP-OCRv6 models (used by invoice_extraction_colab.py) require PaddlePaddle 3.x.
# The official CPU-only index provides a slim wheel without CUDA dependencies.
RUN pip install --no-cache-dir paddlepaddle==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ \
 && pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
# Copy only the source files needed at runtime.  Everything else is excluded
# by .dockerignore.

COPY invoice_extraction_colab.py .
COPY native_invoice_parser.py .
COPY qwen_semantic_fallback.py .
COPY pdf_router.py .
COPY presentation_ocr.py .
COPY api.py .
COPY router/router/ router/
COPY test_upload.html .

# ── Runtime environment ───────────────────────────────────────────────────────

# PaddleOCR model cache — mount a volume here to persist across container runs.
ENV PADDLE_HOME=/models

# Where the router writes per-invoice JSON/XML output files.
# Callers mount a volume here to retrieve results.
ENV OUTPUT_DIR=/output

# Qwen/Ollama sidecar — only consulted when enable_qwen=true in the request.
ENV OLLAMA_HOST=http://ollama:11434
ENV INVOICE_QWEN_MODEL=qwen2.5vl:3b

# FastAPI server bind settings — override at runtime with -e if needed.
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Create the volume mount points so Docker knows they are intended mount points.
RUN mkdir -p /models /output

# ── Health check ──────────────────────────────────────────────────────────────
# Docker / docker-compose / orchestrators use this to determine readiness.
# The first health check is delayed 30 s to allow PaddleOCR model loading
# on cold start.

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c \
        "import urllib.request, sys; \
         r = urllib.request.urlopen('http://localhost:' + __import__('os').environ.get('API_PORT','8000') + '/health', timeout=8); \
         sys.exit(0 if r.status == 200 else 1)"

# ── Entry point ───────────────────────────────────────────────────────────────

EXPOSE 8000

CMD uvicorn api:app \
      --host "${API_HOST}" \
      --port "${API_PORT}" \
      --workers 1 \
      --timeout-keep-alive 75
