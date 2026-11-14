FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PADDLE_PDX_CACHE_HOME="/app/.model_cache/paddle"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv==0.12.5

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --no-dev

COPY main.py ./

# Keep the API unprivileged. The named Paddle cache volume inherits this
# directory ownership when Docker creates it.
RUN groupadd --gid 10001 appgroup \
    && useradd --uid 10001 --gid appgroup --home-dir /app --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/.model_cache/paddle \
    && chown -R appuser:appgroup /app/.model_cache

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
