# 05: Write `docker-compose.yml` with optional Ollama profile

**What to build:** A `docker-compose.yml` that makes the invoice OCR service easy to run locally with a single command. By default (`docker compose up`) it starts only the `invoice-ocr` service on port 8000 with volumes wired for models and output. Adding `--profile ollama` brings up an `ollama` sidecar service and wires `OLLAMA_HOST` into the main container so `enable_qwen=true` API requests route through it.

An end-to-end smoke test script verifies the container starts, `/health` responds correctly, and a sample extraction request returns a valid response.

**Blocked by:** 04

**Status:** ready-for-agent

- [ ] `docker compose up` starts `invoice-ocr` on port 8000 with `./models:/models` and `./output:/output` volumes
- [ ] `GET /health` returns `{"status": "ok"}` after `docker compose up`
- [ ] `docker compose --profile ollama up` starts both `invoice-ocr` and `ollama` services
- [ ] When `ollama` profile is active, `OLLAMA_HOST=http://ollama:11434` is set on `invoice-ocr` and it depends on the `ollama` service
- [ ] `ollama` service uses the `ollama/ollama` image with a named volume for model persistence
- [ ] End-to-end smoke test: script starts the container, polls `/health` until ready, sends a `POST /extract` with the sample invoice, asserts `results` is non-empty and `summary.total_invoices >= 1`, then stops the container
- [ ] `docker compose down` cleans up without errors
- [ ] `README` or inline comments in the compose file explain how to activate the Ollama profile and where to put model weights
