"""Classify errors into clear source origins and retryability."""


def error_origin(code: str, status_code: int) -> tuple[str, bool]:
    """Return the error source (client, ocr, ollama, server) and whether it is retryable."""

    if code.startswith("ollama_"):
        source = "ollama"
    elif code.startswith("ocr_") or code.startswith("document_extraction"):
        source = "ocr"
    elif 400 <= status_code < 500:
        source = "client"
    else:
        source = "server"

    retryable = status_code in {429, 502, 503, 504}
    return source, retryable
