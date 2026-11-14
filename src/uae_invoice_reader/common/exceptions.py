"""Shared errors that the API knows how to return safely.

Each error contains an HTTP status, a short machine-readable code, and a safe
message. Internal tracebacks and upstream model provider details are never
returned to the client.
"""


class AppError(Exception):
    """Base error for an expected failure that has a controlled API response."""

    def __init__(
        self,
        status_code: int,
        code: str,
        detail: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.code = code
        self.detail = detail
        self.headers = headers or {}


class ConfigurationError(AppError):
    """A required server setting is missing or invalid."""

    def __init__(self, detail: str) -> None:
        super().__init__(503, "configuration_error", detail)


class DocumentValidationError(AppError):
    """The upload is empty, unsafe, too large, corrupt, or unsupported."""

    pass


class DocumentExtractionError(AppError):
    """A valid upload did not contain enough readable invoice text."""

    def __init__(self, detail: str) -> None:
        super().__init__(422, "document_extraction_failed", detail)


class OcrUnavailableError(AppError):
    """A configured local OCR service could not be started."""

    def __init__(self, detail: str) -> None:
        super().__init__(503, "ocr_unavailable", detail)


class OllamaServiceError(AppError):
    """Ollama service failed, timed out, or returned an unparseable response."""

    def __init__(self, detail: str, status_code: int = 502) -> None:
        super().__init__(status_code, "ollama_service_error", detail)


