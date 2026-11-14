"""Create the FastAPI application and connect global HTTP behavior."""

import logging
from uuid import uuid4

logger = logging.getLogger("uae_invoice_reader.api")

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from uae_invoice_reader.common.config import get_settings
from uae_invoice_reader.common.dependencies import get_operational_error_log
from uae_invoice_reader.common.dto.errors import ErrorBodyDTO, ErrorResponseDTO
from uae_invoice_reader.common.exceptions import AppError
from uae_invoice_reader.common.logging_config import configure_console_logging
from uae_invoice_reader.common.middleware import RequestContextMiddleware
from uae_invoice_reader.common.routes.system import router as system_router
from uae_invoice_reader.common.services.error_classification import error_origin
from uae_invoice_reader.countries.india.routes.invoices import (
    router as india_invoice_router,
)
from uae_invoice_reader.countries.uae.routes.invoices import router as invoice_router
from uae_invoice_reader.countries.usa.routes.invoices import router as usa_invoice_router


def _request_id(request: Request) -> str:
    """Read the middleware request ID, or create one as a safety fallback."""
    return getattr(request.state, "request_id", str(uuid4()))


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    detail: str,
    source: str,
    retryable: bool,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Build the same safe JSON shape for every API error."""
    payload = ErrorResponseDTO(
        request_id=_request_id(request),
        error=ErrorBodyDTO(
            code=code,
            detail=detail,
            source=source,
            retryable=retryable,
        ),
    )
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json"),
        headers=headers,
    )


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()
    configure_console_logging(settings)
    application = FastAPI(
        title="Invoice Reader API",
        version="0.2.0",
        description="High-throughput invoice extraction using PaddleOCR and Ollama Gemma structured JSON",
    )
    application.add_middleware(RequestContextMiddleware)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=[
            "Content-Type",
            "SAVVY_API_KEY",
            "ADMIN_API_KEY",
            "X-Request-ID",
        ],
    )

    @application.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        source, retryable = error_origin(exc.code, exc.status_code)
        get_operational_error_log().record(
            request_id=_request_id(request),
            source=source,
            code=exc.code,
            http_status=exc.status_code,
            retryable=retryable,
            detail=exc.detail,
        )
        return _error_response(
            request,
            exc.status_code,
            exc.code,
            exc.detail,
            source,
            retryable,
            exc.headers,
        )

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        source, retryable = error_origin("invalid_request", 400)
        error_msgs = [f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in exc.errors()]
        detail_msg = "; ".join(error_msgs) if error_msgs else "The request is invalid"
        get_operational_error_log().record(
            request_id=_request_id(request),
            source=source,
            code="invalid_request",
            http_status=400,
            retryable=retryable,
            detail=detail_msg,
        )
        return _error_response(
            request,
            400,
            "invalid_request",
            detail_msg,
            source,
            retryable,
        )

    @application.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unexpected error on %s %s: %s", request.method, request.url, exc)
        error_detail = f"An unexpected error occurred: {exc}"
        get_operational_error_log().record(
            request_id=_request_id(request),
            source="api_server",
            code="internal_error",
            http_status=500,
            retryable=False,
            detail=error_detail,
        )
        return _error_response(
            request,
            500,
            "internal_error",
            error_detail,
            "api_server",
            False,
        )

    @application.get("/", tags=["System"])
    async def root() -> dict[str, str]:
        return {"message": "Invoice Reader API (PaddleOCR + Ollama Gemma)"}

    @application.get("/health", tags=["System"])
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "model": settings.ollama_model,
            "provider": "ollama",
            "device": settings.paddle_device,
        }

    application.include_router(invoice_router)
    application.include_router(usa_invoice_router)
    application.include_router(india_invoice_router)
    application.include_router(system_router)
    return application


app = create_app()
