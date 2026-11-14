"""Add a safe request ID and processing time to every HTTP response."""

import logging
import re
from time import perf_counter
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
logger = logging.getLogger("uae_invoice_reader.requests")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Run before and after every route without changing route code."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        requested_id = request.headers.get("X-Request-ID", "")
        request_id = (
            requested_id if _SAFE_REQUEST_ID.fullmatch(requested_id) else str(uuid4())
        )
        request.state.request_id = request_id

        request_started_at = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            processing_time_ms = round((perf_counter() - request_started_at) * 1000)
            logger.error(
                "request_aborted request_id=%s method=%s path=%s duration_ms=%d",
                request_id,
                request.method,
                request.url.path,
                processing_time_ms,
            )
            raise

        response.headers["X-Request-ID"] = request_id
        processing_time_ms = round((perf_counter() - request_started_at) * 1000)
        response.headers["X-Process-Time-Ms"] = str(processing_time_ms)
        if request.url.path != "/health":
            logger.info(
                "request_completed request_id=%s method=%s path=%s status=%d duration_ms=%d",
                request_id,
                request.method,
                request.url.path,
                response.status_code,
                processing_time_ms,
            )
        return response
