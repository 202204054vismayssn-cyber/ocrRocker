"""Keep a small sanitized history of recent API and provider failures."""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock


logger = logging.getLogger("uae_invoice_reader.errors")


@dataclass(frozen=True, slots=True)
class OperationalError:
    """Safe diagnostic fields that never contain invoice data or secrets."""

    occurred_at: datetime
    request_id: str
    source: str
    code: str
    http_status: int
    retryable: bool
    detail: str


class OperationalErrorLog:
    """Store recent errors in memory and emit the same safe fields to logs."""

    def __init__(self, max_entries: int = 100) -> None:
        self._entries: deque[OperationalError] = deque(maxlen=max_entries)
        self._lock = Lock()

    def record(
        self,
        *,
        request_id: str,
        source: str,
        code: str,
        http_status: int,
        retryable: bool,
        detail: str,
    ) -> None:
        entry = OperationalError(
            occurred_at=datetime.now(UTC),
            request_id=request_id,
            source=source,
            code=code,
            http_status=http_status,
            retryable=retryable,
            detail=detail,
        )
        with self._lock:
            self._entries.appendleft(entry)
        logger.warning(
            "request_failed request_id=%s source=%s code=%s http_status=%d retryable=%s",
            request_id,
            source,
            code,
            http_status,
            retryable,
        )

    def recent(self, limit: int = 20) -> list[OperationalError]:
        with self._lock:
            return list(self._entries)[:limit]
