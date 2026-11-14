"""Async semaphore concurrency gate for document extraction."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from time import monotonic
from uuid import uuid4

from uae_invoice_reader.common.config import Settings


class ExtractionGate:
    """Limits concurrent extractions using asyncio.Semaphore to protect GPU resources."""

    def __init__(self, settings: Settings) -> None:
        self._max_concurrent = settings.max_concurrent_extractions
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        self._active: dict[str, dict[str, object]] = {}

    @asynccontextmanager
    async def slot(
        self,
        request_id: str = "unknown",
        operation: str = "invoice_extraction",
    ) -> AsyncIterator[str]:
        """Reserve one extraction slot until the block finishes."""
        await self._semaphore.acquire()
        token = uuid4().hex[:8]
        self._active[token] = {
            "request_id": request_id,
            "operation": operation,
            "phase": "processing",
            "started_at": datetime.now(UTC),
            "started_monotonic": monotonic(),
        }
        try:
            yield token
        finally:
            self._active.pop(token, None)
            self._semaphore.release()

    async def update_phase(self, *args: object, **kwargs: object) -> None:
        """No-op stub for backwards compatibility."""

    async def status_snapshot(self) -> dict[str, object]:
        """Return point-in-time concurrency and processing state."""
        now = monotonic()
        active_list = [
            {
                "request_id": str(req["request_id"]),
                "operation": str(req["operation"]),
                "phase": str(req["phase"]),
                "started_at": req["started_at"],
                "elapsed_seconds": round(max(0.0, now - float(req["started_monotonic"])), 3),
            }
            for req in list(self._active.values())
        ]
        count = len(active_list)
        return {
            "is_processing": count > 0,
            "accepting_requests": True,
            "active_count": count,
            "max_concurrent": self._max_concurrent,
            "available_slots": max(0, self._max_concurrent - count),
            "active_requests": active_list,
        }

    async def begin_restart(self) -> None:
        """No-op stub for backwards compatibility."""

    async def cancel_restart(self) -> None:
        """No-op stub for backwards compatibility."""
