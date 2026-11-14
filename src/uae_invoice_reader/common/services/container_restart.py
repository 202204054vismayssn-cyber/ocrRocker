"""Schedule a container restart if running inside Docker."""

import asyncio
import os
import signal
from pathlib import Path

from uae_invoice_reader.common.config import Settings
from uae_invoice_reader.common.exceptions import AppError
from uae_invoice_reader.common.services.extraction_gate import ExtractionGate


class ContainerRestartService:
    """Safely terminate container process after returning HTTP 202."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def schedule(self, gate: ExtractionGate | None = None) -> float:
        if not self.settings.enable_self_restart:
            raise AppError(503, "self_restart_disabled", "Container restart is disabled by configuration")
        if not Path("/.dockerenv").exists():
            raise AppError(503, "not_running_in_docker", "Container restart is available only when running in Docker")

        async def _terminate() -> None:
            await asyncio.sleep(self.settings.restart_delay_seconds)
            os.kill(1, signal.SIGTERM)

        asyncio.create_task(_terminate())
        return self.settings.restart_delay_seconds
