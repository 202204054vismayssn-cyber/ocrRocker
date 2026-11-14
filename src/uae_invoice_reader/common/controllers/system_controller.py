"""Coordinate live processing status and guarded container restarts."""

from uae_invoice_reader.common.dto.system import (
    ProcessingStatusResponseDTO,
    RestartResponseDTO,
)
from uae_invoice_reader.common.services.container_restart import ContainerRestartService
from uae_invoice_reader.common.services.extraction_gate import ExtractionGate
from uae_invoice_reader.common.services.operational_error_log import OperationalErrorLog


class SystemController:
    """Expose operational state without exposing invoice contents."""

    def __init__(
        self,
        extraction_gate: ExtractionGate,
        container_restart: ContainerRestartService,
        operational_error_log: OperationalErrorLog,
    ) -> None:
        self.extraction_gate = extraction_gate
        self.container_restart = container_restart
        self.operational_error_log = operational_error_log

    async def processing_status(self, request_id: str) -> ProcessingStatusResponseDTO:
        snapshot = await self.extraction_gate.status_snapshot()
        errors = self.operational_error_log.recent(limit=20)
        return ProcessingStatusResponseDTO(
            request_id=request_id,
            **snapshot,
            recent_error_count=len(errors),
            recent_errors=errors,
        )

    async def restart(self, request_id: str) -> RestartResponseDTO:
        delay = await self.container_restart.schedule(self.extraction_gate)
        return RestartResponseDTO(
            request_id=request_id,
            restart_in_seconds=delay,
        )

