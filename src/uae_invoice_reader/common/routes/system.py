"""Authenticated operational endpoints for status and safe restart."""

from fastapi import APIRouter, Depends, Request, status

from uae_invoice_reader.common.controllers.system_controller import SystemController
from uae_invoice_reader.common.dependencies import get_system_controller
from uae_invoice_reader.common.dto.system import ProcessingStatusResponseDTO, RestartResponseDTO
from uae_invoice_reader.common.security import require_admin_api_key, require_savvy_api_key

router = APIRouter(prefix="/api/v1/system", tags=["System"])


@router.get(
    "/processing-status",
    response_model=ProcessingStatusResponseDTO,
    dependencies=[Depends(require_savvy_api_key)],
    summary="Read live invoice-processing status",
)
async def processing_status(
    request: Request,
    controller: SystemController = Depends(get_system_controller),
) -> ProcessingStatusResponseDTO:
    return await controller.processing_status(request.state.request_id)


@router.post(
    "/restart",
    response_model=RestartResponseDTO,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_admin_api_key)],
    summary="Safely restart an idle API container",
)
async def restart_container(
    request: Request,
    controller: SystemController = Depends(get_system_controller),
) -> RestartResponseDTO:
    return await controller.restart(request.state.request_id)

