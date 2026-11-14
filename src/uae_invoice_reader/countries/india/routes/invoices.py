"""Authenticated HTTP routes for India invoice workflows with PaddleOCR and Ollama Gemma."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from uae_invoice_reader.common.dto.batch import (
    BatchPrepareResponseDTO,
    BatchResultResponseDTO,
    BatchStatusResponseDTO,
    BatchSubmitRequestDTO,
    BatchSubmitResponseDTO,
)
from uae_invoice_reader.common.security import require_savvy_api_key
from uae_invoice_reader.countries.india.controllers.invoice_controller import (
    InvoiceController,
)
from uae_invoice_reader.countries.india.dependencies import get_invoice_controller
from uae_invoice_reader.countries.india.dto.response import (
    CompletionExtractionResponseDTO,
    ExtractionResponseDTO,
)

router = APIRouter(prefix="/api/v1/india/invoices", tags=["India Invoices"])


@router.post(
    "/extract",
    response_model=ExtractionResponseDTO,
    dependencies=[Depends(require_savvy_api_key)],
    summary="Extract an India invoice",
)
async def extract_invoice(
    request: Request,
    file: Annotated[
        UploadFile,
        File(description="PDF, JPG, JPEG, or PNG invoice"),
    ],
    controller: InvoiceController = Depends(get_invoice_controller),
) -> ExtractionResponseDTO:
    """Accept one uploaded invoice, run local PaddleOCR, and extract structured India fields via Gemma."""

    return await controller.extract(file, request.state.request_id)


@router.post(
    "/complete",
    response_model=CompletionExtractionResponseDTO,
    dependencies=[Depends(require_savvy_api_key)],
    summary="Rescan an India invoice with PaddleOCR and Gemma with correlation ID",
)
async def complete_invoice(
    request: Request,
    file: Annotated[
        UploadFile,
        File(description="PDF, JPG, JPEG, or PNG invoice"),
    ],
    invoice_id: Annotated[
        str,
        Form(
            min_length=1,
            max_length=100,
            description="Caller invoice-row correlation ID",
        ),
    ],
    controller: InvoiceController = Depends(get_invoice_controller),
) -> CompletionExtractionResponseDTO:
    return await controller.complete(
        file,
        invoice_id,
        request.state.request_id,
    )


@router.post(
    "/batch/prepare",
    response_model=BatchPrepareResponseDTO,
    dependencies=[Depends(require_savvy_api_key)],
    summary="Prepare one invoice for batch processing using local PaddleOCR",
)
async def batch_prepare(
    request: Request,
    file: Annotated[UploadFile, File(description="PDF, JPG, JPEG, or PNG invoice")],
    controller: InvoiceController = Depends(get_invoice_controller),
) -> BatchPrepareResponseDTO:
    """Run local PaddleOCR and compaction without LLM to prepare text for batch submit."""
    return await controller.prepare_batch(file, request.state.request_id)


@router.post(
    "/batch/submit",
    response_model=BatchSubmitResponseDTO,
    dependencies=[Depends(require_savvy_api_key)],
    summary="Submit 1 to 100 prepared invoices for asynchronous GPU extraction",
)
async def batch_submit(
    payload: BatchSubmitRequestDTO,
    controller: InvoiceController = Depends(get_invoice_controller),
) -> BatchSubmitResponseDTO:
    """Queue prepared texts for parallel GPU processing with Ollama Gemma."""
    return controller.submit_batch(payload)


@router.get(
    "/batch/status/{batch_job_name:path}",
    response_model=BatchStatusResponseDTO,
    dependencies=[Depends(require_savvy_api_key)],
    summary="Poll status and progress of a batch job",
)
async def batch_status(
    batch_job_name: str,
    controller: InvoiceController = Depends(get_invoice_controller),
) -> BatchStatusResponseDTO:
    return controller.get_batch_status(batch_job_name)


@router.get(
    "/batch/result/{batch_job_name:path}",
    response_model=BatchResultResponseDTO,
    dependencies=[Depends(require_savvy_api_key)],
    summary="Retrieve final extraction results for a completed batch job",
)
async def batch_result(
    batch_job_name: str,
    controller: InvoiceController = Depends(get_invoice_controller),
) -> BatchResultResponseDTO:
    return controller.get_batch_result(batch_job_name)
