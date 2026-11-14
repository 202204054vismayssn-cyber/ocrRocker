"""India invoice extraction controller using PaddleOCR and Ollama Gemma."""

from anyio import to_thread
from fastapi import UploadFile

from uae_invoice_reader.common.dto.batch import (
    BatchPrepareResponseDTO,
    BatchResultResponseDTO,
    BatchStatusResponseDTO,
    BatchSubmitRequestDTO,
    BatchSubmitResponseDTO,
)
from uae_invoice_reader.common.exceptions import AppError
from uae_invoice_reader.common.models.document import ExtractedDocumentText
from uae_invoice_reader.common.services.document_text import DocumentTextService
from uae_invoice_reader.common.services.extraction_gate import ExtractionGate
from uae_invoice_reader.common.services.file_validation import FileValidationService
from uae_invoice_reader.common.services.local_batch import LocalBatchService
from uae_invoice_reader.common.services.ollama import OllamaService
from uae_invoice_reader.common.services.text_compaction import TextCompactionService
from uae_invoice_reader.countries.india.dto.response import (
    CompletionExtractionResponseDTO,
    ExtractionMetadataDTO,
    ExtractionResponseDTO,
    IndiaInvoiceDTO,
    TokenUsageDTO,
)
from uae_invoice_reader.countries.india.services.prompt import INDIA_SYSTEM_PROMPT


class InvoiceController:
    """Orchestrate file validation, PaddleOCR, Ollama Gemma inference for Indian invoices."""

    def __init__(
        self,
        file_validation: FileValidationService,
        document_text: DocumentTextService,
        text_compaction: TextCompactionService,
        ollama: OllamaService,
        extraction_gate: ExtractionGate,
        local_batch: LocalBatchService,
    ) -> None:
        self.file_validation = file_validation
        self.document_text = document_text
        self.text_compaction = text_compaction
        self.ollama = ollama
        self.extraction_gate = extraction_gate
        self.local_batch = local_batch

    async def prepare_batch(
        self,
        upload: UploadFile,
        request_id: str,
    ) -> BatchPrepareResponseDTO:
        """Run local PaddleOCR and compaction without LLM to prepare text for batch submit."""

        async with self.extraction_gate.slot(
            request_id=request_id, operation="india_batch_prepare"
        ):
            document = await self.file_validation.validate(upload)
            extracted: ExtractedDocumentText = await to_thread.run_sync(
                self.document_text.extract, document
            )
            compacted = self.text_compaction.compact(extracted.text)

            return BatchPrepareResponseDTO(
                request_id=request_id,
                character_stored=compacted.text,
                total_pages=extracted.total_pages,
                character_count=len(compacted.text),
            )

    def submit_batch(
        self,
        payload: BatchSubmitRequestDTO,
    ) -> BatchSubmitResponseDTO:
        """Queue 1 to 100 prepared invoices for asynchronous GPU extraction."""

        return self.local_batch.submit_batch(
            items=payload.items,
            system_prompt=INDIA_SYSTEM_PROMPT,
            response_schema=IndiaInvoiceDTO,
            post_process_fn=self._post_process_india_invoice,
        )

    def get_batch_status(self, batch_job_name: str) -> BatchStatusResponseDTO:
        """Check live progress of a queued batch job."""
        return self.local_batch.get_status(batch_job_name)

    def get_batch_result(self, batch_job_name: str) -> BatchResultResponseDTO:
        """Fetch final structured JSON results for a completed batch job."""
        return self.local_batch.get_result(batch_job_name)

    async def extract(
        self,
        upload: UploadFile,
        request_id: str,
    ) -> ExtractionResponseDTO:
        async with self.extraction_gate.slot(
            request_id=request_id, operation="india_invoice_extraction"
        ):
            document = await self.file_validation.validate(upload)
            extracted: ExtractedDocumentText = await to_thread.run_sync(
                self.document_text.extract, document
            )
            compacted = self.text_compaction.compact(extracted.text)

            invoice_dto, usage_info = await self.ollama.extract_structured(
                system_prompt=INDIA_SYSTEM_PROMPT,
                ocr_text=compacted.text,
                response_schema=IndiaInvoiceDTO,
            )

            # Deterministic post-processing for Indian invoices
            self._post_process_india_invoice(invoice_dto)

            return ExtractionResponseDTO(
                request_id=request_id,
                extracted_text=compacted.text,
                data=invoice_dto,
                token_usage=TokenUsageDTO(
                    input_tokens=usage_info.get("input_tokens", 0),
                    output_tokens=usage_info.get("output_tokens", 0),
                    total_tokens=usage_info.get("total_tokens", 0),
                    execution_time_ms=usage_info.get("execution_time_ms"),
                ),
                metadata=ExtractionMetadataDTO(
                    total_pages=extracted.total_pages,
                    character_count=len(compacted.text),
                    finish_reason="stop",
                    model_name=usage_info.get("model", "gemma"),
                    model_provider="ollama",
                ),
            )

    async def complete(
        self,
        upload: UploadFile,
        invoice_id: str,
        request_id: str,
    ) -> CompletionExtractionResponseDTO:
        normalized_id = invoice_id.strip()
        if not normalized_id or len(normalized_id) > 100:
            raise AppError(
                400, "invalid_invoice_id", "invoice_id must contain between 1 and 100 characters"
            )

        response = await self.extract(upload, request_id)
        return CompletionExtractionResponseDTO(
            **response.model_dump(mode="python", by_alias=False),
            invoice_id=normalized_id,
        )

    @staticmethod
    def _post_process_india_invoice(dto: IndiaInvoiceDTO) -> None:
        """Assign item serial numbers, derive PANs, and reconcile totals when absent."""

        # 1. Deterministically assign item serial numbers
        for idx, item in enumerate(dto.items or [], start=1):
            if not item.sr_no:
                item.sr_no = str(idx)

        # 2. Derive PANs from GSTINs if missing (characters 3-12)
        if not dto.customer_pan and dto.customer_gst and len(dto.customer_gst.strip()) == 15:
            dto.customer_pan = dto.customer_gst.strip()[2:12].upper()
        if not dto.vendor_pan and dto.vendor_gst and len(dto.vendor_gst.strip()) == 15:
            dto.vendor_pan = dto.vendor_gst.strip()[2:12].upper()

        # 3. Date fallback: If invoice_date is missing but due_date was captured from invoice date
        if not dto.invoice_date and dto.due_date:
            dto.invoice_date = dto.due_date

        # 4. Total reconciliations from line items or HSN summary if LLM missed top-level amounts
        from decimal import Decimal

        if dto.taxable_amount is None:
            item_sum = sum((item.amount for item in dto.items if item.amount is not None), Decimal(0))
            if item_sum > Decimal(0):
                dto.taxable_amount = item_sum

        if dto.cgst_amount is None and dto.hsn_summary:
            cgst_sum = sum((h.cgst for h in dto.hsn_summary if h.cgst is not None), Decimal(0))
            if cgst_sum > Decimal(0):
                dto.cgst_amount = cgst_sum

        if dto.sgst_amount is None and dto.hsn_summary:
            sgst_sum = sum((h.sgst for h in dto.hsn_summary if h.sgst is not None), Decimal(0))
            if sgst_sum > Decimal(0):
                dto.sgst_amount = sgst_sum

        if dto.invoice_amount is None and dto.taxable_amount is not None:
            taxes = (
                (dto.cgst_amount or Decimal(0))
                + (dto.sgst_amount or Decimal(0))
                + (dto.igst_amount or Decimal(0))
            )
            if taxes > Decimal(0):
                dto.invoice_amount = dto.taxable_amount + taxes

        # 5. Numeric zero defaults for specific missing fields
        if dto.igst_amount is None:
            dto.igst_amount = 0  # type: ignore[assignment]
        if dto.discount is None:
            dto.discount = 0  # type: ignore[assignment]
        if dto.state_cess is None:
            dto.state_cess = 0  # type: ignore[assignment]
        if dto.other_charges is None:
            dto.other_charges = 0  # type: ignore[assignment]
        if not dto.irn_number:
            dto.irn_number = 0
