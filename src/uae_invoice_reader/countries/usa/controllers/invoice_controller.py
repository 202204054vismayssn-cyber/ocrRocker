"""USA invoice extraction controller using PaddleOCR and Ollama Gemma."""

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
from uae_invoice_reader.countries.usa.dto.response import (
    CompletionExtractionResponseDTO,
    ExtractionMetadataDTO,
    ExtractionResponseDTO,
    TokenUsageDTO,
    UsaInvoiceDTO,
)
from uae_invoice_reader.countries.usa.services.prompt import USA_SYSTEM_PROMPT


class InvoiceController:
    """Orchestrate file validation, PaddleOCR, Ollama Gemma inference for USA invoices."""

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
            request_id=request_id, operation="usa_batch_prepare"
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
            system_prompt=USA_SYSTEM_PROMPT,
            response_schema=UsaInvoiceDTO,
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
            request_id=request_id, operation="usa_invoice_extraction"
        ):
            document = await self.file_validation.validate(upload)
            extracted: ExtractedDocumentText = await to_thread.run_sync(
                self.document_text.extract, document
            )
            compacted = self.text_compaction.compact(extracted.text)

            invoice_dto, usage_info = await self.ollama.extract_structured(
                system_prompt=USA_SYSTEM_PROMPT,
                ocr_text=compacted.text,
                response_schema=UsaInvoiceDTO,
            )

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
