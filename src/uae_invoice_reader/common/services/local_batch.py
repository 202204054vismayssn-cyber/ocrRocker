"""Local GPU batch extraction coordinator for Ollama and Gemma."""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from threading import Lock
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from uae_invoice_reader.common.dto.batch import (
    BatchItemResultDTO,
    BatchResultResponseDTO,
    BatchStatusResponseDTO,
    BatchSubmitItemDTO,
    BatchSubmitResponseDTO,
)
from uae_invoice_reader.common.exceptions import AppError
from uae_invoice_reader.common.services.extraction_gate import ExtractionGate
from uae_invoice_reader.common.services.ollama import OllamaService


class BatchJobRecord:
    """In-memory record tracking an active or completed local batch job."""

    def __init__(self, batch_job_name: str, total_items: int) -> None:
        self.batch_job_name = batch_job_name
        self.total_items = total_items
        self.completed_items = 0
        self.failed_items = 0
        self.state = "PROCESSING"
        self.created_at = datetime.now(UTC).isoformat()
        self.updated_at = self.created_at
        self.results: list[BatchItemResultDTO] = []
        self._lock = Lock()

    def record_item(self, result: BatchItemResultDTO) -> None:
        with self._lock:
            self.results.append(result)
            if result.success:
                self.completed_items += 1
            else:
                self.failed_items += 1
            self.updated_at = datetime.now(UTC).isoformat()

    def mark_completed(self) -> None:
        with self._lock:
            self.state = "COMPLETED"
            self.updated_at = datetime.now(UTC).isoformat()

    def mark_failed(self) -> None:
        with self._lock:
            self.state = "FAILED"
            self.updated_at = datetime.now(UTC).isoformat()

    def to_status_dto(self) -> BatchStatusResponseDTO:
        with self._lock:
            return BatchStatusResponseDTO(
                batch_job_name=self.batch_job_name,
                state=self.state,
                total_items=self.total_items,
                completed_items=self.completed_items,
                failed_items=self.failed_items,
                created_at=self.created_at,
                updated_at=self.updated_at,
            )

    def to_result_dto(self) -> BatchResultResponseDTO:
        with self._lock:
            return BatchResultResponseDTO(
                batch_job_name=self.batch_job_name,
                state=self.state,
                total_items=self.total_items,
                completed_items=self.completed_items,
                failed_items=self.failed_items,
                results=list(self.results),
            )


class LocalBatchService:
    """Manages asynchronous local batch jobs running on GPU with concurrency control."""

    def __init__(self, ollama: OllamaService, extraction_gate: ExtractionGate) -> None:
        self._ollama = ollama
        self._extraction_gate = extraction_gate
        self._jobs: dict[str, BatchJobRecord] = {}
        self._jobs_lock = Lock()

    def submit_batch(
        self,
        items: list[BatchSubmitItemDTO],
        system_prompt: str,
        response_schema: type[BaseModel],
        post_process_fn: Callable[[Any], None] | None = None,
    ) -> BatchSubmitResponseDTO:
        """Create and queue a local batch extraction job, returning immediately."""

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        batch_job_name = f"batches/loc_{timestamp}_{uuid4().hex[:8]}"

        job = BatchJobRecord(batch_job_name=batch_job_name, total_items=len(items))

        with self._jobs_lock:
            self._jobs[batch_job_name] = job
            # Normalize alternative query formats (e.g. without 'batches/' prefix)
            clean_name = batch_job_name.removeprefix("batches/")
            self._jobs[clean_name] = job

        # Launch the async worker task in the background
        asyncio.create_task(
            self._execute_batch(
                job=job,
                items=items,
                system_prompt=system_prompt,
                response_schema=response_schema,
                post_process_fn=post_process_fn,
            )
        )

        return BatchSubmitResponseDTO(
            batch_job_name=batch_job_name,
            total_items=len(items),
            state="PROCESSING",
        )

    def get_status(self, batch_job_name: str) -> BatchStatusResponseDTO:
        """Read the live progress of a batch job."""
        job = self._find_job(batch_job_name)
        return job.to_status_dto()

    def get_result(self, batch_job_name: str) -> BatchResultResponseDTO:
        """Retrieve full structured results for a completed batch job."""
        job = self._find_job(batch_job_name)
        return job.to_result_dto()

    def _find_job(self, batch_job_name: str) -> BatchJobRecord:
        normalized = batch_job_name.strip()
        with self._jobs_lock:
            job = self._jobs.get(normalized)
            if not job:
                # Try with or without 'batches/' prefix
                if normalized.startswith("batches/"):
                    job = self._jobs.get(normalized.removeprefix("batches/"))
                else:
                    job = self._jobs.get(f"batches/{normalized}")

        if not job:
            raise AppError(
                status_code=404,
                code="batch_job_not_found",
                detail=f"Batch job '{batch_job_name}' was not found.",
            )
        return job

    async def _execute_batch(
        self,
        job: BatchJobRecord,
        items: list[BatchSubmitItemDTO],
        system_prompt: str,
        response_schema: type[BaseModel],
        post_process_fn: Callable[[Any], None] | None,
    ) -> None:
        """Worker task processing batch items through the extraction gate slots."""

        async def process_item(item: BatchSubmitItemDTO) -> None:
            # Use extraction gate slot so single extractions and batch items share GPU concurrency
            async with self._extraction_gate.slot(
                request_id=job.batch_job_name,
                operation="batch_extraction_item",
            ):
                try:
                    model_instance, token_usage = await self._ollama.extract_structured(
                        system_prompt=system_prompt,
                        ocr_text=item.character_stored,
                        response_schema=response_schema,
                    )
                    if post_process_fn is not None:
                        post_process_fn(model_instance)

                    data_payload = model_instance.model_dump(mode="json")
                    item_result = BatchItemResultDTO(
                        custom_id=item.custom_id,
                        success=True,
                        data=data_payload,
                        token_usage=token_usage,
                    )
                except Exception as exc:
                    item_result = BatchItemResultDTO(
                        custom_id=item.custom_id,
                        success=False,
                        error=str(exc),
                    )

                job.record_item(item_result)

        try:
            tasks = [process_item(item) for item in items]
            await asyncio.gather(*tasks, return_exceptions=True)
            job.mark_completed()
        except Exception:
            job.mark_failed()
