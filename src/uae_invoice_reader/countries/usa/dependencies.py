"""Construct USA workflows from country-neutral infrastructure."""

from functools import lru_cache

from uae_invoice_reader.common.config import get_settings
from uae_invoice_reader.common.dependencies import (
    get_document_text_service,
    get_extraction_gate,
    get_local_batch_service,
    get_ollama_service,
)
from uae_invoice_reader.common.services.file_validation import FileValidationService
from uae_invoice_reader.common.services.text_compaction import TextCompactionService
from uae_invoice_reader.countries.usa.controllers.invoice_controller import InvoiceController


@lru_cache(maxsize=1)
def get_invoice_controller() -> InvoiceController:
    settings = get_settings()
    return InvoiceController(
        file_validation=FileValidationService(settings),
        document_text=get_document_text_service(),
        text_compaction=TextCompactionService(settings),
        ollama=get_ollama_service(),
        extraction_gate=get_extraction_gate(),
        local_batch=get_local_batch_service(),
    )
