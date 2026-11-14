"""Construct country-neutral operational dependencies."""

from functools import lru_cache

from uae_invoice_reader.common.config import get_settings
from uae_invoice_reader.common.controllers.system_controller import SystemController
from uae_invoice_reader.common.services.container_restart import ContainerRestartService
from uae_invoice_reader.common.services.document_text import DocumentTextService
from uae_invoice_reader.common.services.extraction_gate import ExtractionGate
from uae_invoice_reader.common.services.light_paddle_ocr import LightPaddleOcrService
from uae_invoice_reader.common.services.local_batch import LocalBatchService
from uae_invoice_reader.common.services.ollama import OllamaService
from uae_invoice_reader.common.services.operational_error_log import OperationalErrorLog


@lru_cache(maxsize=1)
def get_extraction_gate() -> ExtractionGate:
    return ExtractionGate(get_settings())


@lru_cache(maxsize=1)
def get_operational_error_log() -> OperationalErrorLog:
    return OperationalErrorLog(max_entries=100)


@lru_cache(maxsize=1)
def get_light_paddle_ocr_service() -> LightPaddleOcrService:
    """Load one PaddleOCR instance for all country extraction workflows."""
    return LightPaddleOcrService(get_settings())


@lru_cache(maxsize=1)
def get_document_text_service() -> DocumentTextService:
    """Document text service using Digital-First (PyMuPDF) + OCR Fallback (PaddleOCR)."""
    return DocumentTextService(get_settings(), get_light_paddle_ocr_service())


@lru_cache(maxsize=1)
def get_ollama_service() -> OllamaService:
    """Shared asynchronous Ollama service for Gemma structured inference."""
    return OllamaService(get_settings())


@lru_cache(maxsize=1)
def get_local_batch_service() -> LocalBatchService:
    """Shared local GPU batch extraction service."""
    return LocalBatchService(
        ollama=get_ollama_service(),
        extraction_gate=get_extraction_gate(),
    )


@lru_cache(maxsize=1)
def get_system_controller() -> SystemController:
    settings = get_settings()
    return SystemController(
        extraction_gate=get_extraction_gate(),
        container_restart=ContainerRestartService(settings),
        operational_error_log=get_operational_error_log(),
    )
