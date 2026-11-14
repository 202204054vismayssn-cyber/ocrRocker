"""Extract text from PDFs and images using PaddleOCR exclusively."""

import io
from typing import Protocol

import numpy as np
import pymupdf
from PIL import Image

from uae_invoice_reader.common.config import Settings
from uae_invoice_reader.common.exceptions import DocumentExtractionError
from uae_invoice_reader.common.models.document import (
    DocumentKind,
    ExtractedDocumentText,
    ExtractionMethod,
    ValidatedDocument,
)


class OcrTextExtractor(Protocol):
    """Protocol satisfied by LightPaddleOcrService."""

    def extract(self, image: np.ndarray) -> str: ...


class DocumentTextService:
    """Uses PaddleOCR exclusively for all documents (PDFs rendered to images and images directly)."""

    PDF_RENDER_SCALE = 2.0

    def __init__(self, settings: Settings, paddle: OcrTextExtractor) -> None:
        self.settings = settings
        self.paddle = paddle

    def extract(self, document: ValidatedDocument) -> ExtractedDocumentText:
        """OCR every page of PDF or image using local PaddleOCR."""

        if document.kind is DocumentKind.PDF:
            return self._extract_pdf(document)
        return self._extract_image(document)

    def _extract_pdf(self, document: ValidatedDocument) -> ExtractedDocumentText:
        # Step 1: Digital First (PyMuPDF instantaneous extraction for digital PDFs)
        digital_pages: list[str] = []
        try:
            with pymupdf.open(stream=document.content, filetype="pdf") as pdf:
                for page_number, page in enumerate(pdf, start=1):
                    page_text = page.get_text().strip()
                    if page_text:
                        digital_pages.append(f"--- PAGE {page_number} ---\n{page_text}")
                
                combined_digital = "\n\n".join(digital_pages).strip()
                if len(combined_digital) >= 40:
                    return ExtractedDocumentText(
                        text=combined_digital,
                        total_pages=document.total_pages,
                        character_count=len(combined_digital),
                        extraction_method=ExtractionMethod.PYMUPDF,
                        used_ocr=False,
                    )
        except Exception:
            pass

        # Step 2: OCR Fallback (PaddleOCR on GPU for scanned PDFs/photos)
        return self._extract_pdf_ocr(document)

    def _extract_pdf_ocr(self, document: ValidatedDocument) -> ExtractedDocumentText:
        pages: list[str] = []
        try:
            with pymupdf.open(stream=document.content, filetype="pdf") as pdf:
                for page_number, page in enumerate(pdf, start=1):
                    pixmap = page.get_pixmap(
                        matrix=pymupdf.Matrix(self.PDF_RENDER_SCALE, self.PDF_RENDER_SCALE),
                        alpha=False,
                    )
                    with Image.open(io.BytesIO(pixmap.tobytes("png"))) as source:
                        image = np.asarray(source.convert("RGB"))
                    page_text = self.paddle.extract(image).strip()
                    pages.append(f"--- PAGE {page_number} ---\n{page_text}")
        except DocumentExtractionError:
            raise
        except Exception as exc:
            raise DocumentExtractionError(
                f"Failed to render and extract PDF text with PaddleOCR: {exc}"
            ) from exc

        text = "\n\n".join(pages).strip()
        if not text:
            raise DocumentExtractionError("PaddleOCR found no readable text in the PDF document")

        return ExtractedDocumentText(
            text=text,
            total_pages=document.total_pages,
            character_count=len(text),
            extraction_method=ExtractionMethod.PADDLEOCR,
            used_ocr=True,
        )

    def _extract_image(self, document: ValidatedDocument) -> ExtractedDocumentText:
        try:
            with Image.open(io.BytesIO(document.content)) as source:
                image = np.asarray(source.convert("RGB"))
            extracted = self.paddle.extract(image).strip()
        except DocumentExtractionError:
            raise
        except Exception as exc:
            raise DocumentExtractionError(
                "Failed to extract image text with PaddleOCR"
            ) from exc

        text = f"--- PAGE 1 ---\n{extracted}"
        return ExtractedDocumentText(
            text=text,
            total_pages=1,
            character_count=len(text),
            extraction_method=ExtractionMethod.PADDLEOCR,
            used_ocr=True,
        )
