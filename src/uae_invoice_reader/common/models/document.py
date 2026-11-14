"""Country-neutral objects used while a document moves through the app."""

from dataclasses import dataclass
from enum import StrEnum


class DocumentKind(StrEnum):
    """File formats accepted by the API after content validation."""

    PDF = "pdf"
    JPEG = "jpeg"
    PNG = "png"


class ExtractionMethod(StrEnum):
    """How the final text was obtained."""

    PYMUPDF = "pymupdf"
    PADDLEOCR = "paddleocr"


@dataclass(frozen=True, slots=True)
class ValidatedDocument:
    """An upload that passed safety and format checks."""

    original_filename: str
    kind: DocumentKind
    content: bytes
    total_pages: int


@dataclass(frozen=True, slots=True)
class ExtractedDocumentText:
    """Locally extracted text and facts about the extraction process."""

    text: str
    total_pages: int
    character_count: int
    extraction_method: ExtractionMethod
    used_ocr: bool
