"""Read an upload safely and prove that its real content is supported."""

import io
import warnings

import pymupdf
from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from uae_invoice_reader.common.config import Settings
from uae_invoice_reader.common.exceptions import DocumentValidationError
from uae_invoice_reader.common.models.document import DocumentKind, ValidatedDocument

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SIGNATURE = b"\xff\xd8\xff"
_PDF_SIGNATURE = b"%PDF-"
_READ_CHUNK_SIZE = 1024 * 1024


class FileValidationService:
    """Reject unsafe, oversized, corrupt, encrypted, or fake uploads."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def validate(self, upload: UploadFile) -> ValidatedDocument:
        """Read and validate one upload before text extraction starts."""

        # Do not trust the filename or browser-provided content type. The actual
        # bytes decide whether this is a PDF, JPEG, or PNG.
        content = await self._read_limited(upload)
        kind = self._detect_kind(content)
        filename = upload.filename or "upload"

        if kind is DocumentKind.PDF:
            page_count = self._validate_pdf(content)
        else:
            self._validate_image(content, kind)
            page_count = 1

        return ValidatedDocument(
            original_filename=filename,
            kind=kind,
            content=content,
            total_pages=page_count,
        )

    async def _read_limited(self, upload: UploadFile) -> bytes:
        """Read in small chunks and stop as soon as the size limit is crossed."""

        chunks: list[bytes] = []
        size = 0
        while chunk := await upload.read(_READ_CHUNK_SIZE):
            size += len(chunk)
            if size > self.settings.max_upload_bytes:
                raise DocumentValidationError(
                    413,
                    "file_too_large",
                    f"Upload exceeds the {self.settings.max_upload_mb} MB limit",
                )
            chunks.append(chunk)
        if size == 0:
            raise DocumentValidationError(400, "empty_file", "The uploaded file is empty")
        return b"".join(chunks)

    @staticmethod
    def _detect_kind(content: bytes) -> DocumentKind:
        """Identify the file from its standard starting bytes (magic bytes)."""

        if content.startswith(_PDF_SIGNATURE):
            return DocumentKind.PDF
        if content.startswith(_PNG_SIGNATURE):
            return DocumentKind.PNG
        if content.startswith(_JPEG_SIGNATURE):
            return DocumentKind.JPEG
        raise DocumentValidationError(
            415,
            "unsupported_file_type",
            "Only genuine PDF, JPG, JPEG, and PNG files are supported",
        )

    def _validate_pdf(self, content: bytes) -> int:
        """Open the PDF and enforce encryption and page-count rules."""

        try:
            with pymupdf.open(stream=content, filetype="pdf") as document:
                if document.needs_pass:
                    raise DocumentValidationError(
                        400, "encrypted_pdf", "Password-protected PDFs are not supported"
                    )
                page_count = document.page_count
                if page_count < 1:
                    raise DocumentValidationError(400, "empty_pdf", "The PDF has no pages")
                if page_count > self.settings.max_pdf_pages:
                    raise DocumentValidationError(
                        413,
                        "too_many_pages",
                        f"PDF exceeds the {self.settings.max_pdf_pages}-page limit",
                    )
                return page_count
        except DocumentValidationError:
            raise
        except Exception as exc:
            raise DocumentValidationError(400, "invalid_pdf", "The PDF is corrupted") from exc

    def _validate_image(self, content: bytes, expected_kind: DocumentKind) -> None:
        """Decode the full image and enforce format and pixel-count rules."""

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(content)) as image:
                    actual_format = (image.format or "").upper()
                    expected_format = "JPEG" if expected_kind is DocumentKind.JPEG else "PNG"
                    if actual_format != expected_format:
                        raise DocumentValidationError(
                            415,
                            "spoofed_file_type",
                            "The decoded image format does not match its file signature",
                        )
                    width, height = image.size
                    if width * height > self.settings.max_image_pixels:
                        raise DocumentValidationError(
                            413,
                            "image_too_large",
                            "Image dimensions exceed the configured pixel limit",
                        )
                    image.verify()
        except DocumentValidationError:
            raise
        except (
            UnidentifiedImageError,
            OSError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as exc:
            raise DocumentValidationError(400, "invalid_image", "The image is corrupted") from exc

