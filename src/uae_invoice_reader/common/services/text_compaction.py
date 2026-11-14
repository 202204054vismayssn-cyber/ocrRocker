"""Reduce unnecessary text spacing without damaging document structure."""

import re
import unicodedata
from dataclasses import dataclass

from uae_invoice_reader.common.config import Settings
from uae_invoice_reader.common.exceptions import DocumentExtractionError


@dataclass(frozen=True, slots=True)
class CompactedText:
    """Cleaned invoice text plus its approximate token count."""

    text: str
    estimated_tokens: int


class TextCompactionService:
    """Prepare local extracted text before sending to LLM extraction."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @staticmethod
    def clean_for_output(text: str) -> str:
        """Remove blank lines and repeated spaces while preserving row order."""

        # Unicode NFKC makes equivalent characters use a consistent form. Line
        # endings also differ between Windows and Linux, so normalize them here.
        normalized = (
            unicodedata.normalize("NFKC", text)
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )
        cleaned_lines: list[str] = []

        # Keep one line break between meaningful rows (important for tables),
        # but remove every empty line to reduce stored characters and model
        # input tokens.
        for raw_line in normalized.split("\n"):
            cleaned_line = re.sub(r"[\t ]+", " ", raw_line).strip()
            if not cleaned_line:
                continue
            cleaned_lines.append(cleaned_line)

        cleaned_text = "\n".join(cleaned_lines)
        if len(cleaned_text) < 20:
            raise DocumentExtractionError("Extracted document text is insufficient")
        return cleaned_text

    def compact(self, text: str) -> CompactedText:
        """Normalize spacing, preserve lines, and enforce the input limit."""

        compacted_text = self.clean_for_output(text)

        # Four characters per token is a conservative local estimate.
        estimated_tokens = max(1, (len(compacted_text) + 3) // 4)
        if estimated_tokens > self.settings.max_input_tokens:
            raise DocumentExtractionError(
                "Compacted invoice text exceeds the configured input-token ceiling"
            )
        return CompactedText(compacted_text, estimated_tokens)

