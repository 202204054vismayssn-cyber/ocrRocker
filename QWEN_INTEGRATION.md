# Qwen semantic fallback integration

The fallback is deliberately independent from PDF detection and OCR. It accepts
the common canonical result produced after both extraction paths converge.

## Programmatic use

```python
from invoice_extraction_colab import CANONICAL_SCHEMA
from qwen_semantic_fallback import QwenFallbackConfig, QwenSemanticFallback
from pdf_router import run_pdf_batch

config = QwenFallbackConfig(
    model="qwen2.5vl:3b",
    base_url="http://127.0.0.1:11434",
    timeout_seconds=180,
    image_escalation=True,
    max_images=3,
)

semantic_fallback = QwenSemanticFallback(
    CANONICAL_SCHEMA,
    config=config,
)

results = run_pdf_batch(
    ["invoice.pdf"],
    output_dir="pdf_router_output",
    semantic_fallback=semantic_fallback,
)
```

## Contract

`QwenSemanticFallback.apply()` receives:

- the canonical result containing `fields`, `missing_required`, and
  `field_sources`;
- normalized text from pdfplumber or PaddleOCR;
- optional rendered image paths, present only for OCR-routed pages.

It returns the same result shape. Newly recovered values are marked as either
`qwen_text_fallback` or `qwen_vision_fallback` in `field_sources`.

The module never overwrites a present value and returns the deterministic result
unchanged when no required field is missing. Network/server/model failures are
recorded in `semantic_fallback.errors` instead of terminating the invoice batch.

## Direct use after another parser

```python
recovered = semantic_fallback.apply(
    canonical_result,
    raw_text=extracted_text,
    image_paths=scanned_page_images,  # use [] for text-native PDFs
)
```

Do not pass rendered images for a text-native PDF. For scanned documents, the
class always attempts text first and uses images only if required fields remain.
