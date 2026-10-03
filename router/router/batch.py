"""Tunable defaults for the batch pipeline.

These live here so callers and tests have one place to read the pipeline's
tuning knobs.  The batch orchestration itself (``process_pdf``,
``run_pdf_batch``, ``collect_pdf_inputs``) stays in ``pdf_router.py`` because
that module is the one callers and tests mock-patch against.
"""

from __future__ import annotations

# How many required fields a native page may be missing before the router
# escalates it to OCR.  Continuation pages legitimately miss header fields, so
# the threshold is deliberately above zero.
DEFAULT_NATIVE_MISSING_THRESHOLD = 2

# Recorded on every result to tell consumers that pages were stitched back into
# logical invoices rather than returned one-per-page.
MULTI_PAGE_HANDLING = "grouped_by_invoice"
