# 01: Refactor `pdf_router.py` into `router/` package

**What to build:** Split the single 900-line `pdf_router.py` into four focused modules under a new `router/` package, each with a single clear responsibility. The top-level `pdf_router.py` becomes a thin re-export shim so all existing callers — the CLI, the tests, and `presentation_ocr.py` — continue to work without any changes. The router behaviour itself is unchanged; this is a pure structural refactor.

The four modules:
- `router/page_detector.py` — decides whether a page is native text or scanned, renders pages to images, manages the OCR pipeline lifecycle
- `router/page_extractor.py` — extracts one page via either the native or scanned path, attaches router metadata
- `router/invoice_grouper.py` — groups per-page results into logical invoice groups, merges multi-page invoices into one canonical record
- `router/batch.py` — orchestrates multi-PDF batches, collects inputs from files/folders/ZIPs, writes output files

Every public function gets a one-line docstring if it is missing one. Inline comments added on non-obvious logic blocks.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] `router/` package exists with `__init__.py`, four module files, and all functions moved into the correct module
- [ ] `pdf_router.py` re-exports every public name so existing imports still resolve
- [ ] `python -m unittest discover -s tests -v` passes with zero failures
- [ ] `python pdf_router.py --help` still works (CLI entry point intact)
- [ ] Every public function in the four new modules has a docstring
