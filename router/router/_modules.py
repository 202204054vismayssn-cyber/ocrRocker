"""Shared loaders for the sibling modules the router package depends on.

``invoice_extraction_colab.py``, ``native_invoice_parser.py`` and
``qwen_semantic_fallback.py`` live alongside the router package (one directory
up).  Loading them by file path avoids any name collision with same-named
installed packages.

Every router sub-module must import ``ocr`` / ``native_parser`` /
``qwen_fallback`` from here.  They are loaded once, so patching
``pdf_router.ocr.<name>`` is visible from every sub-module.  Loading them a
second time elsewhere would create a distinct module object and silently break
those patches.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# The router package lives at  <project>/router/
# Sibling modules live at      <project>/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load(filename: str, module_name: str):
    """Load a .py file from the project root by path, avoiding package name collisions."""
    module_path = _PROJECT_ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load required module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Loaded once at import time; all router sub-modules share the same objects.
ocr = _load("invoice_extraction_colab.py", "invoice_extraction_colab")
native_parser = _load("native_invoice_parser.py", "native_invoice_parser")
qwen_fallback = _load("qwen_semantic_fallback.py", "qwen_semantic_fallback")
