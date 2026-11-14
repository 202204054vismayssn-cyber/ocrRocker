"""ASGI entry point used by FastAPI and Uvicorn."""

import sys
from pathlib import Path

# Ensure src/ is on python search path when running directly without pip install -e .
_src_dir = Path(__file__).resolve().parent / "src"
if _src_dir.is_dir() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from uae_invoice_reader.api import app

__all__ = ["app"]
