"""Configure safe operational logging through the container console."""

import logging
import sys
from typing import TextIO

from uae_invoice_reader.common.config import Settings

_LOGGER_NAME = "uae_invoice_reader"


def configure_console_logging(
    settings: Settings,
    stream: TextIO | None = None,
) -> None:
    """Write application events to stdout without touching the filesystem."""

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(settings.log_level)
    logger.propagate = False

    # ``create_app`` can run more than once in tests. Reuse the handler instead
    # of writing every log entry multiple times.
    for handler in logger.handlers:
        if getattr(handler, "_uae_console_handler", False):
            return

    handler = logging.StreamHandler(stream or sys.stdout)
    handler._uae_console_handler = True  # type: ignore[attr-defined]
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    logger.addHandler(handler)
