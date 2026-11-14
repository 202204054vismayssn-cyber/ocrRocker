"""UAE Invoice Reader application package."""


def main() -> None:
    """Run the development server through the project script entry point."""
    import uvicorn

    uvicorn.run("uae_invoice_reader.api:app", host="0.0.0.0", port=8000, workers=1)


__all__ = ["main"]
