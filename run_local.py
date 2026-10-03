"""Convenient local runner for invoice_extraction_colab.py."""

import json
import sys
from pathlib import Path

from invoice_extraction_colab import run_batch


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".pdf"}


def collect_inputs(arguments):
    files = []
    for argument in arguments:
        path = Path(argument)
        if path.is_dir():
            files.extend(
                candidate
                for candidate in sorted(path.rglob("*"))
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        elif path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
        else:
            print(f"Skipping unsupported or missing path: {path}")
    return [str(path) for path in files]


def main():
    arguments = sys.argv[1:] or ["samples"]
    input_paths = collect_inputs(arguments)
    if not input_paths:
        raise SystemExit("No supported invoice images or PDFs were found.")

    results = run_batch(input_paths, export_xml=True)
    output_path = Path("output") / "batch_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Combined JSON saved to: {output_path}")


if __name__ == "__main__":
    main()

