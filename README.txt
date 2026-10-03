INVOICE OCR PDF ROUTER V3 + LOCAL QWEN FALLBACK
=========================

This Colab/local package routes each PDF page through the correct path:

- native text page -> PyMuPDF detection + pdfplumber extraction (no OCR/GPU)
- scanned/image page -> 300-DPI render + the existing PaddleOCR pipeline

It then groups pages by logical invoice and produces the same canonical JSON
schema regardless of the extraction path.


WHAT V2 FIXES
-------------

- One multi-page invoice produces one JSON result, not one result per page.
- A new invoice number starts a new result inside a combined/batch PDF.
- A page with an item table but no invoice header is a table continuation.
- IRN/Ack pages stay attached as supporting pages.
- Bill To is extracted from its own table column; Ship To is never merged.
- Line items are merged across pages and de-duplicated.
- Subtotal, CGST, SGST/UTGST, IGST and Total come from the terminal table
  summary. A tax component that is not applicable is output as 0.0.
- Payment Made and Balance Due remain separate from the gross invoice Total.
- Native continuation/support pages do not waste GPU memory on OCR merely
  because they omit invoice header fields.
- One uploaded ZIP containing many PDFs is accepted directly.
- One PaddleOCR model instance is reused sequentially for all scanned pages.

The deterministic parser remains the primary extractor. An optional local
Qwen2.5-VL-3B/Ollama layer runs only when required fields remain missing. It
never overwrites a value already found by the rule-based parser.


PACKAGE CONTENTS
----------------

invoice_extraction_colab.py   Existing tested OCR/parser pipeline (unchanged)
native_invoice_parser.py      Native layout, Bill To, item and total helpers
pdf_router.py                 Main PDF/ZIP batch entry point
qwen_semantic_fallback.py     Optional text-first, vision-second Ollama module
presentation_ocr.py           Minimal fields + missing_required output
run_local.py                  Existing image/local runner
requirements.txt              Python dependencies
samples/                      Canara Bank image samples
tests/                        Fast CPU-only regression tests


GOOGLE COLAB: CLEAN SETUP
-------------------------

Use a fresh Colab runtime and select Runtime > Change runtime type > T4 GPU.
Upload this ZIP and extract it so this path exists:

  /content/invoice_ocr_qwen_ollama_v3/

Run these cells in order.

Cell 1 - remove conflicting Paddle/PyTorch builds:

  !python -m pip uninstall -y paddlepaddle paddlepaddle-gpu torch torchvision torchaudio

Cell 2 - install Paddle GPU and CPU-only PyTorch:

  !python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
  !python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu

CPU-only PyTorch is deliberate. Paddle owns the GPU in this package, avoiding
the incompatible NCCL/cuDNN packages that previously broke import torch.

Cell 3 - install the router dependencies:

  !python -m pip install -r "/content/invoice_ocr_qwen_ollama_v3/requirements.txt"

Cell 4 - verify the runtime:

  import torch
  import paddle
  import paddleocr
  import pymupdf
  import pdfplumber
  from paddleocr import PaddleOCR

  print("Torch:", torch.__version__, "Torch CUDA:", torch.cuda.is_available())
  print("Paddle:", paddle.__version__)
  print("Paddle CUDA:", paddle.device.is_compiled_with_cuda())
  print("Paddle device:", paddle.device.get_device())
  print("PaddleOCR imported successfully")

The expected healthy state is:

  Torch CUDA: False
  Paddle CUDA: True
  Paddle device: gpu:0
  PaddleOCR imported successfully

Cell 5 - run the router:

  %run "/content/invoice_ocr_qwen_ollama_v3/pdf_router.py"

The upload dialog accepts:

- one or more PDFs, or
- one ZIP containing any number of PDFs (nested folders are okay).

The program downloads invoice_pdf_router_results.zip when complete.

Do not run the install cells again in the same healthy runtime. After a runtime
restart, rerun the install and verification cells because Colab packages are
ephemeral.


CCACHE WARNING
--------------

The message "No ccache found" is only an optional compilation-speed warning.
It does not change OCR results, installed libraries, CUDA availability, or GPU
memory. No fix is required for normal execution.


OUTPUT
------

pdf_router_output/
  pdf_router_batch_summary.json
  <source-pdf>/
    invoice_0001_<invoice-number>/
      invoice_extracted.json
      invoice_extracted.xml
    invoice_0002_<invoice-number>/
      ...
    page_diagnostics/
      ...raw OCR files only for pages that actually used OCR...

Important audit fields in each full result:

  extraction_method
  page_extraction_methods
  page_numbers
  page_roles
  invoice_page_count
  document_page_count
  multi_page_handling: grouped_by_invoice

Tax fields include both canonical and explicit total aliases:

  subtotal
  cgst_amount / total_cgst_amount
  sgst_amount / total_sgst_amount
  igst_amount / total_igst_amount
  tax / total_tax_amount
  total_amount


MINIMAL PRESENTATION OUTPUT
---------------------------

For a single image/PDF demonstration run:

  %run "/content/invoice_ocr_qwen_ollama_v3/presentation_ocr.py"

Its extracted_invoice.json contains only:

  {
    "fields": { ... },
    "missing_required": []
  }

One logical invoice returns one object even if it spans several pages. A batch
PDF containing multiple invoices returns an ordered list of these objects.


LOCAL CPU EXECUTION (NO GPU REQUIRED)
------------------------------------

Native-text PDFs need no OCR model and run quickly on CPU. Scanned PDFs also
work on CPU but PaddleOCR will be considerably slower.

Create an isolated environment, then install CPU Paddle:

Windows:

  python -m venv .venv_invoice
  .venv_invoice\Scripts\activate
  python -m pip install paddlepaddle==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
  python -m pip install -r requirements.txt
  python pdf_router.py invoice.pdf

Linux/macOS:

  python3 -m venv .venv_invoice
  source .venv_invoice/bin/activate
  python -m pip install paddlepaddle==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
  python -m pip install -r requirements.txt
  python pdf_router.py invoice.pdf

Other accepted local inputs:

  python pdf_router.py batch.zip
  python pdf_router.py folder_with_pdfs --output-dir pdf_router_output


LOCAL QWEN2.5-VL-3B FALLBACK (OLLAMA)
------------------------------------

Ollama is optional and is intended for local execution. Install Ollama, then
download the model once:

  ollama pull qwen2.5vl:3b

Make sure Ollama is running. On Windows the desktop app normally starts the
local service automatically. Otherwise run this in a separate terminal:

  ollama serve

Run the router with semantic fallback enabled:

  python pdf_router.py invoice.pdf --enable-qwen

For a ZIP or folder batch:

  python pdf_router.py invoices.zip --enable-qwen
  python pdf_router.py folder_with_pdfs --enable-qwen

Useful options:

  --ollama-model qwen2.5vl:3b
  --ollama-url http://127.0.0.1:11434
  --ollama-timeout 180
  --no-qwen-image-fallback

Behavior:

- If missing_required is empty, Ollama is not called.
- Native PDFs use embedded PDF text only; no image is sent.
- Scanned pages try PaddleOCR text first. Only unresolved fields escalate
  to the original rendered page image.
- Qwen fills only required fields that are still missing.
- Bill To/Buyer/Recipient is accepted as customer; Ship To/Deliver To is not.
- total_amount means the gross invoice total, never Balance Due or Payment Made.
- If Ollama is stopped, missing, or times out, the batch does not crash. The
  deterministic result is preserved and semantic_fallback.errors explains it.

The Python Ollama package is not required: the module uses Python's standard
library to call the local Ollama API. CPU execution works but vision inference
will be slower and needs enough system RAM for the model.


TESTS (NO GPU/MODEL DOWNLOAD)
-----------------------------

From the package directory:

  python -m unittest discover -s tests -v


IMPORTANT
---------

- Never install paddlepaddle and paddlepaddle-gpu together.
- torch.cuda.is_available() being False is correct for this setup.
- Check Paddle GPU with paddle.device.get_device(), not torch.cuda.
- 300 DPI is the minimum render resolution for scanned PDF pages.
- Cache cleanup releases unused GPU buffers only; it does not uninstall or
  modify packages.
