# 02: Add missing docstrings to `native_invoice_parser.py` and `qwen_semantic_fallback.py`

**What to build:** Audit every public function in `native_invoice_parser.py` and `qwen_semantic_fallback.py` and add a clear one-line docstring to any that are missing one. No logic changes, no renames, no restructuring — documentation only.

**Blocked by:** None (can start immediately, parallel with 01)

**Status:** ready-for-agent

- [ ] Every public function in `native_invoice_parser.py` has a docstring
- [ ] Every public function and method in `qwen_semantic_fallback.py` has a docstring
- [ ] `python -m unittest discover -s tests -v` still passes with zero failures
- [ ] No logic or behaviour changes in either file
