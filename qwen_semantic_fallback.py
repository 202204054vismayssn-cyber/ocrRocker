"""Optional local Qwen semantic fallback for canonical invoice results.

The rule-based native/OCR paths remain authoritative.  This module is called
only when required fields are missing, and it never overwrites a present value.
"""

from __future__ import annotations

import base64
import copy
import datetime as dt
import json
import math
import re
import urllib.error
import urllib.request
from pathlib import Path


class QwenFallbackConfig:
    """Settings for the optional Qwen semantic fallback.

    The fallback is opt-in: with ``enabled=False`` the router never contacts
    Ollama, so the OCR pipeline stays fully self-contained.  When enabled it is
    consulted only for fields the deterministic pipeline could not resolve —
    it never overwrites a value that was already extracted.
    """

    def __init__(
        self,
        enabled=True,
        model="qwen2.5vl:3b",
        base_url="http://127.0.0.1:11434",
        timeout_seconds=180,
        image_escalation=True,
        max_images=3,
        max_text_chars=30_000,
    ):
        self.enabled = enabled
        self.model = model
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.image_escalation = image_escalation
        self.max_images = max_images
        self.max_text_chars = max_text_chars


class OllamaClient:
    """Small stdlib-only client for Ollama's local chat endpoint."""

    def __init__(self, config: QwenFallbackConfig):
        self.config = config

    def chat(self, prompt: str, response_schema: dict, image_paths=None) -> dict:
        """Send one prompt to Ollama and return its JSON reply.

        ``response_schema`` is passed through as Ollama's ``format`` parameter
        so the model is constrained to structured output.  ``image_paths`` are
        base64-encoded and attached, which is what makes the vision tier
        possible.  Raises on any transport or HTTP error — callers treat that
        as non-fatal and keep the partial result.
        """
        message = {"role": "user", "content": prompt}
        paths = list(image_paths or [])[: self.config.max_images]
        if paths:
            message["images"] = [
                base64.b64encode(Path(path).read_bytes()).decode("ascii")
                for path in paths
            ]
        payload = {
            "model": self.config.model,
            "messages": [message],
            "format": response_schema,
            "stream": False,
            "options": {"temperature": 0},
        }
        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                envelope = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        content = (envelope.get("message") or {}).get("content")
        if isinstance(content, dict):
            return content
        try:
            return json.loads(content or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned non-JSON content.") from exc


class QwenSemanticFallback:
    """Fill unresolved required fields using text, then optional page images."""

    def __init__(self, canonical_schema: dict, config=None, client=None):
        self.schema = canonical_schema
        self.config = config or QwenFallbackConfig()
        self.client = client or OllamaClient(self.config)

    @staticmethod
    def _present(value) -> bool:
        """Return True when a field holds a real value, not None or blank text."""
        return value is not None and (not isinstance(value, str) or bool(value.strip()))

    def _missing(self, fields: dict) -> list[str]:
        """List the required canonical fields that are still unresolved.

        Only required fields are ever sent to the model — the fallback exists to
        close statutory gaps, not to second-guess values already extracted.
        """
        return [
            name for name, spec in self.schema.items()
            if spec.get("required") and not self._present(fields.get(name))
        ]

    def _response_schema(self, missing: list[str]) -> dict:
        """Build the JSON schema constraining the model's reply.

        Only the missing fields are declared, and every one is nullable, so
        returning null for an unreadable value is a valid answer.  This is
        passed to Ollama as ``format``, which is what keeps the model from
        inventing prose around the JSON.
        """
        properties = {}
        for name in missing:
            kind = self.schema.get(name, {}).get("type")
            json_type = "number" if kind == "float" else "string"
            properties[name] = {"type": [json_type, "null"]}
        return {
            "type": "object",
            "properties": {
                "fields": {
                    "type": "object",
                    "properties": properties,
                    "required": missing,
                    "additionalProperties": False,
                }
            },
            "required": ["fields"],
            "additionalProperties": False,
        }

    def _prompt(self, result: dict, missing: list[str], raw_text: str, vision: bool) -> str:
        """Build the instruction sent to Qwen for one escalation tier.

        Already-extracted values are included as context so the model fills
        gaps rather than re-reading the document from scratch.  The wording
        encodes the two mistakes that actually happen on invoices: treating
        Ship To as the customer, and returning a subtotal or balance due as
        the total.
        """
        known = {
            key: value for key, value in (result.get("fields") or {}).items()
            if key != "line_items" and self._present(value)
        }
        source_note = (
            "You can inspect the attached original invoice page image(s)."
            if vision else
            "Use only the extracted invoice text below."
        )
        return f"""You are a precise invoice field resolver. {source_note}
Return JSON matching the supplied schema. Fill only these missing fields: {missing}.
Use null when evidence is absent or ambiguous. Never guess and never overwrite known fields.
Customer means Bill To, Buyer, Billed To, or Recipient. Never use Ship To or Deliver To.
total_amount means the final gross invoice total, not subtotal, balance due, payment made, or an individual line amount.
invoice_date must be YYYY-MM-DD. Numeric amounts must be JSON numbers without currency symbols.

Known canonical fields:
{json.dumps(known, ensure_ascii=False)}

Extracted document text:
{raw_text[-self.config.max_text_chars:]}
"""

    def _coerce(self, field: str, value):
        """Validate and normalise one model-supplied value, or return None.

        This is the trust boundary: the model is told to return null when it is
        unsure, and every value it does return is re-checked against the
        canonical type before being accepted.  Amounts are stripped of
        currency noise, dates must be ISO, and two domain-specific rules reject
        a customer name that actually came from Ship To and an invoice number
        containing no digits.
        """
        if value is None:
            return None
        kind = self.schema.get(field, {}).get("type")
        if kind == "float":
            if isinstance(value, bool):
                return None
            if isinstance(value, str):
                cleaned = re.sub(r"[^0-9.()\-]", "", value.replace(",", ""))
                if cleaned.startswith("(") and cleaned.endswith(")"):
                    cleaned = "-" + cleaned[1:-1]
                try:
                    value = float(cleaned)
                except ValueError:
                    return None
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return round(float(value), 2)
            return None
        value = re.sub(r"\s+", " ", str(value)).strip()
        if not value:
            return None
        if kind == "date":
            try:
                dt.date.fromisoformat(value)
            except ValueError:
                return None
        if field == "invoice_number" and not re.search(r"\d", value):
            return None
        if field == "customer_name" and re.search(r"\b(?:ship|deliver)\s*to\b", value, re.I):
            return None
        return value

    def _apply_tier(self, result: dict, missing: list[str], raw_text: str, image_paths, source: str):
        """Run one escalation tier and merge back only the fields it filled.

        Values that survived ``_coerce`` are written to the result and tagged in
        ``field_sources`` so it stays auditable which fields came from the
        deterministic pipeline and which from the model.  Already-populated
        fields are never touched.
        """
        response = self.client.chat(
            self._prompt(result, missing, raw_text, vision=bool(image_paths)),
            self._response_schema(missing),
            image_paths=image_paths,
        )
        proposed = response.get("fields") if isinstance(response, dict) else None
        proposed = proposed if isinstance(proposed, dict) else {}
        filled = []
        fields = result.setdefault("fields", {})
        sources = result.setdefault("field_sources", {})
        for name in missing:
            if self._present(fields.get(name)):
                continue
            value = self._coerce(name, proposed.get(name))
            if self._present(value):
                fields[name] = value
                sources[name] = source
                filled.append(name)
        return filled

    def apply(self, result: dict, *, raw_text: str = "", image_paths=None) -> dict:
        """Fill still-missing required fields, text tier first then vision.

        Returns a deep copy — the input is never mutated.  Two tiers are tried
        in order: plain text first (cheaper), then the original page images if
        text wasn't enough and escalation is enabled.  The vision tier is
        skipped entirely for native-PDF results, which have no rendered page
        image and gain nothing from it.

        Every failure is non-fatal.  If Ollama is unreachable, times out or
        returns nonsense, the deterministic result is returned unchanged with
        the error recorded in the audit block — a missing optional dependency
        must never turn a partial extraction into a crash.
        """
        output = copy.deepcopy(result)
        missing = self._missing(output.get("fields") or {})
        if not self.config.enabled or not missing:
            return output

        audit = {
            "enabled": True,
            "triggered": True,
            "model": self.config.model,
            "tiers_attempted": [],
            "filled_fields": [],
            "remaining_missing": list(missing),
            "errors": [],
        }
        try:
            audit["tiers_attempted"].append("text")
            audit["filled_fields"].extend(
                self._apply_tier(output, missing, raw_text, None, "qwen_text_fallback")
            )
        except Exception as exc:  # fail open: deterministic output is preserved
            audit["errors"].append(f"text: {exc}")

        remaining = self._missing(output.get("fields") or {})
        usable_images = [str(path) for path in (image_paths or []) if Path(path).is_file()]
        if remaining and self.config.image_escalation and usable_images:
            try:
                audit["tiers_attempted"].append("vision")
                audit["filled_fields"].extend(
                    self._apply_tier(
                        output, remaining, raw_text, usable_images,
                        "qwen_vision_fallback",
                    )
                )
            except Exception as exc:
                audit["errors"].append(f"vision: {exc}")

        remaining = self._missing(output.get("fields") or {})
        output["missing_required"] = remaining
        output["unparsed_or_low_confidence"] = list(remaining)
        output["needs_review"] = bool(remaining or output.get("validation_issues"))
        audit["remaining_missing"] = list(remaining)
        output["semantic_fallback"] = audit
        return output
