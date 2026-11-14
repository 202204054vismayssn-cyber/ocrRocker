"""Asynchronous Ollama client for Gemma structured JSON invoice extraction."""

import json
import logging
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from uae_invoice_reader.common.config import Settings
from uae_invoice_reader.common.exceptions import OllamaServiceError

logger = logging.getLogger("uae_invoice_reader.ollama")

T = TypeVar("T", bound=BaseModel)


class OllamaService:
    """Invokes local or remote Ollama models with native JSON Schema output."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model
        self.timeout = settings.ollama_timeout_seconds

    async def extract_structured(
        self,
        system_prompt: str,
        ocr_text: str,
        response_schema: type[T],
    ) -> tuple[T, dict[str, Any]]:
        """Send invoice text to Gemma via Ollama and return validated structured data."""

        user_content = f"Extract all details from the following invoice OCR text into the requested schema:\n\n{ocr_text}"

        # Dereference and inline schema $defs so Ollama grammar parser handles nested items
        clean_schema = self._clean_schema_for_ollama(response_schema.model_json_schema())

        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_content},
            ],
            "format": clean_schema,
            "stream": False,
            "options": {
                "temperature": 0.0,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
        except httpx.ConnectError as exc:
            logger.error("Failed to connect to Ollama at %s: %s", url, exc)
            raise OllamaServiceError(
                f"Could not connect to Ollama at {self.base_url}. Ensure Ollama is running.",
                status_code=503,
            ) from exc
        except httpx.TimeoutException as exc:
            logger.error("Ollama extraction timed out after %s seconds", self.timeout)
            raise OllamaServiceError(
                f"Ollama extraction timed out after {self.timeout} seconds",
                status_code=504,
            ) from exc
        except Exception as exc:
            logger.error("Unexpected error calling Ollama: %s", exc)
            raise OllamaServiceError(
                f"Ollama request failed: {exc}", status_code=502
            ) from exc

        if response.status_code != 200:
            logger.error("Ollama HTTP %s: %s", response.status_code, response.text)
            raise OllamaServiceError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:200]}",
                status_code=502,
            )

        data = response.json()
        raw_message = data.get("message", {}).get("content", "").strip()

        cleaned_json_str = self._clean_json_markdown(raw_message)

        try:
            parsed_data = json.loads(cleaned_json_str)
            validated_model = response_schema.model_validate(parsed_data)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.error("Failed to parse or validate Gemma JSON output: %s\nRaw output: %s", exc, raw_message[:1000])
            raise OllamaServiceError(
                f"Gemma output validation failed: {exc}",
                status_code=422,
            ) from exc

        # Collect performance and token usage metadata from Ollama
        prompt_tokens = data.get("prompt_eval_count", 0) or 0
        completion_tokens = data.get("eval_count", 0) or 0
        total_duration_ms = int((data.get("total_duration", 0) or 0) / 1_000_000)

        token_usage = {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "execution_time_ms": total_duration_ms,
            "model": self.model,
        }

        return validated_model, token_usage

    @classmethod
    def _clean_schema_for_ollama(cls, schema: dict[str, Any]) -> dict[str, Any]:
        """Resolve and inline $defs/$ref pointers for Ollama GBNF grammar parser."""
        import copy

        schema = copy.deepcopy(schema)
        defs = schema.pop("$defs", {})
        if not defs:
            return schema

        def resolve(node: Any) -> Any:
            if isinstance(node, dict):
                if "$ref" in node:
                    ref_key = node["$ref"].split("/")[-1]
                    if ref_key in defs:
                        return resolve(copy.deepcopy(defs[ref_key]))
                return {k: resolve(v) for k, v in node.items()}
            elif isinstance(node, list):
                return [resolve(x) for x in node]
            return node

        return resolve(schema)

    @staticmethod
    def _clean_json_markdown(raw: str) -> str:
        """Extract valid JSON from LLM output, ignoring markdown code blocks and conversational text."""
        import re

        text = raw.strip()
        # 1. Match ```json { ... } ```
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # 2. Match from first '{' to last '}'
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1].strip()
        return text
