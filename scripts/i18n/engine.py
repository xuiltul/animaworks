"""Small synchronous HTTP clients for OpenAI-compatible translation engines."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TranslationResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


class TranslationEngineError(RuntimeError):
    """A sanitized engine failure that never includes credentials or response bodies."""


class EngineClient:
    def __init__(self, name: str, definition: dict[str, Any]) -> None:
        self.name = name
        self.definition = definition
        self.kind = str(definition.get("kind", "openai"))
        self.model = str(definition.get("model", definition.get("deployment", "")))
        self.timeout = float(definition.get("timeout_seconds", 600))

    def _request_details(self) -> tuple[str, dict[str, str], dict[str, Any]]:
        api_key_env = str(self.definition.get("api_key_env", ""))
        api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        if self.kind == "azure_openai":
            endpoint_env = str(self.definition.get("endpoint_env", "AZURE_OPENAI_ENDPOINT"))
            endpoint = os.environ.get(endpoint_env, "").rstrip("/")
            deployment = str(self.definition.get("deployment", ""))
            api_version = str(self.definition.get("api_version", "2025-04-01-preview"))
            if not endpoint or not deployment:
                raise TranslationEngineError(
                    f"Engine {self.name!r} is missing Azure endpoint or deployment configuration"
                )
            url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
            headers = {"api-key": api_key} if api_key else {}
            model_field: dict[str, Any] = {}
        else:
            base_url = str(self.definition.get("base_url", "")).rstrip("/")
            model = str(self.definition.get("model", ""))
            if not base_url or not model:
                raise TranslationEngineError(f"Engine {self.name!r} is missing base URL or model configuration")
            url = f"{base_url}/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}"} if api_key and api_key != "EMPTY" else {}
            model_field = {"model": model}
        return url, headers, model_field

    def translate(self, system_prompt: str, user_prompt: str) -> TranslationResponse:
        url, headers, model_field = self._request_details()
        payload: dict[str, Any] = {
            **model_field,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        # Azure reasoning deployments reject max_tokens and non-default temperatures.
        token_field = "max_completion_tokens" if self.kind == "azure_openai" else "max_tokens"
        payload[token_field] = int(self.definition.get("max_output_tokens", 16000))
        if "temperature" in self.definition:
            payload["temperature"] = float(self.definition["temperature"])
        # Engine-specific request fields, e.g. turning off a model's thinking mode.
        extra_body = self.definition.get("extra_body")
        if isinstance(extra_body, dict):
            payload.update(extra_body)
        try:
            import httpx

            response = httpx.post(url, headers=headers, json=payload, timeout=self.timeout)
        except ImportError:
            raise TranslationEngineError("The httpx package is required to call translation engines") from None
        except httpx.HTTPError as exc:
            raise TranslationEngineError(f"Engine {self.name!r} request failed ({type(exc).__name__})") from None
        if response.status_code >= 400:
            raise TranslationEngineError(f"Engine {self.name!r} returned HTTP {response.status_code}")
        try:
            data = response.json()
            text = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise TranslationEngineError(f"Engine {self.name!r} returned an invalid completion response") from None
        if not isinstance(text, str):
            raise TranslationEngineError(f"Engine {self.name!r} returned non-text content")
        usage = data.get("usage") or {}
        return TranslationResponse(
            text=text,
            input_tokens=int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0),
            output_tokens=int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0),
            model=str(data.get("model") or self.model),
        )
