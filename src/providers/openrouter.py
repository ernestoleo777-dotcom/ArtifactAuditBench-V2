from __future__ import annotations

import json
import http.client
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


class ProviderConfigurationError(RuntimeError):
    pass


@dataclass
class ProviderResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency_seconds: float
    raw_usage: dict


class OpenRouterProvider:
    def __init__(self, config: dict):
        api_key = os.environ.get(config["api_key_env"])
        model = os.environ.get(config["model_env"])
        if not api_key:
            raise ProviderConfigurationError(f"missing environment variable: {config['api_key_env']}")
        if not model:
            raise ProviderConfigurationError(f"missing environment variable: {config['model_env']}")
        self._api_key = api_key
        self.model = model
        self.base_url = os.environ.get(config["base_url_env"], config["default_base_url"]).rstrip("/")
        self.temperature = config["temperature"]
        self.max_tokens = config["max_output_tokens"]
        self.max_retries = config["max_retries"]
        self.timeout = config["timeout_seconds"]

    def safe_configuration(self) -> dict:
        return {
            "provider": "openrouter",
            "model": self.model,
            "base_url": self.base_url,
            "api_key_configured": bool(self._api_key),
        }

    def complete(self, messages: list[dict], *, max_retries: int | None = None) -> ProviderResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://artifactaudit.local",
                "X-Title": "ArtifactAudit Direct MLLM",
            },
            method="POST",
        )
        attempts = self.max_retries if max_retries is None else max_retries
        last_error = None
        for attempt in range(attempts + 1):
            start = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode())
                latency = time.perf_counter() - start
                usage = body.get("usage") or {}
                return ProviderResponse(
                    text=body["choices"][0]["message"]["content"],
                    input_tokens=int(usage.get("prompt_tokens", 0)),
                    output_tokens=int(usage.get("completion_tokens", 0)),
                    cost=float(usage.get("cost", 0.0) or 0.0),
                    latency_seconds=latency,
                    raw_usage={key: value for key, value in usage.items() if "key" not in key.lower()},
                )
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                TimeoutError,
                KeyError,
                ValueError,
                http.client.HTTPException,
            ) as exc:
                last_error = exc
                if attempt < attempts:
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"OpenRouter request failed after retries: {type(last_error).__name__}")

    def repair_json_format(self, invalid_text: str, schema: dict) -> ProviderResponse:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a JSON formatting repair utility. Convert the supplied text into one JSON "
                    "object matching the supplied schema. Preserve the original analysis and values. "
                    "Do not add new scientific reasoning. Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"schema": schema, "invalid_output": invalid_text}),
            },
        ]
        return self.complete(messages, max_retries=1)
