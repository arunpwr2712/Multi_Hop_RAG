"""Minimal LLM client for Ollama text generation."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from .config import LLMConfig


@dataclass
class LLMClient:
    """Thin HTTP client for Ollama text generation."""

    config: LLMConfig

    @property
    def available(self) -> bool:
        """Return whether the client has enough configuration to make a request."""

        return bool(self.config.model and self.config.base_url)

    def is_ollama_ready(self) -> bool:
        """Check if Ollama server is responding to requests."""

        if not self.available:
            return False

        try:
            endpoint = urljoin(self.config.base_url.rstrip("/") + "/", "api/tags")
            request = Request(endpoint, method="GET")
            with urlopen(request, timeout=5) as response:
                response.read()
                return response.status == 200
        except (HTTPError, URLError, TimeoutError, socket.timeout, Exception):
            return False

    def generate(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 512) -> str:
        """Generate text using Ollama."""

        if not self.available:
            raise RuntimeError("Ollama is not configured (model/base URL missing)")
        return self._generate_ollama(prompt=prompt, system_prompt=system_prompt, max_tokens=max_tokens)

    def _generate_ollama(self, prompt: str, system_prompt: Optional[str], max_tokens: int) -> str:
        """Call the Ollama generate endpoint."""

        if not self.config.base_url:
            raise RuntimeError("Ollama base URL is not configured")

        endpoint = urljoin(self.config.base_url.rstrip("/") + "/", "api/generate")
        final_prompt = prompt
        if system_prompt:
            final_prompt = f"System: {system_prompt}\n\nUser: {prompt}"

        payload = {
            "model": self.config.model,
            "prompt": final_prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": max_tokens,
            },
        }
        request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, socket.timeout) as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        answer = raw.get("response", "")
        if not answer:
            raise RuntimeError("Ollama returned an empty response")
        return answer.strip()
