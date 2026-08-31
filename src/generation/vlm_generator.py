"""Placeholder VLM generator for multimodal answer synthesis."""

from __future__ import annotations

from typing import Any


class VLMGenerator:
    """Simple stub used to integrate an LLM/VLM answer generation step."""

    def __init__(self, model_name: str | None = None, api_key: str | None = None) -> None:
        self.model_name = model_name
        self.api_key = api_key

    def generate(self, prompt: str, context: list[dict[str, Any]] | None = None) -> str:
        """Return a generated answer from the provided prompt and optional context."""
        if context:
            return f"[VLM mock response]\nPrompt: {prompt}\nContext items: {len(context)}"
        return f"[VLM mock response]\nPrompt: {prompt}"
