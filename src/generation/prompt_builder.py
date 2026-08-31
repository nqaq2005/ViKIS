"""Prompt construction helpers for multimodal answer generation."""

from __future__ import annotations

from typing import Any


class PromptBuilder:
    """Build a compact prompt from retrieved context and evidence."""

    def __init__(self, system_prompt: str | None = None) -> None:
        self.system_prompt = system_prompt or (
            "Bạn là trợ lý trả lời câu hỏi dựa trên nội dung video và transcript đã tìm được."
        )

    def build(self, query: str, context: list[dict[str, Any]]) -> str:
        """Assemble a prompt string from a user query and evidence."""
        evidence = []
        for index, item in enumerate(context, start=1):
            timestamp = item.get("timestamp", "unknown")
            text = item.get("text", "")
            evidence.append(f"[{index}] Timestamp: {timestamp}\n{text}")

        context_text = "\n\n".join(evidence) if evidence else "Không có ngữ cảnh phù hợp."
        return (
            f"{self.system_prompt}\n\n"
            f"Câu hỏi: {query}\n\n"
            f"Ngữ cảnh:\n{context_text}"
        )
