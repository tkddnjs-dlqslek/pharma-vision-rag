"""Claude Haiku text generator — used for query translation (QT) and HyDE.

Cheap and fast. Kept separate from ClaudeVisionGenerator because:
- different default model (Haiku vs Sonnet)
- no image handling
- different system prompts per use case (we pass them per-call)
"""
from __future__ import annotations

from typing import Any

import anthropic

DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 256


class ClaudeTextGenerator:
    """Thin wrapper around Anthropic Messages API for short text tasks."""

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, system: str, user: str, max_tokens: int | None = None) -> dict[str, Any]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        answer = next((b.text for b in response.content if b.type == "text"), "").strip()
        usage = response.usage
        return {
            "answer": answer,
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            },
        }
