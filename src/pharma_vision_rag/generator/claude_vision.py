"""Claude Vision generator — wraps Anthropic SDK for pharma PDF Q&A.

Input: query text + list of retrieved page images (usually top-3).
Output: natural-language answer grounded in those images.

Design choices
--------------
- Model default: ``claude-sonnet-4-6`` (latest Sonnet as of 2026-04). The PRD
  specifies "Claude Sonnet 4.7"; swap when 4.7 ships.
- Prompt caching: system prompt + up to 3 images get ``cache_control`` so that
  repeat queries sharing pages pay ~0.1x for cached tokens (max 4 breakpoints).
- Image inputs: accept ``PIL.Image.Image``, ``Path``/``str``, or raw ``bytes``.
- No streaming: answers are short (<=1024 tokens). Add streaming only if outputs
  grow or batch eval hits SDK HTTP timeouts.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Union

import anthropic
from PIL import Image

ImageInput = Union[Path, str, bytes, Image.Image]

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 1024
MAX_CACHED_IMAGES = 3  # + 1 for system prompt = 4 breakpoints (the API cap)

DEFAULT_SYSTEM_PROMPT = """You are a pharma analyst answering questions about Sanofi's public disclosures (Form 20-F, quarterly press releases, clinical trial summaries).

Rules:
- Answer ONLY from the provided page images. If the answer is not on the pages, reply "정보를 찾을 수 없습니다." (Korean questions) or "Not found in the provided pages." (English questions).
- Quote numbers verbatim with currency/units (e.g., €3.5 billion, 20.3%) and period labels (FY2025, Q1 2025, H1 2025).
- Prefer the more specific page when multiple pages cover the same metric at different granularity (e.g., Q1 press release over 20-F annual).
- Keep the answer to 2-3 sentences. No preamble ("Based on the provided...", "According to...")."""


def _image_to_base64(image: ImageInput) -> tuple[str, str]:
    """Convert any supported image input to ``(media_type, base64 string)``."""
    if isinstance(image, (str, Path)):
        path = Path(image)
        media_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(path.suffix.lower(), "image/png")
        return media_type, base64.standard_b64encode(path.read_bytes()).decode()

    if isinstance(image, bytes):
        return "image/png", base64.standard_b64encode(image).decode()

    if isinstance(image, Image.Image):
        buf = io.BytesIO()
        # Force RGB to avoid CMYK/alpha edge cases from PDF render backends.
        rgb = image.convert("RGB")
        rgb.save(buf, format="PNG")
        return "image/png", base64.standard_b64encode(buf.getvalue()).decode()

    raise TypeError(f"Unsupported image type: {type(image).__name__}")


class ClaudeVisionGenerator:
    """Answer questions from pages using Claude Sonnet Vision."""

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model: str = DEFAULT_MODEL,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

    def generate(
        self,
        query: str,
        images: list[ImageInput] | None = None,
        cache_images: bool = True,
    ) -> dict[str, Any]:
        """Generate an answer for ``query`` grounded in ``images``.

        Returns a dict with: ``answer`` (str), ``usage`` (token + cache counts),
        ``stop_reason``, and ``raw`` (the full SDK response object).
        """
        images = images or []
        user_content: list[dict[str, Any]] = []

        for i, image in enumerate(images):
            media_type, data = _image_to_base64(image)
            block: dict[str, Any] = {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            }
            if cache_images and i < MAX_CACHED_IMAGES:
                block["cache_control"] = {"type": "ephemeral"}
            user_content.append(block)

        user_content.append({"type": "text", "text": query})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{
                "type": "text",
                "text": self.system_prompt,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_content}],
        )

        answer = next((b.text for b in response.content if b.type == "text"), "")
        usage = response.usage
        return {
            "answer": answer,
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
            },
            "stop_reason": response.stop_reason,
            "raw": response,
        }
