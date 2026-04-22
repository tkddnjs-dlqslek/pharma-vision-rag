"""Generators — VLM/LLM wrappers: vision Q&A, translation, HyDE passage drafts."""
from pharma_vision_rag.generator.claude_text import ClaudeTextGenerator
from pharma_vision_rag.generator.claude_vision import ClaudeVisionGenerator

__all__ = ["ClaudeTextGenerator", "ClaudeVisionGenerator"]
