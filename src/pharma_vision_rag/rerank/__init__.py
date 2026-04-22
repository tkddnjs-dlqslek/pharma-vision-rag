"""Rerankers — re-score (query, candidate) pairs to refine retriever top-k.

Two implementations for Phase 3 A/B:
- ZeRank2Reranker (here)        — generic multilingual cross-encoder.
- NemotronRerankReranker (TBD)  — NVIDIA's llama-nemotron-rerank-vl-1b-v2,
  same ecosystem as the Nemotron retriever (Colab tunnel).
"""
from pharma_vision_rag.rerank.zerank2 import ZeRank2Reranker

__all__ = ["ZeRank2Reranker"]
