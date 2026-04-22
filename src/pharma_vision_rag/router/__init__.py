"""LangGraph wrappers — express Hybrid mode as an explicit graph.

The graph nodes are: route -> (text + vision in parallel) -> merge -> render -> answer.
Identical end behavior to ``modes.hybrid.HybridMode`` but with named graph nodes
that show up cleanly in Langfuse traces and demonstrate LangGraph patterns
for portfolio purposes.
"""
from pharma_vision_rag.router.langgraph_router import HybridGraph

__all__ = ["HybridGraph"]
