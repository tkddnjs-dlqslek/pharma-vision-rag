"""Modal (CPU, free credits) copy of the text-side query models: BGE-M3 dense query embedding and bge-reranker scoring.

Used only when a question's wording changes (scripts/28_apply_pending_questions.py): the dev box cannot hold
BGE-M3 and the reranker (2.3 GB each) next to everything else, and the RunPod pod that computed
data/embeddings/v2/text/ is gone. Same models and settings as scripts/text_retrieval_gpu.py.

    modal deploy serverless/modal_text.py
    modal run serverless/modal_text.py      # smoke test
"""
from __future__ import annotations

import modal

APP_NAME = "pharma-text-models"
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
EMBED_MAX_SEQ = 1024   # = scripts/text_retrieval_gpu.py
RERANK_MAX_LEN = 512

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", extra_index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("sentence-transformers>=3,<6", "numpy<3", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf"})
)
weights = modal.Volume.from_name("pharma-text-hf", create_if_missing=True)
app = modal.App(APP_NAME)


@app.cls(image=image, cpu=8.0, memory=16_384, volumes={"/hf": weights}, scaledown_window=120, timeout=900)
class TextModels:
    @modal.enter()
    def load(self):
        from sentence_transformers import CrossEncoder, SentenceTransformer
        self.embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
        self.embedder.max_seq_length = EMBED_MAX_SEQ
        self.reranker = CrossEncoder(RERANK_MODEL, max_length=RERANK_MAX_LEN, device="cpu")
        weights.commit()

    @modal.method()
    def embed(self, queries: list[str]) -> list[list[float]]:
        """Normalised 1024-d dense vectors, one per query."""
        return self.embedder.encode(queries, batch_size=16, normalize_embeddings=True).astype("float32").tolist()

    @modal.method()
    def rerank(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self.reranker.predict([[query, t] for t in texts], show_progress_bar=False)]


@app.local_entrypoint()
def main():
    import time
    t = time.time()
    m = TextModels()
    v = m.embed.remote(["What were Dupixent sales in Q2 2025?"])
    s = m.rerank.remote("What were Dupixent sales in Q2 2025?", ["Dupixent sales were EUR 3,832 million", "Beyfortus sales rose"])
    print(f"embed dim {len(v[0])}, rerank scores {s}, {time.time() - t:.1f}s including cold start")
