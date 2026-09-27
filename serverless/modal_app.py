"""Modal deployment of the query encoder: query text -> Nemotron ColEmbed multi-vector embedding.

Same model loading and output format as serverless/handler.py (RunPod), on Modal instead: no Docker build,
no registry, weights cached in a Modal Volume after the first cold start, container scales to zero.

    modal token new                       # once, browser login
    modal deploy serverless/modal_app.py  # from the repo root
    modal run serverless/modal_app.py     # smoke test: encodes one query and prints the shape

Client side: retriever/vision_remote.modal_query_encoder() looks the deployed class up by name, so the MCP
server needs only the Modal token (~/.modal.toml), no endpoint URL or API key of its own.
"""
from __future__ import annotations

import base64
import os

import modal

APP_NAME = "pharma-vision-encoder"
MODEL_ID = "nvidia/llama-nemotron-colembed-vl-3b-v2"
GPU = os.environ.get("MODAL_GPU", "L4")     # 24 GB, bf16 native; T4 works too (bf16 emulated, slower)
MAX_QUERIES, MAX_CHARS = 16, 2_000

image = (
    modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime", add_python=None)
    .pip_install("transformers>=4.45,<5", "accelerate", "einops", "sentencepiece", "datasets", "Pillow",
                 "huggingface_hub", "hf_transfer", "numpy<3")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf"})
)
weights = modal.Volume.from_name("pharma-nemotron-hf", create_if_missing=True)
app = modal.App(APP_NAME)


def parse_queries(qs) -> list[str]:
    if isinstance(qs, str):
        qs = [qs]
    if not isinstance(qs, list) or not qs:
        raise ValueError("queries must be a non-empty list of strings")
    if len(qs) > MAX_QUERIES:
        raise ValueError(f"at most {MAX_QUERIES} queries per call, got {len(qs)}")
    for i, q in enumerate(qs):
        if not isinstance(q, str) or not q.strip():
            raise ValueError(f"query {i} must be a non-empty string")
        if len(q) > MAX_CHARS:
            raise ValueError(f"query {i} has {len(q)} chars, max {MAX_CHARS}")
    return qs


@app.cls(image=image, gpu=GPU, volumes={"/hf": weights}, scaledown_window=60, timeout=600)
class Encoder:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(MODEL_ID, device_map="cuda", trust_remote_code=True,
                                               torch_dtype=torch.bfloat16).eval()
        floats = (torch.float32, torch.float16, torch.bfloat16, torch.float64)
        for t in list(self.model.parameters()) + list(self.model.buffers()):
            if t.dtype in floats:  # internal ViT is fp32-pinned
                t.data = t.data.to(torch.bfloat16)
        self.no_grad = torch.no_grad
        weights.commit()  # keep the downloaded weights for the next cold start

    @modal.method()
    def encode(self, queries) -> list[dict]:
        """[{"shape": [tokens, 3072], "dtype": "float16", "data": base64}, ...] in input order."""
        import numpy as np
        out = []
        for q in parse_queries(queries):
            with self.no_grad():
                emb = self.model.forward_queries([q], batch_size=1)
            t = emb[0] if isinstance(emb, list) or emb.dim() == 3 else emb
            a = np.ascontiguousarray(t.detach().float().cpu().numpy(), dtype=np.float16)
            out.append({"shape": list(a.shape), "dtype": "float16",
                        "data": base64.b64encode(a.tobytes()).decode("ascii")})
        return out


@app.local_entrypoint()
def main(query: str = "Dupixent quarterly sales chart"):
    import time
    t = time.time()
    item = Encoder().encode.remote([query])[0]
    print(f"{item['shape']} {item['dtype']} in {time.time() - t:.1f}s (includes the cold start on a first call)")
