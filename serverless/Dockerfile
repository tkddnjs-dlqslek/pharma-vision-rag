# RunPod Serverless query encoder (handler.py). Build from this directory:
#   docker build --platform linux/amd64 --build-arg BAKE_MODEL=1 -t <registry>/<user>/nemotron-query:v1 .
#
# BAKE_MODEL=1  weights downloaded into the image at build time (fast cold start, ~15 GB image).
#               nvidia/llama-nemotron-colembed-vl-3b-v2 is under an NVIDIA NON-COMMERCIAL license:
#               push this image ONLY to a PRIVATE registry, never a public Docker Hub repo.
# BAKE_MODEL=0  small image; weights downloaded at cold start into /runpod-volume/hf when a network volume
#               is attached (kept across workers), otherwise into container disk (every cold start).
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

ARG BAKE_MODEL=1
ENV BAKED_MODEL=${BAKE_MODEL} \
    HF_HOME=/models/hf \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1

# query path only: no pypdfium2/docling (pages are embedded offline by scripts/embed_pages_gpu.py)
# datasets is imported by the model's own remote code, not by us
RUN pip install --no-cache-dir "transformers>=4.45,<5" accelerate einops sentencepiece datasets Pillow \
        huggingface_hub hf_transfer runpod "numpy<3"

# model repo is public (gated=False, checked 2026-09-20), so no HF token is needed at build time
RUN if [ "$BAKE_MODEL" = "1" ]; then \
        python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/llama-nemotron-colembed-vl-3b-v2')"; \
    fi

WORKDIR /app
COPY handler.py .
CMD ["python", "-u", "handler.py"]
