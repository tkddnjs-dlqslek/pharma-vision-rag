# The pharma corpus MCP server over streamable HTTP on port 8000 (app.py). Build context = the REPO ROOT, because
# src/ is copied in:  docker build -f services/mcp/Dockerfile .   (docker compose does this). Multi-arch base image.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    MCP_DATA_DIR=/data \
    PORT=8000 \
    PIP_NO_CACHE_DIR=1

RUN pip install "mcp>=2.2,<3" "numpy<3" "pypdfium2>=4.30" "Pillow>=10.4" "python-dotenv>=1.0" \
        "qdrant-client>=1.12,<2" huggingface_hub hf_transfer uvicorn

# non-root; /data holds the downloaded corpus (DATA_REPO) and auth.json: mount a volume there.
# mcp_server.py reads ROOT/data and ROOT/eval, so both point into the volume.
RUN useradd -m -u 1000 user && mkdir -p /data && chown user /data \
    && mkdir -p /home/user/app && ln -s /data/data /home/user/app/data && ln -s /data/eval /home/user/app/eval
USER user
WORKDIR /home/user/app
COPY --chown=user src ./src
COPY --chown=user services/mcp/app.py .

EXPOSE 8000
CMD ["python", "app.py"]
