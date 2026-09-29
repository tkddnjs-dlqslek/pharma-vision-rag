# Nemotron query encoder + BGE-M3 + bge-reranker over HTTP on port 7860 (app.py). CPU only, about 12 GB resident:
# run it on a box with 16 GB RAM. Build context = this folder (docker compose does that from the repo root).
#
# ARM (aarch64, e.g. Oracle Cloud Always Free Ampere): python:3.11-slim is multi-arch and the PyTorch CPU index ships
# aarch64 wheels for 2.8.0; if that pip line fails on your arch, drop the --index-url (PyPI's aarch64 torch is CPU-only).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/data/hf \
    PIP_NO_CACHE_DIR=1

RUN pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# non-root; /data is the weights cache (mount a volume there, see docker-compose.yml)
RUN useradd -m -u 1000 user && mkdir -p /data/hf && chown -R user /data
USER user
WORKDIR /home/user/app
COPY --chown=user app.py .

EXPOSE 7860
CMD ["python", "app.py"]
