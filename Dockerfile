# Laya playground in a container.
#
#   CPU (default):
#     docker build -t laya-playground .
#
#   GPU (needs the NVIDIA driver on the host and, for Docker, nvidia-container-toolkit):
#     docker build -t laya-playground:cuda --build-arg TORCH_BACKEND=cuda .
#
#   With the weights baked into the image (useful where there is no persistent volume):
#     docker build -t laya-playground:baked --build-arg BAKE=1 .
#
# The two backends differ only in the index PyTorch comes from: the *Linux* cu130 wheel brings
# the CUDA runtime with it (the `nvidia-*` dependencies), so the base `nvidia/cuda` image is not
# needed at build time -- it is only needed at runtime, if you want the GPU.

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ARG TORCH_BACKEND=cpu
ARG TORCH_VERSION=2.14.1
ARG LAYA_VERSION=0.4.9
ARG BAKE=0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    # In the container the playground listens on all interfaces and, with no GPU, runs on CPU.
    LAYA_HOST=0.0.0.0 \
    LAYA_PORT=8000 \
    LAYA_DEVICE=cpu \
    # The weights go in /models: that is the path to mount as a volume.
    LAYA_HF_CACHE=/models
    # LAYA_THREADS is deliberately not set: the server derives the number of threads from the
    # container CPU limit (cgroup), otherwise from the host cores with a cap at 16.
    # Set it only to force a value: LAYA_THREADS=8

# uv from the official binary: consistent with the rest of the project and faster than pip.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

RUN useradd --create-home --uid 10001 laya \
    && mkdir -p /models \
    && chown laya:laya /models

# PyTorch first, from the right index; then `laya`, which finds it already satisfied and does not
# reinstall it. Note: `uv sync` is NOT used here, because pyproject.toml pins the cu130 index,
# which is the wrong one for the CPU image.
RUN set -eux; \
    if [ "$TORCH_BACKEND" = "cuda" ]; then index=cu130; else index=cpu; fi; \
    uv pip install --system --no-cache \
        "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${index}"; \
    uv pip install --system --no-cache "laya[serve]==${LAYA_VERSION}"

COPY src/ ./src/

# Optional: download the weights at build time (image +~1.7 GB, but no download on first start).
# Without a persistent volume this is almost always the right choice for a deploy.
RUN if [ "$BAKE" = "1" ]; then \
        python src/warmup.py --device cpu --checkpoint multilingual \
        && chown -R laya:laya /models; \
    fi

USER laya

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# `--warm` preloads in the background: without it, the first request pays the load.
# A single process: every worker would keep a copy of the weights in memory.
CMD ["python", "src/server.py", "--warm"]
