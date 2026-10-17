# Playground Laya in un container.
#
#   CPU (default):
#     docker build -t laya-playground .
#
#   GPU (serve il driver NVIDIA sull'host e, per Docker, nvidia-container-toolkit):
#     docker build -t laya-playground:cuda --build-arg TORCH_BACKEND=cuda .
#
#   Con i pesi gia' dentro l'immagine (utile dove non c'e' un volume persistente):
#     docker build -t laya-playground:baked --build-arg BAKE=1 .
#
# I due backend differiscono solo per l'indice da cui arriva PyTorch: la wheel *Linux* cu130
# porta con se' il runtime CUDA (dipendenze `nvidia-*`), quindi non serve l'immagine base
# `nvidia/cuda` a build time -- serve solo a runtime, se vuoi la GPU.

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ARG TORCH_BACKEND=cpu
ARG TORCH_VERSION=2.14.1
ARG LAYA_VERSION=0.4.9
ARG BAKE=0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    # Nel container il playground ascolta su tutte le interfacce e, senza GPU, va in CPU.
    LAYA_HOST=0.0.0.0 \
    LAYA_PORT=8000 \
    LAYA_DEVICE=cpu \
    # I pesi in /models: e' il percorso da montare come volume.
    LAYA_HF_CACHE=/models
    # LAYA_THREADS non e' impostata di proposito: il server ricava il numero di thread dal
    # limite di CPU del container (cgroup), altrimenti dai core dell'host con tetto a 16.
    # Impostala solo per forzare un valore: LAYA_THREADS=8

# uv dal binario ufficiale: coerente con il resto del progetto e piu' rapido di pip.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

RUN useradd --create-home --uid 10001 laya \
    && mkdir -p /models \
    && chown laya:laya /models

# PyTorch per primo, dall'indice giusto; poi `laya`, che lo trova gia' soddisfatto e non lo
# reinstalla. Nota: qui NON si usa `uv sync`, perche' pyproject.toml fissa l'indice cu130,
# che e' quello sbagliato per l'immagine CPU.
RUN set -eux; \
    if [ "$TORCH_BACKEND" = "cuda" ]; then index=cu130; else index=cpu; fi; \
    uv pip install --system --no-cache \
        "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${index}"; \
    uv pip install --system --no-cache "laya[serve]==${LAYA_VERSION}"

COPY src/ ./src/

# Opzionale: scarica i pesi in build (immagine +~1.7 GB, ma nessun download al primo avvio).
# Senza un volume persistente e' quasi sempre la scelta giusta per un deploy.
RUN if [ "$BAKE" = "1" ]; then \
        python src/warmup.py --device cpu --checkpoint multilingual \
        && chown -R laya:laya /models; \
    fi

USER laya

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# `--warm` precarica in background: senza, la prima richiesta paga il load.
# Un solo processo: ogni worker terrebbe una copia dei pesi in memoria.
CMD ["python", "src/server.py", "--warm"]
