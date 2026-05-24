# llama-swap model orchestration proxy for qontinui.
#
# Time-shares a single GPU across AriaUI and UI-TARS models.
# llama-swap manages backend processes (Python serving scripts) as children,
# starting/stopping them on demand based on the "model" field in API requests.
#
# Ports:
#   8100 - llama-swap proxy (OpenAI-compatible API)
#
# VRAM usage: peak = largest single model (~13GB AriaUI with NF4, or ~6GB UI-TARS 7B)
# vs. previous: ~31GB when all models loaded simultaneously.
#
# Note: AriaUI and vLLM require different transformers versions.
# AriaUI's custom model code uses LLAMA_ATTENTION_CLASSES (removed in 4.46+).
# Solution: AriaUI runs in a separate virtualenv with transformers==4.45.0.
# vLLM uses the system Python with its own transformers version.

# --- Stage 1: Download llama-swap binary ---
FROM ubuntu:22.04 AS llama-swap-download

ARG LLAMA_SWAP_VERSION=latest

RUN apt-get update && apt-get install -y --no-install-recommends curl jq ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN set -e && \
    VERSION="${LLAMA_SWAP_VERSION}" && \
    if [ "$VERSION" = "latest" ]; then \
        VERSION=$(curl -sL https://api.github.com/repos/mostlygeek/llama-swap/releases/latest | jq -r '.tag_name' | sed 's/^v//'); \
    else \
        VERSION=$(echo "$VERSION" | sed 's/^v//'); \
    fi && \
    echo "Downloading llama-swap v${VERSION}" && \
    mkdir -p /install/bin && \
    curl -sL "https://github.com/mostlygeek/llama-swap/releases/download/v${VERSION}/llama-swap_${VERSION}_linux_amd64.tar.gz" \
        | tar xz -C /install/bin && \
    chmod +x /install/bin/llama-swap && \
    echo "${VERSION}" > /install/llama-swap-version

# --- Stage 2: Runtime with CUDA + Python backends ---
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# Python + system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-venv gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*

# PyTorch nightly with sm_120 support for RTX 5090 (Blackwell).
# Pin to a specific nightly date to avoid pip walking backwards through
# every nightly release when vLLM adds conflicting constraints.
# Using 2026-03-25 which is the latest cu128 nightly at time of writing.
RUN pip install --no-cache-dir \
    torch==2.12.0.dev20260325+cu128 \
    torchvision==0.26.0.dev20260325+cu128 \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# vLLM for UI-TARS backend (system Python, uses its own transformers version).
# Install AFTER torch so vLLM picks up the pre-installed CUDA-enabled torch.
RUN pip install --no-cache-dir vllm

# --- AriaUI virtualenv with pinned transformers ---
# AriaUI's custom model code (modeling_aria.py) imports LLAMA_ATTENTION_CLASSES
# which was removed in transformers 4.46+. Must pin to 4.45.x.
RUN python3 -m venv /opt/venvs/aria-ui --system-site-packages && \
    /opt/venvs/aria-ui/bin/pip install --no-cache-dir \
    transformers==4.45.0 \
    accelerate==0.34.1 \
    sentencepiece \
    bitsandbytes \
    Pillow \
    fastapi \
    "uvicorn[standard]" \
    requests \
    huggingface_hub

# --- PaddleOCR virtualenv ---
# Phase 4 of the UI Bridge Vision Pipeline plan: runner's OcrClient routes
# `model: paddleocr` requests through llama-swap, and docker/paddleocr/serve.py
# satisfies the OpenAI chat-completions contract by wrapping PaddleOCR.
# Isolated venv because paddlepaddle pins incompatible numpy/protobuf versus
# vLLM. CPU-only paddlepaddle wheel — the container already has CUDA from
# the base image, so a GPU variant is a `paddlepaddle-gpu==2.6.1.post120`
# swap when a host needs it.
RUN python3 -m venv /opt/venvs/paddleocr --system-site-packages && \
    /opt/venvs/paddleocr/bin/pip install --no-cache-dir \
    paddlepaddle==2.6.1 \
    paddleocr==2.7.0.3 \
    fastapi==0.115.5 \
    "uvicorn[standard]==0.32.1" \
    Pillow==11.0.0 \
    numpy==1.26.4

# Copy llama-swap binary
COPY --from=llama-swap-download /install/bin/llama-swap /usr/local/bin/llama-swap
COPY --from=llama-swap-download /install/llama-swap-version /etc/llama-swap-version

# --- Stable, expensive layers FIRST: pre-download model weights (~29GB) ---
# Layer-order discipline: these large RUN layers must precede the volatile
# build-context COPYs below. Otherwise editing any serve.py or config.yaml
# invalidates the downstream weight layers and re-pulls all ~29GB on rebuild.
# Pre-download AriaUI model weights (~25GB)
RUN /opt/venvs/aria-ui/bin/python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Aria-UI/Aria-UI-base')"

# Pre-download UI-TARS 2B weights (~4GB)
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('ByteDance-Seed/UI-TARS-2B-SFT')"

# Uncomment to pre-download additional models:
# RUN /opt/venvs/aria-ui/bin/python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Aria-UI/Aria-UI-context-aware')"
# RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('ByteDance-Seed/UI-TARS-7B-SFT')"
# RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('ByteDance-Seed/UI-TARS-1.5-7B')"

# --- Volatile layers LAST: backend scripts + config change on every repo edit ---
# Build context is docker/ (parent dir), set in docker-compose.yml. Kept after
# the weight downloads so a code/config edit rebuilds only these cheap layers,
# never the 29GB of model weights above.
COPY aria-ui/serve.py /opt/backends/aria-ui/serve.py
COPY paddleocr/serve.py /opt/backends/paddleocr/serve.py
COPY llama-swap/config.yaml /etc/llama-swap/config.yaml

EXPOSE 8100

# llama-swap listens on 8100 and manages all backend processes
ENTRYPOINT ["llama-swap", \
    "--config", "/etc/llama-swap/config.yaml", \
    "--listen", "0.0.0.0:8100"]
