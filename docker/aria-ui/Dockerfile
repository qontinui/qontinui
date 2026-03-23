FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# Python + system deps (gcc needed by bitsandbytes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# PyTorch nightly with sm_120 support for RTX 5090 (Blackwell)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

RUN pip install --no-cache-dir \
    transformers==4.45.0 \
    accelerate==0.34.1 \
    sentencepiece \
    bitsandbytes \
    Pillow \
    fastapi \
    uvicorn[standard] \
    requests \
    huggingface_hub

# Pre-download model weights at build time (~25GB each)
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Aria-UI/Aria-UI-base')"
# Uncomment to pre-download context-aware variant (~25GB additional):
# RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Aria-UI/Aria-UI-context-aware')"

# v2: fixed missing return in load_model
COPY serve.py /opt/serve.py

EXPOSE 8100

# Loads with 4-bit NF4 quantization (~13GB VRAM) via bitsandbytes
ENTRYPOINT ["python3", "/opt/serve.py", \
    "--model", "Aria-UI/Aria-UI-base", \
    "--host", "0.0.0.0", \
    "--port", "8100"]
