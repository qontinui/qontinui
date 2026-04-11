FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime

WORKDIR /app

# System deps for OpenCV and EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies — pinned to recent majors with Blackwell (sm_120)
# kernel support via the base image's CUDA 13.0 runtime.
# --break-system-packages: base image is Python 3.12 with PEP 668
# externally-managed — safe to override inside a container.
RUN pip install --no-cache-dir --break-system-packages \
    ultralytics>=8.3 \
    easyocr>=1.7 \
    "transformers>=4.46,<4.50" \
    "accelerate>=1.0,<1.5" \
    einops>=0.8 \
    timm>=1.0 \
    fastapi>=0.115 \
    uvicorn[standard]>=0.32 \
    python-multipart>=0.0.9 \
    pillow>=11
# transformers upper bound: Florence-2's trust_remote_code files use
# ``config.forced_bos_token_id`` without a default, which raises
# AttributeError on transformers 4.50+. 4.49.x is the last compatible
# release. accelerate is bounded in lockstep for compatibility.

# Pre-download model weights at build time.
# OmniParser V2 YOLO weights: ultralytics' YOLO() needs a local .pt path;
# HF repo IDs are not resolved, so fetch the icon_detect weights file
# explicitly via hf_hub_download.
RUN python -c "import os; from huggingface_hub import hf_hub_download; \
    p = hf_hub_download('microsoft/OmniParser-v2.0', 'icon_detect/model.pt'); \
    os.makedirs('/app/weights', exist_ok=True); \
    import shutil; shutil.copy(p, '/app/weights/icon_detect.pt'); \
    print('YOLO weights at /app/weights/icon_detect.pt')"

# Sanity-load to prime any ultralytics post-download state
RUN python -c "from ultralytics import YOLO; YOLO('/app/weights/icon_detect.pt')"

# Florence-2 captioning model (~500MB)
RUN python -c "from transformers import AutoProcessor, AutoModelForCausalLM; \
    AutoProcessor.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True); \
    AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True)"

# EasyOCR models (~100MB)
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)"

COPY server.py .

# Default the server to the pre-downloaded weights path. Overridable via
# the OMNIPARSER_YOLO_MODEL env var in docker-compose.
ENV OMNIPARSER_YOLO_MODEL=/app/weights/icon_detect.pt

EXPOSE 8080

# Health check every 30s
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

ENTRYPOINT ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
