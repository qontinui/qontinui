FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps for OpenCV and EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install --no-cache-dir \
    ultralytics>=8.0 \
    easyocr>=1.7 \
    transformers>=4.46 \
    accelerate>=0.25 \
    fastapi>=0.110 \
    uvicorn[standard]>=0.29 \
    python-multipart>=0.0.9 \
    pillow>=11

# Pre-download model weights at build time
# YOLO detection model (~50MB)
RUN python -c "from ultralytics import YOLO; YOLO('microsoft/OmniParser-v2.0')"

# Florence-2 captioning model (~500MB)
RUN python -c "from transformers import AutoProcessor, AutoModelForCausalLM; \
    AutoProcessor.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True); \
    AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True)"

# EasyOCR models (~100MB)
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)"

COPY server.py .

EXPOSE 8080

# Health check every 30s
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

ENTRYPOINT ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
