# PaddleOCR OpenAI-API shim container, mounted under llama-swap.
# Bound to the same backend slot pattern as aria-ui/uitars so callers
# can hit POST /v1/chat/completions with `"model": "paddleocr"` and the
# proxy dispatches here.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# PaddlePaddle wheels on Debian-slim need a few system libs (libgomp1 for
# the openmp runtime, libglib2.0-0 for opencv-python-headless, libgl1 for
# image decoding). These are tiny additions.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/paddleocr

# CPU-only paddlepaddle is the conservative default; GPU callers should
# rebuild this image with paddlepaddle-gpu pinned to their CUDA. Image
# stays smaller + warm-start faster on CPU.
RUN pip install --no-cache-dir \
        paddlepaddle==2.6.1 \
        paddleocr==2.7.0.3 \
        fastapi==0.115.5 \
        uvicorn==0.32.1 \
        pillow==11.0.0 \
        numpy==1.26.4

COPY serve.py /opt/paddleocr/serve.py

# llama-swap fills ${PORT} dynamically (startPort: 5800 increments per
# launched model); the wrapper here listens on whichever port the
# orchestrator passes.
ENV PORT=8200
EXPOSE 8200

CMD ["python", "/opt/paddleocr/serve.py", "--host", "0.0.0.0", "--port", "8200"]
