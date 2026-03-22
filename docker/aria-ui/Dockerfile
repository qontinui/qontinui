FROM vllm/vllm-openai:latest

# Extra deps for Aria multimodal model
RUN pip install --no-cache-dir \
    accelerate \
    sentencepiece \
    torchvision \
    Pillow

# Pre-download model weights at build time (~25GB)
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Aria-UI/Aria-UI-base')"

# Wrapper script that patches LlamaTokenizerFast to expose image_token
COPY serve.py /opt/serve.py

EXPOSE 8100

# For GPUs with <48GB VRAM, reduce: --max-model-len 1024
ENTRYPOINT ["python3", "/opt/serve.py", \
    "--model", "Aria-UI/Aria-UI-base", \
    "--host", "0.0.0.0", \
    "--port", "8100", \
    "--dtype", "bfloat16", \
    "--max-model-len", "4096"]
