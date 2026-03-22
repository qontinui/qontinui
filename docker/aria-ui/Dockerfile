FROM vllm/vllm-openai:latest

# Aria-UI extra dependencies (don't pin transformers — vllm needs its own version)
RUN pip install --no-cache-dir \
    accelerate \
    sentencepiece \
    torchvision \
    Pillow

# Pre-download model weights at build time (~25GB)
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Aria-UI/Aria-UI-base')"

EXPOSE 8100

# Required flags:
#   --trust-remote-code   Model uses custom HuggingFace code
#   --tokenizer-mode slow Fast tokenizer not supported
#   --max-model-len 4096  Model's max_position_embeddings
#
# For GPUs with <48GB VRAM, add: --quantization awq  (or bitsandbytes)
# or reduce: --max-model-len 1024
ENTRYPOINT ["vllm", "serve", "Aria-UI/Aria-UI-base", \
    "--host", "0.0.0.0", \
    "--port", "8100", \
    "--dtype", "bfloat16", \
    "--trust-remote-code", \
    "--tokenizer-mode", "slow", \
    "--max-model-len", "4096"]
