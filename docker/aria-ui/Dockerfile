FROM vllm/vllm-openai:latest

# Pre-download Aria-UI model weights at build time
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Aria-UI/Aria-UI-base')"
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Aria-UI/Aria-UI-context-aware')"

EXPOSE 8100

# Default: serve base model. Override CMD for context-aware variant:
#   --model Aria-UI/Aria-UI-context-aware
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "Aria-UI/Aria-UI-base", \
     "--port", "8100", \
     "--dtype", "bfloat16"]
