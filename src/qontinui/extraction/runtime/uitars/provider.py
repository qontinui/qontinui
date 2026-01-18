"""UI-TARS inference providers.

This module implements three providers for UI-TARS inference:
- HuggingFaceEndpointProvider: Cloud inference via HuggingFace Inference Endpoints
- LocalTransformersProvider: Local inference with transformers + quantization
- VLLMProvider: Local inference via running vLLM server

All providers parse UI-TARS output into Thought-Action format.
"""

from __future__ import annotations

import base64
import io
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from .config import UITARSSettings
from .models import (
    UITARSAction,
    UITARSActionType,
    UITARSInferenceRequest,
    UITARSInferenceResult,
    UITARSThought,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class UITARSProviderBase(ABC):
    """Abstract base class for UI-TARS inference providers."""

    def __init__(self, settings: UITARSSettings) -> None:
        """Initialize provider with settings.

        Args:
            settings: UI-TARS configuration settings
        """
        self.settings = settings
        self._initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the provider (load model, connect, etc.)."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and ready."""
        pass

    @abstractmethod
    def infer(self, request: UITARSInferenceRequest) -> UITARSInferenceResult:
        """Perform inference on a screenshot with a prompt.

        Args:
            request: Inference request with image and prompt

        Returns:
            Inference result with parsed thought and action
        """
        pass

    def parse_output(self, raw_output: str) -> tuple[UITARSThought, UITARSAction]:
        """Parse UI-TARS raw output into Thought and Action.

        UI-TARS outputs in format:
            Thought: <reasoning about the UI and task>
            Action: <action_type>(param1, param2, ...)

        Args:
            raw_output: Raw text output from the model

        Returns:
            Tuple of (UITARSThought, UITARSAction)
        """
        thought = UITARSThought(reasoning="")
        action = UITARSAction(action_type=UITARSActionType.WAIT, raw_output=raw_output)

        # Parse Thought
        thought_match = re.search(
            r"Thought:\s*(.+?)(?=Action:|$)", raw_output, re.DOTALL | re.IGNORECASE
        )
        if thought_match:
            thought.reasoning = thought_match.group(1).strip()

        # Parse Action
        action_match = re.search(
            r"Action:\s*(\w+)\(([^)]*)\)", raw_output, re.IGNORECASE
        )
        if action_match:
            action_type_str = action_match.group(1).lower()
            params_str = action_match.group(2).strip()

            # Map action type
            action_type_map = {
                "click": UITARSActionType.CLICK,
                "double_click": UITARSActionType.DOUBLE_CLICK,
                "right_click": UITARSActionType.RIGHT_CLICK,
                "type": UITARSActionType.TYPE,
                "scroll": UITARSActionType.SCROLL,
                "hover": UITARSActionType.HOVER,
                "drag": UITARSActionType.DRAG,
                "hotkey": UITARSActionType.HOTKEY,
                "wait": UITARSActionType.WAIT,
                "done": UITARSActionType.DONE,
            }
            action.action_type = action_type_map.get(
                action_type_str, UITARSActionType.WAIT
            )

            # Parse parameters based on action type
            action = self._parse_action_params(action, params_str)

        return thought, action

    def _parse_action_params(self, action: UITARSAction, params_str: str) -> UITARSAction:
        """Parse action parameters from string.

        Args:
            action: Action object to populate
            params_str: Comma-separated parameters string

        Returns:
            Action with parsed parameters
        """
        params = [p.strip().strip("'\"") for p in params_str.split(",") if p.strip()]

        if action.action_type in (
            UITARSActionType.CLICK,
            UITARSActionType.DOUBLE_CLICK,
            UITARSActionType.RIGHT_CLICK,
            UITARSActionType.HOVER,
        ):
            # click(x, y)
            if len(params) >= 2:
                try:
                    action.x = int(float(params[0]))
                    action.y = int(float(params[1]))
                except (ValueError, IndexError):
                    pass

        elif action.action_type == UITARSActionType.TYPE:
            # type(text) or type(x, y, text)
            if len(params) >= 3:
                try:
                    action.x = int(float(params[0]))
                    action.y = int(float(params[1]))
                    action.text = params[2]
                except (ValueError, IndexError):
                    action.text = params[0] if params else None
            elif len(params) >= 1:
                action.text = params[0]

        elif action.action_type == UITARSActionType.SCROLL:
            # scroll(direction) or scroll(x, y, direction, amount)
            if len(params) >= 4:
                try:
                    action.x = int(float(params[0]))
                    action.y = int(float(params[1]))
                    action.scroll_direction = params[2].lower()
                    action.scroll_amount = int(float(params[3]))
                except (ValueError, IndexError):
                    pass
            elif len(params) >= 1:
                action.scroll_direction = params[0].lower()
                action.scroll_amount = int(params[1]) if len(params) > 1 else 100

        elif action.action_type == UITARSActionType.DRAG:
            # drag(start_x, start_y, end_x, end_y)
            if len(params) >= 4:
                try:
                    action.x = int(float(params[0]))
                    action.y = int(float(params[1]))
                    action.end_x = int(float(params[2]))
                    action.end_y = int(float(params[3]))
                except (ValueError, IndexError):
                    pass

        elif action.action_type == UITARSActionType.HOTKEY:
            # hotkey(key1, key2, ...)
            action.keys = params

        elif action.action_type == UITARSActionType.WAIT:
            # wait(duration)
            if params:
                try:
                    action.duration = float(params[0])
                except (ValueError, IndexError):
                    action.duration = 1.0

        return action

    def _image_to_base64(self, image: np.ndarray[Any, Any]) -> str:
        """Convert numpy image to base64 string.

        Args:
            image: RGB image as numpy array

        Returns:
            Base64-encoded PNG string
        """
        pil_image = Image.fromarray(image.astype(np.uint8))
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


class HuggingFaceEndpointProvider(UITARSProviderBase):
    """Provider for HuggingFace Inference Endpoints.

    Uses HTTP API to communicate with a deployed HuggingFace endpoint.
    Supports any model size (2B, 7B, 72B) depending on endpoint configuration.
    """

    def __init__(self, settings: UITARSSettings) -> None:
        super().__init__(settings)
        self._client: Any = None

    def initialize(self) -> None:
        """Initialize HTTP client for HuggingFace endpoint."""
        try:
            import httpx

            self._client = httpx.Client(
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=self.settings.uitars_execution_timeout,
                    write=10.0,
                    pool=5.0,
                ),
                headers=(
                    {"Authorization": f"Bearer {self.settings.huggingface_api_token}"}
                    if self.settings.huggingface_api_token
                    else {}
                ),
            )
            self._initialized = True
            logger.info("HuggingFace endpoint provider initialized")
        except ImportError:
            logger.error("httpx not installed. Run: pip install httpx")
            raise

    def is_available(self) -> bool:
        """Check if HuggingFace endpoint is configured and reachable."""
        if not self.settings.huggingface_endpoint:
            return False
        if not self._initialized:
            return False

        try:
            # Quick health check
            response = self._client.get(
                self.settings.huggingface_endpoint.rstrip("/") + "/health",
                timeout=5.0,
            )
            return bool(response.status_code == 200)
        except Exception:
            return False

    def infer(self, request: UITARSInferenceRequest) -> UITARSInferenceResult:
        """Perform inference via HuggingFace endpoint.

        Args:
            request: Inference request with image and prompt

        Returns:
            Parsed inference result
        """
        if not self._initialized:
            self.initialize()

        start_time = time.time()

        # Convert image to base64
        image_b64 = self._image_to_base64(request.image)

        # Build request payload (HuggingFace TGI format)
        payload = {
            "inputs": request.prompt,
            "parameters": {
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature if request.temperature > 0 else None,
                "do_sample": request.temperature > 0,
            },
            "image": image_b64,
        }

        # Add system prompt if provided
        if request.system_prompt or self.settings.system_prompt:
            payload["system_prompt"] = request.system_prompt or self.settings.system_prompt

        try:
            response = self._client.post(
                self.settings.huggingface_endpoint,
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            # Extract generated text
            raw_output = result.get("generated_text", "") or result.get("text", "")
            if isinstance(result, list) and result:
                raw_output = result[0].get("generated_text", "")

            inference_time = (time.time() - start_time) * 1000
            thought, action = self.parse_output(raw_output)

            return UITARSInferenceResult(
                thought=thought,
                action=action,
                raw_output=raw_output,
                inference_time_ms=inference_time,
                model_name=self.settings.get_model_id(),
                provider="huggingface_endpoint",
            )

        except Exception as e:
            logger.error(f"HuggingFace inference failed: {e}")
            return UITARSInferenceResult(
                thought=UITARSThought(reasoning=f"Inference failed: {e}"),
                action=UITARSAction(action_type=UITARSActionType.WAIT),
                raw_output="",
                inference_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e),
            )


class LocalTransformersProvider(UITARSProviderBase):
    """Provider for local inference using transformers.

    Supports quantization (int4, int8) for running on consumer GPUs.
    Recommended for GTX 1080 (8GB): 2B model with int4 quantization.
    """

    def __init__(self, settings: UITARSSettings) -> None:
        super().__init__(settings)
        self._model: Any = None
        self._processor: Any = None
        self._device: Any = None

    def initialize(self) -> None:
        """Load model and processor with optional quantization."""
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor

            model_id = self.settings.get_model_id()
            logger.info(f"Loading UI-TARS model: {model_id}")

            # Determine device
            if self.settings.device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.settings.device

            # Determine dtype
            if self.settings.torch_dtype == "auto":
                torch_dtype = torch.float16 if self._device == "cuda" else torch.float32
            elif self.settings.torch_dtype == "bfloat16":
                torch_dtype = torch.bfloat16
            elif self.settings.torch_dtype == "float16":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32

            # Load processor
            self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

            # Load model with quantization if configured
            if self.settings.quantization == "int4":
                try:
                    from transformers import BitsAndBytesConfig

                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch_dtype,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    self._model = AutoModelForVision2Seq.from_pretrained(
                        model_id,
                        quantization_config=quantization_config,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                except ImportError:
                    logger.warning(
                        "bitsandbytes not installed, falling back to float16. "
                        "Install with: pip install bitsandbytes"
                    )
                    self._model = AutoModelForVision2Seq.from_pretrained(
                        model_id,
                        torch_dtype=torch_dtype,
                        device_map="auto",
                        trust_remote_code=True,
                    )

            elif self.settings.quantization == "int8":
                try:
                    from transformers import BitsAndBytesConfig

                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    self._model = AutoModelForVision2Seq.from_pretrained(
                        model_id,
                        quantization_config=quantization_config,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                except ImportError:
                    logger.warning("bitsandbytes not installed, falling back to float16")
                    self._model = AutoModelForVision2Seq.from_pretrained(
                        model_id,
                        torch_dtype=torch_dtype,
                        device_map="auto",
                        trust_remote_code=True,
                    )
            else:
                # No quantization
                self._model = AutoModelForVision2Seq.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    device_map="auto" if self._device == "cuda" else None,
                    trust_remote_code=True,
                )
                if self._device != "cuda":
                    self._model = self._model.to(self._device)

            self._initialized = True
            logger.info(f"UI-TARS model loaded on {self._device}")

        except Exception as e:
            logger.error(f"Failed to initialize LocalTransformersProvider: {e}")
            raise

    def is_available(self) -> bool:
        """Check if model is loaded and ready."""
        return self._initialized and self._model is not None

    def infer(self, request: UITARSInferenceRequest) -> UITARSInferenceResult:
        """Perform local inference.

        Args:
            request: Inference request with image and prompt

        Returns:
            Parsed inference result
        """
        if not self._initialized:
            self.initialize()

        import torch

        start_time = time.time()

        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(request.image.astype(np.uint8))

            # Build conversation (content can be str or list of dicts for multimodal)
            messages: list[dict[str, Any]] = []
            if request.system_prompt or self.settings.system_prompt:
                messages.append({
                    "role": "system",
                    "content": request.system_prompt or self.settings.system_prompt,
                })

            # Add history if present
            if request.history:
                for action_text, observation in request.history:
                    messages.append({"role": "assistant", "content": action_text})
                    messages.append({"role": "user", "content": observation})

            # Add current request
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": request.prompt},
                ],
            })

            # Process inputs
            inputs = self._processor(
                messages,
                return_tensors="pt",
                padding=True,
            )

            # Move to device
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature if request.temperature > 0 else None,
                    do_sample=request.temperature > 0,
                    top_p=self.settings.top_p if request.temperature > 0 else None,
                    pad_token_id=self._processor.tokenizer.eos_token_id,
                )

            # Decode output
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            raw_output = self._processor.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

            inference_time = (time.time() - start_time) * 1000
            thought, action = self.parse_output(raw_output)

            return UITARSInferenceResult(
                thought=thought,
                action=action,
                raw_output=raw_output,
                inference_time_ms=inference_time,
                tokens_used=len(generated_ids),
                model_name=self.settings.get_model_id(),
                provider="local_transformers",
            )

        except Exception as e:
            logger.error(f"Local inference failed: {e}")
            return UITARSInferenceResult(
                thought=UITARSThought(reasoning=f"Inference failed: {e}"),
                action=UITARSAction(action_type=UITARSActionType.WAIT),
                raw_output="",
                inference_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e),
            )


class VLLMProvider(UITARSProviderBase):
    """Provider for vLLM server inference.

    Communicates with a running vLLM server via OpenAI-compatible API.
    Requires vLLM server to be running with UI-TARS model loaded.

    Start server with:
        vllm serve ByteDance-Seed/UI-TARS-2B-SFT --trust-remote-code
    """

    def __init__(self, settings: UITARSSettings) -> None:
        super().__init__(settings)
        self._client: Any = None

    def initialize(self) -> None:
        """Initialize HTTP client for vLLM server."""
        try:
            import httpx

            self._client = httpx.Client(
                base_url=self.settings.vllm_server_url,
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=self.settings.uitars_execution_timeout,
                    write=10.0,
                    pool=5.0,
                ),
            )
            self._initialized = True
            logger.info(f"vLLM provider initialized: {self.settings.vllm_server_url}")
        except ImportError:
            logger.error("httpx not installed. Run: pip install httpx")
            raise

    def is_available(self) -> bool:
        """Check if vLLM server is running and responsive."""
        if not self._initialized:
            return False

        try:
            response = self._client.get("/health", timeout=5.0)
            return bool(response.status_code == 200)
        except Exception:
            return False

    def infer(self, request: UITARSInferenceRequest) -> UITARSInferenceResult:
        """Perform inference via vLLM server.

        Args:
            request: Inference request with image and prompt

        Returns:
            Parsed inference result
        """
        if not self._initialized:
            self.initialize()

        start_time = time.time()

        # Convert image to base64
        image_b64 = self._image_to_base64(request.image)

        # Build messages for chat completion (content can be str or list for multimodal)
        messages: list[dict[str, Any]] = []
        if request.system_prompt or self.settings.system_prompt:
            messages.append({
                "role": "system",
                "content": request.system_prompt or self.settings.system_prompt,
            })

        # Add current request with image
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text", "text": request.prompt},
            ],
        })

        # Model name for vLLM
        model_name = self.settings.vllm_model_name or self.settings.get_model_id()

        # Build request payload (OpenAI-compatible)
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": request.max_new_tokens,
            "temperature": request.temperature,
        }

        try:
            response = self._client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            result = response.json()

            # Extract generated text
            raw_output = result["choices"][0]["message"]["content"]
            tokens_used = result.get("usage", {}).get("completion_tokens", 0)

            inference_time = (time.time() - start_time) * 1000
            thought, action = self.parse_output(raw_output)

            return UITARSInferenceResult(
                thought=thought,
                action=action,
                raw_output=raw_output,
                inference_time_ms=inference_time,
                tokens_used=tokens_used,
                model_name=model_name,
                provider="vllm",
            )

        except Exception as e:
            logger.error(f"vLLM inference failed: {e}")
            return UITARSInferenceResult(
                thought=UITARSThought(reasoning=f"Inference failed: {e}"),
                action=UITARSAction(action_type=UITARSActionType.WAIT),
                raw_output="",
                inference_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e),
            )


def create_provider(settings: UITARSSettings) -> UITARSProviderBase:
    """Factory function to create the appropriate provider.

    Args:
        settings: UI-TARS configuration settings

    Returns:
        Initialized provider instance

    Raises:
        ValueError: If provider type is invalid
    """
    provider_map: dict[str, type[UITARSProviderBase]] = {
        "cloud": HuggingFaceEndpointProvider,
        "local_transformers": LocalTransformersProvider,
        "local_vllm": VLLMProvider,
    }

    provider_class = provider_map.get(settings.provider)
    if not provider_class:
        raise ValueError(f"Unknown provider: {settings.provider}")

    return provider_class(settings)
