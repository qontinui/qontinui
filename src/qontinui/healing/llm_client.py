"""LLM client interface and implementations for vision-based healing.

Provides abstract interface and concrete implementations for:
- Disabled (default): No LLM, returns None
- Local: Ollama for local inference
- Remote: OpenAI, Anthropic, Google for cloud inference
"""

import base64
import logging
import re
from abc import ABC, abstractmethod

from .healing_types import ElementLocation, HealingContext

logger = logging.getLogger(__name__)


class VisionLLMClient(ABC):
    """Abstract interface for vision-capable LLM clients.

    Implementations must provide find_element() which takes a screenshot
    and description, returning the element's location if found.
    """

    @abstractmethod
    def find_element(
        self,
        screenshot: bytes,
        context: HealingContext,
    ) -> ElementLocation | None:
        """Ask LLM to locate element in screenshot.

        Args:
            screenshot: PNG image bytes of current screen.
            context: Healing context with element description.

        Returns:
            ElementLocation if found, None otherwise.
        """
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM client is available and ready."""
        pass

    def _build_prompt(self, context: HealingContext) -> str:
        """Build prompt for element location.

        Args:
            context: Healing context.

        Returns:
            Prompt string for LLM.
        """
        prompt = f"""Look at this screenshot and find the UI element described below.

Element to find: {context.original_description}
"""
        if context.action_type:
            prompt += f"Action intended: {context.action_type}\n"

        if context.failure_reason:
            prompt += f"Previous lookup failed because: {context.failure_reason}\n"

        prompt += """
Return ONLY the coordinates of the element's center in this exact format:
COORDINATES: x,y

For example: COORDINATES: 450,320

If you cannot find the element, respond with:
NOT_FOUND: reason why

Do not include any other text in your response."""

        return prompt

    def _parse_response(self, response: str) -> ElementLocation | None:
        """Parse LLM response to extract coordinates.

        Args:
            response: Raw LLM response text.

        Returns:
            ElementLocation if coordinates found, None otherwise.
        """
        response = response.strip()

        # Check for NOT_FOUND
        if response.upper().startswith("NOT_FOUND"):
            logger.debug(f"LLM reported element not found: {response}")
            return None

        # Look for COORDINATES pattern
        coord_pattern = r"COORDINATES:\s*(\d+)\s*,\s*(\d+)"
        match = re.search(coord_pattern, response, re.IGNORECASE)

        if match:
            x = int(match.group(1))
            y = int(match.group(2))
            return ElementLocation(x=x, y=y, confidence=0.8)

        # Try simple x,y pattern as fallback
        simple_pattern = r"^(\d+)\s*,\s*(\d+)$"
        match = re.match(simple_pattern, response)

        if match:
            x = int(match.group(1))
            y = int(match.group(2))
            return ElementLocation(x=x, y=y, confidence=0.7)

        logger.warning(f"Could not parse coordinates from response: {response[:100]}")
        return None


class DisabledVisionClient(VisionLLMClient):
    """Default client when LLM healing is disabled.

    Always returns None, triggering fallback to mechanical retry.
    """

    def find_element(
        self,
        screenshot: bytes,
        context: HealingContext,
    ) -> ElementLocation | None:
        """Always returns None (LLM disabled).

        Args:
            screenshot: Ignored.
            context: Ignored.

        Returns:
            Always None.
        """
        logger.debug("LLM healing disabled, returning None")
        return None

    @property
    def is_available(self) -> bool:
        """Disabled client is always 'available' (does nothing)."""
        return True


class LocalVisionClient(VisionLLMClient):
    """Local vision LLM via Ollama.

    Requires Ollama to be running locally with a vision model installed.
    No internet required after initial model download.

    Example:
        # Install Ollama and pull a vision model
        # ollama pull llava:7b

        client = LocalVisionClient(model_name="llava:7b")
        location = client.find_element(screenshot_bytes, context)
    """

    def __init__(
        self,
        model_name: str = "llava:7b",
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 30.0,
    ) -> None:
        """Initialize local Ollama client.

        Args:
            model_name: Ollama model name (e.g., llava:7b, bakllava).
            base_url: Ollama API base URL.
            timeout_seconds: Request timeout.
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def find_element(
        self,
        screenshot: bytes,
        context: HealingContext,
    ) -> ElementLocation | None:
        """Find element using Ollama vision model.

        Args:
            screenshot: PNG image bytes.
            context: Healing context.

        Returns:
            ElementLocation if found, None otherwise.
        """
        try:
            import httpx
        except ImportError:
            logger.error("httpx not installed. Run: pip install httpx")
            return None

        prompt = self._build_prompt(context)
        image_b64 = base64.b64encode(screenshot).decode("utf-8")

        # Ollama generate API with images
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        }

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            response_text = data.get("response", "")
            logger.debug(f"Ollama response: {response_text[:200]}")

            return self._parse_response(response_text)

        except httpx.HTTPError as e:
            logger.error(f"Ollama API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error calling Ollama: {e}")
            return None

    @property
    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            import httpx

            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                if response.status_code != 200:
                    return False

                data = response.json()
                models = [m.get("name", "") for m in data.get("models", [])]

                # Check if our model is available
                for model in models:
                    if model.startswith(self.model_name.split(":")[0]):
                        return True

                logger.warning(f"Model {self.model_name} not found. Available: {models}")
                return False

        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            return False


class RemoteVisionClient(VisionLLMClient):
    """Remote vision LLM via cloud APIs.

    Supports OpenAI, Anthropic, and Google providers.
    Requires API key and internet connection.

    Example:
        client = RemoteVisionClient(
            provider="openai",
            api_key=os.environ["OPENAI_API_KEY"],
            model="gpt-4o",
        )
        location = client.find_element(screenshot_bytes, context)
    """

    # Default models per provider
    DEFAULT_MODELS = {
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4-20250514",
        "google": "gemini-1.5-flash",
    }

    def __init__(
        self,
        provider: str,
        api_key: str,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Initialize remote API client.

        Args:
            provider: Provider name (openai, anthropic, google).
            api_key: API key for the provider.
            model: Model name (uses default if not specified).
            base_url: Optional base URL override.
            timeout_seconds: Request timeout.
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODELS.get(self.provider)
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

        if not self.model:
            raise ValueError(f"Unknown provider: {provider}")

    def find_element(
        self,
        screenshot: bytes,
        context: HealingContext,
    ) -> ElementLocation | None:
        """Find element using remote vision API.

        Args:
            screenshot: PNG image bytes.
            context: Healing context.

        Returns:
            ElementLocation if found, None otherwise.
        """
        if self.provider == "openai":
            return self._call_openai(screenshot, context)
        elif self.provider == "anthropic":
            return self._call_anthropic(screenshot, context)
        elif self.provider == "google":
            return self._call_google(screenshot, context)
        else:
            logger.error(f"Unknown provider: {self.provider}")
            return None

    def _call_openai(
        self,
        screenshot: bytes,
        context: HealingContext,
    ) -> ElementLocation | None:
        """Call OpenAI Vision API."""
        try:
            import httpx
        except ImportError:
            logger.error("httpx not installed. Run: pip install httpx")
            return None

        prompt = self._build_prompt(context)
        image_b64 = base64.b64encode(screenshot).decode("utf-8")

        base_url = self.base_url or "https://api.openai.com/v1"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 100,
        }

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            response_text = data["choices"][0]["message"]["content"]
            logger.debug(f"OpenAI response: {response_text}")

            return self._parse_response(response_text)

        except httpx.HTTPError as e:
            logger.error(f"OpenAI API error: {e}")
            return None
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected OpenAI response format: {e}")
            return None

    def _call_anthropic(
        self,
        screenshot: bytes,
        context: HealingContext,
    ) -> ElementLocation | None:
        """Call Anthropic Vision API."""
        try:
            import httpx
        except ImportError:
            logger.error("httpx not installed. Run: pip install httpx")
            return None

        prompt = self._build_prompt(context)
        image_b64 = base64.b64encode(screenshot).decode("utf-8")

        base_url = self.base_url or "https://api.anthropic.com/v1"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(
                    f"{base_url}/messages",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            response_text = data["content"][0]["text"]
            logger.debug(f"Anthropic response: {response_text}")

            return self._parse_response(response_text)

        except httpx.HTTPError as e:
            logger.error(f"Anthropic API error: {e}")
            return None
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected Anthropic response format: {e}")
            return None

    def _call_google(
        self,
        screenshot: bytes,
        context: HealingContext,
    ) -> ElementLocation | None:
        """Call Google Gemini Vision API."""
        try:
            import httpx
        except ImportError:
            logger.error("httpx not installed. Run: pip install httpx")
            return None

        prompt = self._build_prompt(context)
        image_b64 = base64.b64encode(screenshot).decode("utf-8")

        base_url = self.base_url or "https://generativelanguage.googleapis.com/v1beta"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": image_b64,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 100,
            },
        }

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(
                    f"{base_url}/models/{self.model}:generateContent",
                    params={"key": self.api_key},
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            response_text = data["candidates"][0]["content"]["parts"][0]["text"]
            logger.debug(f"Google response: {response_text}")

            return self._parse_response(response_text)

        except httpx.HTTPError as e:
            logger.error(f"Google API error: {e}")
            return None
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected Google response format: {e}")
            return None

    @property
    def is_available(self) -> bool:
        """Check if API key is configured (doesn't verify it works)."""
        return bool(self.api_key)
