"""Aria-UI grounding client via OpenAI-compatible API.

Aria-UI is a 25B MoE vision model (3.9B activated) that performs
element grounding on screenshots. It returns normalized [x, y]
coordinates in 0-1000 scale.

Requires a running Aria-UI server (see docker/aria-ui/).
"""

import base64
import logging
import re

from .healing_types import ElementLocation, HealingContext
from .llm_client import VisionLLMClient

logger = logging.getLogger(__name__)

# Aria-UI's canonical prompt template
_ARIA_UI_PROMPT = (
    "Given a GUI image, what are the relative (0-1000) pixel point "
    "coordinates for the element corresponding to the following "
    "instruction or description: {description}"
)


class AriaUIClient(VisionLLMClient):
    """Aria-UI grounding via HTTP API.

    Sends screenshots to the Aria-UI server and parses
    the returned [x, y] coordinates (0-1000 normalized scale).

    Example:
        client = AriaUIClient("http://localhost:8100")
        if client.is_available:
            location = client.find_element(screenshot_bytes, context)
    """

    # Model identifier on the server
    MODEL_NAME = "Aria-UI/Aria-UI-base"

    def __init__(
        self,
        endpoint: str = "http://localhost:8100",
        timeout: float = 120.0,
        model: str | None = None,
    ) -> None:
        """Initialize Aria-UI client.

        Args:
            endpoint: Base URL of the vLLM server.
            timeout: HTTP request timeout in seconds.
            model: Model name override (defaults to Aria-UI/Aria-UI-base).
        """
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout
        self._model = model or self.MODEL_NAME

    def find_element(
        self,
        screenshot: bytes,
        context: HealingContext,
    ) -> ElementLocation | None:
        """Locate an element in a screenshot using Aria-UI.

        Args:
            screenshot: PNG image bytes of current screen.
            context: Healing context with element description.

        Returns:
            ElementLocation with absolute pixel coordinates, or None.
        """
        try:
            import httpx
        except ImportError:
            logger.error("httpx not installed. Run: pip install httpx")
            return None

        image_b64 = base64.b64encode(screenshot).decode("utf-8")
        prompt = _ARIA_UI_PROMPT.format(description=context.original_description)

        payload = {
            "model": self._model,
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
            "max_tokens": 50,
            "temperature": 0.0,
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    f"{self._endpoint}/v1/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            response_text = data["choices"][0]["message"]["content"]
            logger.debug(f"Aria-UI response: {response_text}")

            return self._parse_aria_response(response_text, context)

        except httpx.HTTPError as e:
            logger.error(f"Aria-UI API error: {e}")
            return None
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected Aria-UI response format: {e}")
            return None

    def _parse_aria_response(
        self,
        response: str,
        context: HealingContext,
    ) -> ElementLocation | None:
        """Parse Aria-UI [x, y] response and convert to absolute coordinates.

        Aria-UI returns coordinates in 0-1000 normalized scale.
        We convert to absolute pixels using screenshot_shape from context.

        Args:
            response: Raw model response text (e.g., "[523, 187]").
            context: Healing context (needs screenshot_shape for conversion).

        Returns:
            ElementLocation with absolute coordinates, or None.
        """
        response = response.strip()

        # Try to extract [x, y] pattern
        coord = self._extract_coordinates(response)
        if coord is None:
            logger.warning(
                f"Could not parse Aria-UI coordinates from: {response[:100]}"
            )
            return None

        norm_x, norm_y = coord

        # Validate range
        if not (0 <= norm_x <= 1000 and 0 <= norm_y <= 1000):
            logger.warning(f"Aria-UI coordinates out of range: [{norm_x}, {norm_y}]")
            return None

        # Convert from 0-1000 scale to absolute pixels
        if context.screenshot_shape:
            screen_height, screen_width = context.screenshot_shape
        else:
            # Fallback: assume 1080p if shape not provided
            logger.debug("No screenshot_shape in context, assuming 1920x1080")
            screen_width, screen_height = 1920, 1080

        abs_x = int(norm_x * screen_width / 1000)
        abs_y = int(norm_y * screen_height / 1000)

        return ElementLocation(
            x=abs_x,
            y=abs_y,
            confidence=1.0,
            description=f"Aria-UI grounding [{norm_x}, {norm_y}] -> ({abs_x}, {abs_y})",
        )

    @staticmethod
    def _extract_coordinates(text: str) -> tuple[int, int] | None:
        """Extract [x, y] coordinates from response text.

        Handles formats like:
        - [523, 187]
        - (523, 187)
        - 523, 187

        Args:
            text: Raw response text.

        Returns:
            (x, y) tuple or None.
        """
        match = re.search(r"[\[\(]?\s*(\d+)\s*,\s*(\d+)\s*[\]\)]?", text)
        if match:
            return (int(match.group(1)), int(match.group(2)))

        return None

    @property
    def is_available(self) -> bool:
        """Check if the vLLM server is healthy."""
        try:
            import httpx

            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self._endpoint}/health")
                return response.status_code == 200
        except Exception as e:
            logger.debug(f"Aria-UI server not available: {e}")
            return False
