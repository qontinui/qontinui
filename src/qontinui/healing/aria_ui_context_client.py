"""Context-aware Aria-UI client that uses action history for disambiguation.

Uses the Aria-UI-context-aware model variant which accepts interleaved
(previous_screenshot, action_description) pairs as multi-turn conversation
history, enabling disambiguation of ambiguous elements across workflow steps.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from .aria_ui_client import _ARIA_UI_PROMPT, AriaUIClient
from .healing_types import ElementLocation, HealingContext

logger = logging.getLogger(__name__)


class AriaUIContextClient(AriaUIClient):
    """Context-aware Aria-UI that uses action history for disambiguation.

    Extends AriaUIClient with multi-turn conversation support. Previous
    screenshots and action descriptions are sent as conversation history
    so the model can disambiguate elements that are identical without context.

    Example:
        client = AriaUIContextClient("http://localhost:8100")
        history = [
            (prev_screenshot_bytes, "Clicked 'File' menu"),
            (prev_screenshot_bytes_2, "Clicked 'Save As'"),
        ]
        location = client.find_element_with_history(
            screenshot_bytes, context, history
        )
    """

    MODEL_NAME = "Aria-UI/Aria-UI-context-aware"

    def __init__(
        self,
        endpoint: str = "http://localhost:8100",
        timeout: float = 10.0,
        max_history: int = 3,
        model: str | None = None,
    ) -> None:
        """Initialize context-aware Aria-UI client.

        Args:
            endpoint: Base URL of the vLLM server.
            timeout: HTTP request timeout in seconds.
            max_history: Maximum number of history entries to include.
            model: Model name override.
        """
        super().__init__(endpoint=endpoint, timeout=timeout, model=model or self.MODEL_NAME)
        self._max_history = max_history

    def find_element(
        self,
        screenshot: bytes,
        context: HealingContext,
    ) -> ElementLocation | None:
        """Locate element, using action history from context if available.

        When called through the standard VisionHealer pipeline, action
        history is read from ``context.additional_context["action_history"]``.
        If no history is present, falls back to stateless base grounding.

        Args:
            screenshot: PNG image bytes of current screen.
            context: Healing context. If ``additional_context`` contains
                an ``"action_history"`` key with a list of
                ``(screenshot_bytes, action_description)`` tuples, context-
                aware grounding is used.

        Returns:
            ElementLocation with absolute pixel coordinates, or None.
        """
        action_history: list[tuple[bytes, str]] = context.additional_context.get(
            "action_history", []
        )
        if action_history:
            return self.find_element_with_history(screenshot, context, action_history)
        return super().find_element(screenshot, context)

    def find_element_with_history(
        self,
        screenshot: bytes,
        context: HealingContext,
        action_history: list[tuple[bytes, str]],
    ) -> ElementLocation | None:
        """Locate an element using action history for context.

        Builds a multi-turn conversation where each previous turn contains
        a screenshot and the action that was taken, ending with the current
        screenshot and the element to find.

        Args:
            screenshot: PNG image bytes of current screen.
            context: Healing context with element description.
            action_history: List of (screenshot_bytes, action_description)
                tuples representing previous actions in chronological order.

        Returns:
            ElementLocation with absolute pixel coordinates, or None.
        """
        try:
            import httpx
        except ImportError:
            logger.error("httpx not installed. Run: pip install httpx")
            return None

        messages = self._build_context_messages(screenshot, context, action_history)

        payload = {
            "model": self._model,
            "messages": messages,
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
            logger.debug(f"Aria-UI context response: {response_text}")

            return self._parse_aria_response(response_text, context)

        except httpx.HTTPError as e:
            logger.error(f"Aria-UI context API error: {e}")
            return None
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected Aria-UI context response format: {e}")
            return None

    def _build_context_messages(
        self,
        screenshot: bytes,
        context: HealingContext,
        action_history: list[tuple[bytes, str]],
    ) -> list[dict[str, Any]]:
        """Build multi-turn conversation with action history.

        Creates interleaved user/assistant message pairs for each historical
        action, followed by the current grounding request.

        Format:
            Turn 1 (user): [prev_screenshot_1] "action description 1"
            Turn 1 (asst): "done"
            Turn 2 (user): [prev_screenshot_2] "action description 2"
            Turn 2 (asst): "done"
            Turn N (user): [current_screenshot] "find element: {description}"

        Args:
            screenshot: Current screenshot bytes.
            context: Current healing context.
            action_history: Previous (screenshot, action) pairs.

        Returns:
            List of message dicts for the chat completions API.
        """
        messages: list[dict[str, Any]] = []

        # Add history turns (limited to max_history)
        recent_history = action_history[-self._max_history :]
        for hist_screenshot, hist_action in recent_history:
            hist_b64 = base64.b64encode(hist_screenshot).decode("utf-8")

            # User turn: screenshot + action description
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{hist_b64}",
                            },
                        },
                        {"type": "text", "text": hist_action},
                    ],
                }
            )

            # Assistant acknowledgment turn
            messages.append(
                {
                    "role": "assistant",
                    "content": "done",
                }
            )

        # Final turn: current screenshot + grounding request
        current_b64 = base64.b64encode(screenshot).decode("utf-8")
        prompt = _ARIA_UI_PROMPT.format(description=context.original_description)

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{current_b64}",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        )

        return messages
