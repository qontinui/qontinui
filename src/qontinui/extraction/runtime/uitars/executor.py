"""UI-TARS Executor for runtime action execution.

This module provides execution capabilities using UI-TARS for element
grounding and action execution. Supports three modes:

1. Local (default): Uses existing RAG/template matching
2. UI-TARS: Always uses UI-TARS for grounding
3. Hybrid: Tries local first, falls back to UI-TARS if confidence is low
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from .config import UITARSSettings
from .models import (
    ActionResult,
    GroundingResult,
    UITARSAction,
    UITARSActionType,
    UITARSInferenceRequest,
)
from .provider import UITARSProviderBase, create_provider

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class LocalGrounder(Protocol):
    """Protocol for local grounding implementations.

    Allows UITARSExecutor to integrate with existing RAG/template grounding.
    """

    async def find(
        self,
        element_name: str,
        screenshot: np.ndarray[Any, Any],
    ) -> LocalGroundingResult:
        """Find element using local grounding (RAG/template).

        Args:
            element_name: Name or description of element to find
            screenshot: Current screenshot

        Returns:
            LocalGroundingResult with location and confidence
        """
        ...


@dataclass
class LocalGroundingResult:
    """Result from local grounding (RAG/template matching)."""

    x: int
    y: int
    confidence: float
    bbox: tuple[int, int, int, int] | None = None
    found: bool = True


class UITARSExecutor:
    """Execute actions using UI-TARS for visual grounding.

    Integrates with Qontinui's action system as an alternative or
    fallback grounding strategy.

    Example (UI-TARS only):
        executor = UITARSExecutor.from_settings(settings)
        result = await executor.ground_element(
            screenshot,
            "the blue Submit button"
        )

    Example (Hybrid with local fallback):
        executor = UITARSExecutor.from_settings(settings)
        executor.set_local_grounder(rag_grounder)
        result = await executor.hybrid_ground(
            "Submit button",
            screenshot,
            confidence_threshold=0.7
        )
    """

    def __init__(
        self,
        provider: UITARSProviderBase,
        settings: UITARSSettings,
    ) -> None:
        """Initialize executor.

        Args:
            provider: UI-TARS inference provider
            settings: Configuration settings
        """
        self.provider = provider
        self.settings = settings
        self._local_grounder: LocalGrounder | None = None
        self._screen_capture: Any = None
        self._initialized = False

    @classmethod
    def from_settings(cls, settings: UITARSSettings | None = None) -> UITARSExecutor:
        """Create executor from settings.

        Args:
            settings: UI-TARS settings (uses defaults if not provided)

        Returns:
            Initialized UITARSExecutor
        """
        settings = settings or UITARSSettings()
        provider = create_provider(settings)
        return cls(provider, settings)

    def set_local_grounder(self, grounder: LocalGrounder) -> None:
        """Set the local grounder for hybrid mode.

        Args:
            grounder: Implementation of LocalGrounder protocol
        """
        self._local_grounder = grounder

    def _ensure_initialized(self) -> None:
        """Ensure provider and screen capture are initialized."""
        if not self._initialized:
            if not self.provider.is_available():
                self.provider.initialize()

            try:
                import mss

                self._screen_capture = mss.mss()
            except ImportError:
                logger.warning("mss not installed, screenshot capture unavailable")

            self._initialized = True

    async def ground_element(
        self,
        screenshot: np.ndarray[Any, Any],
        element_description: str,
    ) -> GroundingResult:
        """Find element coordinates using UI-TARS visual grounding.

        Args:
            screenshot: Current screen capture (RGB numpy array)
            element_description: Natural language description of element
                               (e.g., "the Submit button", "search input field")

        Returns:
            GroundingResult with coordinates, confidence, and bounding box
        """
        self._ensure_initialized()

        start_time = time.time()

        # Build grounding prompt
        prompt = (
            f"Find the exact location of: {element_description}\n"
            "Output the center coordinates as Action: click(x, y)"
        )

        request = UITARSInferenceRequest(
            image=screenshot,
            prompt=prompt,
            max_new_tokens=128,  # Shorter for grounding
            temperature=0.0,  # Deterministic
        )

        result = self.provider.infer(request)
        inference_time = (time.time() - start_time) * 1000

        # Extract coordinates from action
        if result.action.x is not None and result.action.y is not None:
            return GroundingResult(
                x=result.action.x,
                y=result.action.y,
                confidence=result.action.confidence,
                element_description=element_description,
                found_description=result.thought.reasoning,
                raw_output=result.raw_output,
                inference_time_ms=inference_time,
            )
        else:
            # Grounding failed
            return GroundingResult(
                x=0,
                y=0,
                confidence=0.0,
                element_description=element_description,
                found_description=f"Failed to ground: {result.thought.reasoning}",
                raw_output=result.raw_output,
                inference_time_ms=inference_time,
            )

    async def execute_action(
        self,
        screenshot: np.ndarray[Any, Any],
        action_description: str,
    ) -> ActionResult:
        """Execute an action described in natural language.

        Uses UI-TARS Thought-Action decomposition to:
        1. Understand the action intent
        2. Ground the target element
        3. Execute via pyautogui

        Args:
            screenshot: Current screen (RGB numpy array)
            action_description: Natural language action description
                               (e.g., "Click the Save button",
                                "Type 'hello' in the search box")

        Returns:
            ActionResult with success status, coordinates used, confidence
        """
        self._ensure_initialized()

        start_time = time.time()

        # Build action prompt
        prompt = (
            f"Task: {action_description}\n"
            "Analyze the screenshot and perform the action. "
            "Output your reasoning as Thought: and the action as Action:."
        )

        request = UITARSInferenceRequest(
            image=screenshot,
            prompt=prompt,
            max_new_tokens=self.settings.max_new_tokens,
            temperature=self.settings.temperature,
        )

        result = self.provider.infer(request)

        # Create grounding result if coordinates available
        grounding = None
        if result.action.x is not None and result.action.y is not None:
            grounding = GroundingResult(
                x=result.action.x,
                y=result.action.y,
                confidence=result.action.confidence,
                element_description=action_description,
            )

        # Execute the action
        success = await self._execute_pyautogui_action(result.action)

        # Capture after screenshot
        screenshot_after = None
        if self._screen_capture:
            screenshot_after = await self._capture_screenshot()

        return ActionResult(
            success=success,
            thought=result.thought,
            action=result.action,
            grounding=grounding,
            screenshot_before=screenshot,
            screenshot_after=screenshot_after,
            execution_time_ms=(time.time() - start_time) * 1000,
            state_changed=True if success else False,
        )

    async def hybrid_ground(
        self,
        element_name: str,
        screenshot: np.ndarray[Any, Any],
        confidence_threshold: float | None = None,
    ) -> GroundingResult:
        """Hybrid grounding: try local first, fall back to UI-TARS.

        Args:
            element_name: Name/description of element to find
            screenshot: Current screenshot
            confidence_threshold: Min confidence for local grounding
                                 (uses settings default if not provided)

        Returns:
            GroundingResult from best available source
        """
        threshold = confidence_threshold or self.settings.confidence_threshold

        # 1. Try local grounding first if available
        if self._local_grounder is not None:
            try:
                local_result = await self._local_grounder.find(element_name, screenshot)

                if local_result.found and local_result.confidence >= threshold:
                    logger.debug(
                        f"Local grounding succeeded: {element_name} "
                        f"(confidence: {local_result.confidence:.2f})"
                    )
                    return GroundingResult(
                        x=local_result.x,
                        y=local_result.y,
                        confidence=local_result.confidence,
                        bbox=local_result.bbox,
                        element_description=element_name,
                        found_description="Found via local RAG/template matching",
                    )

                logger.debug(
                    f"Local grounding low confidence: {local_result.confidence:.2f} "
                    f"< {threshold:.2f}, falling back to UI-TARS"
                )

            except Exception as e:
                logger.warning(f"Local grounding failed: {e}, falling back to UI-TARS")

        # 2. Fall back to UI-TARS
        if self.settings.uitars_fallback_enabled:
            uitars_result = await self.ground_element(screenshot, element_name)
            logger.debug(
                f"UI-TARS grounding: {element_name} "
                f"(confidence: {uitars_result.confidence:.2f})"
            )
            return uitars_result

        # 3. Return local result if UI-TARS fallback disabled
        if self._local_grounder is not None:
            local_result = await self._local_grounder.find(element_name, screenshot)
            return GroundingResult(
                x=local_result.x,
                y=local_result.y,
                confidence=local_result.confidence,
                bbox=local_result.bbox,
                element_description=element_name,
                found_description="Local grounding (fallback disabled)",
            )

        # 4. No grounding available
        return GroundingResult(
            x=0,
            y=0,
            confidence=0.0,
            element_description=element_name,
            found_description="No grounding method available",
        )

    async def _execute_pyautogui_action(self, action: UITARSAction) -> bool:
        """Execute action via pyautogui.

        Args:
            action: UITARSAction to execute

        Returns:
            True if successful
        """
        try:
            import pyautogui

            pyautogui.FAILSAFE = True

            action_type = action.action_type

            if action_type == UITARSActionType.CLICK:
                if action.x is not None and action.y is not None:
                    pyautogui.click(action.x, action.y)
            elif action_type == UITARSActionType.DOUBLE_CLICK:
                if action.x is not None and action.y is not None:
                    pyautogui.doubleClick(action.x, action.y)
            elif action_type == UITARSActionType.RIGHT_CLICK:
                if action.x is not None and action.y is not None:
                    pyautogui.rightClick(action.x, action.y)
            elif action_type == UITARSActionType.TYPE:
                if action.x is not None and action.y is not None:
                    pyautogui.click(action.x, action.y)
                if action.text:
                    pyautogui.typewrite(action.text, interval=0.02)
            elif action_type == UITARSActionType.SCROLL:
                scroll_amt = action.scroll_amount or 100
                if action.scroll_direction == "up":
                    pyautogui.scroll(scroll_amt)
                elif action.scroll_direction == "down":
                    pyautogui.scroll(-scroll_amt)
                elif action.scroll_direction == "left":
                    pyautogui.hscroll(-scroll_amt)
                elif action.scroll_direction == "right":
                    pyautogui.hscroll(scroll_amt)
            elif action_type == UITARSActionType.HOVER:
                if action.x is not None and action.y is not None:
                    pyautogui.moveTo(action.x, action.y)
            elif action_type == UITARSActionType.DRAG:
                if (
                    action.x is not None
                    and action.y is not None
                    and action.end_x is not None
                    and action.end_y is not None
                ):
                    pyautogui.moveTo(action.x, action.y)
                    pyautogui.drag(action.end_x - action.x, action.end_y - action.y)
            elif action_type == UITARSActionType.HOTKEY:
                if action.keys:
                    pyautogui.hotkey(*action.keys)
            elif action_type == UITARSActionType.WAIT:
                await asyncio.sleep(action.duration or 1.0)
            elif action_type == UITARSActionType.DONE:
                pass  # No action needed
            else:
                logger.warning(f"Unknown action type: {action_type}")
                return False

            return True

        except Exception as e:
            logger.error(f"pyautogui action failed: {e}")
            return False

    async def _capture_screenshot(self) -> np.ndarray[Any, Any] | None:
        """Capture current screenshot.

        Returns:
            RGB numpy array or None if capture unavailable
        """
        if not self._screen_capture:
            return None

        try:
            monitor = self._screen_capture.monitors[1]  # Primary monitor
            screenshot = self._screen_capture.grab(monitor)
            img = np.array(screenshot)
            # BGRA to RGB
            return img[:, :, [2, 1, 0]]
        except Exception as e:
            logger.error(f"Screenshot capture failed: {e}")
            return None


class HybridGrounder:
    """Combines local RAG grounding with UI-TARS fallback.

    This class provides a unified interface for element grounding that
    automatically handles fallback logic based on confidence thresholds.

    Example:
        grounder = HybridGrounder(
            local_grounder=rag_grounder,
            uitars_executor=uitars_executor,
            settings=settings
        )
        location = await grounder.ground(state_image, screenshot)
    """

    def __init__(
        self,
        local_grounder: LocalGrounder | None = None,
        uitars_executor: UITARSExecutor | None = None,
        settings: UITARSSettings | None = None,
    ) -> None:
        """Initialize hybrid grounder.

        Args:
            local_grounder: Local grounding implementation (RAG/template)
            uitars_executor: UI-TARS executor for fallback
            settings: Configuration settings
        """
        self.local_grounder = local_grounder
        self.uitars_executor = uitars_executor
        self.settings = settings or UITARSSettings()

    async def ground(
        self,
        element_name: str,
        screenshot: np.ndarray[Any, Any],
    ) -> GroundingResult:
        """Find element using hybrid grounding strategy.

        Behavior depends on execution_mode setting:
        - "local": Only use local grounding
        - "uitars": Only use UI-TARS
        - "hybrid": Try local first, fall back to UI-TARS

        Args:
            element_name: Name/description of element
            screenshot: Current screenshot

        Returns:
            GroundingResult with location and confidence
        """
        mode = self.settings.execution_mode

        if mode == "local":
            return await self._ground_local(element_name, screenshot)
        elif mode == "uitars":
            return await self._ground_uitars(element_name, screenshot)
        else:  # hybrid
            return await self._ground_hybrid(element_name, screenshot)

    async def _ground_local(
        self,
        element_name: str,
        screenshot: np.ndarray[Any, Any],
    ) -> GroundingResult:
        """Ground using local method only."""
        if self.local_grounder is None:
            return GroundingResult(
                x=0,
                y=0,
                confidence=0.0,
                element_description=element_name,
                found_description="No local grounder configured",
            )

        try:
            result = await self.local_grounder.find(element_name, screenshot)
            return GroundingResult(
                x=result.x,
                y=result.y,
                confidence=result.confidence,
                bbox=result.bbox,
                element_description=element_name,
                found_description="Local grounding",
            )
        except Exception as e:
            logger.error(f"Local grounding failed: {e}")
            return GroundingResult(
                x=0,
                y=0,
                confidence=0.0,
                element_description=element_name,
                found_description=f"Local grounding failed: {e}",
            )

    async def _ground_uitars(
        self,
        element_name: str,
        screenshot: np.ndarray[Any, Any],
    ) -> GroundingResult:
        """Ground using UI-TARS only."""
        if self.uitars_executor is None:
            return GroundingResult(
                x=0,
                y=0,
                confidence=0.0,
                element_description=element_name,
                found_description="No UI-TARS executor configured",
            )

        return await self.uitars_executor.ground_element(screenshot, element_name)

    async def _ground_hybrid(
        self,
        element_name: str,
        screenshot: np.ndarray[Any, Any],
    ) -> GroundingResult:
        """Ground using hybrid strategy (local first, UI-TARS fallback)."""
        # Try local first
        if self.local_grounder is not None:
            try:
                local_result = await self.local_grounder.find(element_name, screenshot)

                if (
                    local_result.found
                    and local_result.confidence >= self.settings.confidence_threshold
                ):
                    return GroundingResult(
                        x=local_result.x,
                        y=local_result.y,
                        confidence=local_result.confidence,
                        bbox=local_result.bbox,
                        element_description=element_name,
                        found_description="Local grounding (hybrid mode)",
                    )

            except Exception as e:
                logger.warning(f"Local grounding failed in hybrid mode: {e}")

        # Fall back to UI-TARS
        if self.settings.uitars_fallback_enabled and self.uitars_executor is not None:
            return await self.uitars_executor.ground_element(screenshot, element_name)

        # Return local result even if low confidence
        if self.local_grounder is not None:
            try:
                local_result = await self.local_grounder.find(element_name, screenshot)
                return GroundingResult(
                    x=local_result.x,
                    y=local_result.y,
                    confidence=local_result.confidence,
                    bbox=local_result.bbox,
                    element_description=element_name,
                    found_description="Local grounding (fallback disabled)",
                )
            except Exception:
                pass

        return GroundingResult(
            x=0,
            y=0,
            confidence=0.0,
            element_description=element_name,
            found_description="No grounding method succeeded",
        )
