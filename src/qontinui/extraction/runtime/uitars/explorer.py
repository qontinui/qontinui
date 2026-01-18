"""UI-TARS Explorer for autonomous GUI state discovery.

This module implements RuntimeExtractor using UI-TARS for exploration.
The explorer uses the Thought-Action loop to autonomously navigate and
discover states in a GUI application.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ...models.base import BoundingBox, Screenshot, Viewport
from ...web.models import ExtractedElement
from ..base import (
    DetectedRegion,
    InteractionAction,
    RuntimeExtractor,
    StateChange,
)
from ..types import ExtractionTarget, RuntimeStateCapture, RuntimeType
from .config import UITARSSettings
from .models import (
    ExplorationTrajectory,
    UITARSActionType,
    UITARSInferenceRequest,
    UITARSStep,
)
from .provider import UITARSProviderBase, create_provider

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class UITARSExplorationConfig:
    """Configuration for a UI-TARS exploration session."""

    goal: str  # The exploration goal/task
    max_steps: int | None = None  # Override settings max_steps
    timeout_seconds: int | None = None  # Override settings timeout
    output_dir: Path | None = None  # Directory to save screenshots
    save_screenshots: bool = True
    detect_loops: bool = True  # Detect and break out of loops
    loop_threshold: int = 3  # Number of similar states to consider a loop


class UITARSExplorer(RuntimeExtractor):
    """RuntimeExtractor implementation using UI-TARS.

    Uses UI-TARS Thought-Action loop to autonomously explore GUI applications
    and discover states, elements, and transitions.

    Example:
        settings = UITARSSettings(
            provider="local_transformers",
            model_size="2B",
            quantization="int4"
        )
        explorer = UITARSExplorer(settings)
        await explorer.connect(target)
        trajectory = await explorer.explore(
            goal="Explore the settings menu"
        )
    """

    def __init__(
        self,
        settings: UITARSSettings | None = None,
        provider: UITARSProviderBase | None = None,
    ) -> None:
        """Initialize UI-TARS explorer.

        Args:
            settings: UI-TARS configuration settings
            provider: Pre-initialized provider (optional, created from settings if not provided)
        """
        self.settings = settings or UITARSSettings()
        self._provider = provider
        self._connected = False
        self._target: ExtractionTarget | None = None
        self._screen_capture: Any = None
        self._current_screenshot: np.ndarray[Any, Any] | None = None

    @property
    def provider(self) -> UITARSProviderBase:
        """Get or create the inference provider."""
        if self._provider is None:
            self._provider = create_provider(self.settings)
        return self._provider

    async def connect(self, target: ExtractionTarget) -> None:
        """Connect to the target application.

        For UI-TARS, this initializes screen capture for the target.
        """
        self._target = target
        logger.info(f"Connecting to target: {target.runtime_type}")

        # Initialize screen capture
        try:
            import mss

            self._screen_capture = mss.mss()
        except ImportError:
            logger.error("mss not installed. Run: pip install mss")
            raise

        # Initialize provider
        if not self.provider.is_available():
            self.provider.initialize()

        self._connected = True
        logger.info("UI-TARS explorer connected")

    async def extract_current_state(self) -> RuntimeStateCapture:
        """Extract the current visible UI state.

        Captures a screenshot and uses UI-TARS to identify visible elements.
        """
        if not self._connected:
            raise RuntimeError("Explorer not connected. Call connect() first.")

        # Capture screenshot
        screenshot = await self._capture_screenshot_array()
        self._current_screenshot = screenshot

        # Create capture record
        capture = RuntimeStateCapture(
            capture_id=str(uuid.uuid4()),
            url=self._target.url if self._target else None,
        )

        return capture

    async def extract_elements(self) -> list[ExtractedElement]:
        """Extract all interactive elements from the current UI.

        UI-TARS doesn't explicitly enumerate elements, so this returns
        elements discovered during exploration.
        """
        # UI-TARS works differently - it finds elements on-demand
        # Return empty list; elements are discovered during exploration
        return []

    async def detect_regions(self) -> list[DetectedRegion]:
        """Detect UI regions.

        UI-TARS doesn't explicitly detect regions like traditional extractors.
        """
        return []

    async def capture_screenshot(self, region: BoundingBox | None = None) -> Screenshot:
        """Capture a screenshot of the current state."""
        import tempfile

        from PIL import Image

        screenshot_array = await self._capture_screenshot_array(region)

        # Save to temp file (Screenshot requires a path)
        fd, temp_path = tempfile.mkstemp(suffix=f".{self.settings.screenshot_format}")
        img = Image.fromarray(screenshot_array)
        img.save(temp_path)

        return Screenshot(
            id=str(uuid.uuid4()),
            path=Path(temp_path),
            viewport=Viewport(
                width=screenshot_array.shape[1],
                height=screenshot_array.shape[0],
            ),
        )

    async def navigate_to_route(self, route: str) -> None:
        """Navigate to a specific route.

        For desktop apps, this is handled by UI-TARS actions.
        """
        # UI-TARS handles navigation through actions
        pass

    async def simulate_interaction(self, action: InteractionAction) -> StateChange:
        """Simulate a user interaction and observe state changes."""
        screenshot_before = await self._capture_screenshot_array()

        # Execute the action
        await self._execute_pyautogui_action(
            action.action_type,
            x=int(action.metadata.get("x", 0)) if action.metadata.get("x") else None,
            y=int(action.metadata.get("y", 0)) if action.metadata.get("y") else None,
            text=action.action_value,
        )

        # Wait for UI to settle
        await asyncio.sleep(0.5)

        # Capture after state
        screenshot_after = await self._capture_screenshot_array()

        # Compare states
        state_changed = not np.array_equal(screenshot_before, screenshot_after)

        return StateChange(
            screenshot_before=self._save_temp_screenshot(screenshot_before),
            screenshot_after=self._save_temp_screenshot(screenshot_after),
            metadata={"state_changed": state_changed},
        )

    async def disconnect(self) -> None:
        """Disconnect from the target application."""
        if self._screen_capture:
            self._screen_capture.close()
            self._screen_capture = None
        self._connected = False
        logger.info("UI-TARS explorer disconnected")

    @classmethod
    def supports_target(cls, target: ExtractionTarget) -> bool:
        """Check if this extractor supports the given target type.

        UI-TARS supports desktop applications via screen capture.
        """
        return target.runtime_type in (
            RuntimeType.NATIVE,
            RuntimeType.TAURI,
            RuntimeType.ELECTRON,
        )

    async def explore(
        self,
        goal: str | None = None,
        config: UITARSExplorationConfig | None = None,
        progress_callback: Callable[[int, str, str], None] | None = None,
    ) -> ExplorationTrajectory:
        """Run autonomous exploration using UI-TARS.

        Args:
            goal: Exploration goal (e.g., "Explore the settings menu")
            config: Full exploration configuration
            progress_callback: Optional callback called after each step with
                (step_number, thought_reasoning, action_description)

        Returns:
            ExplorationTrajectory with all discovered states and transitions
        """
        if not self._connected:
            raise RuntimeError("Explorer not connected. Call connect() first.")

        # Build config
        if config is None:
            config = UITARSExplorationConfig(
                goal=goal or "Explore this application and discover all available features",
            )
        elif goal:
            config.goal = goal

        # Initialize trajectory
        trajectory = ExplorationTrajectory(
            trajectory_id=str(uuid.uuid4()),
            goal=config.goal,
            output_dir=config.output_dir,
        )

        # Setup output directory
        if config.save_screenshots and config.output_dir:
            config.output_dir.mkdir(parents=True, exist_ok=True)

        max_steps = config.max_steps or self.settings.max_exploration_steps
        timeout = config.timeout_seconds or self.settings.exploration_timeout_seconds
        start_time = time.time()

        # State tracking for loop detection
        state_hashes: list[str] = []

        logger.info(f"Starting exploration: {config.goal}")

        for step_index in range(max_steps):
            # Check timeout
            if time.time() - start_time > timeout:
                logger.info("Exploration timeout reached")
                trajectory.complete("timeout")
                break

            try:
                # Capture current screenshot
                screenshot = await self._capture_screenshot_array()

                # Check for loops
                if config.detect_loops:
                    state_hash = self._hash_screenshot(screenshot)
                    recent_hashes = state_hashes[-config.loop_threshold:]
                    if state_hash in recent_hashes:
                        logger.info("Loop detected, breaking out")
                        trajectory.complete("loop_detected")
                        break
                    state_hashes.append(state_hash)

                # Build prompt with history context
                prompt = self._build_exploration_prompt(
                    config.goal,
                    step_index,
                    trajectory.steps[-3:] if trajectory.steps else [],
                )

                # Run inference
                request = UITARSInferenceRequest(
                    image=screenshot,
                    prompt=prompt,
                    max_new_tokens=self.settings.max_new_tokens,
                    temperature=self.settings.temperature,
                )

                step_start = time.time()
                result = self.provider.infer(request)

                # Check if done
                if result.action.action_type == UITARSActionType.DONE:
                    logger.info("UI-TARS indicated task complete")
                    trajectory.complete("completed")
                    break

                # Execute action
                success = await self._execute_action(result.action)

                # Wait for UI to settle
                await asyncio.sleep(0.3)

                # Capture after screenshot
                screenshot_after = await self._capture_screenshot_array()

                # Create step record
                step = UITARSStep(
                    step_index=step_index,
                    thought=result.thought,
                    action=result.action,
                    screenshot_before=screenshot if not config.save_screenshots else None,
                    screenshot_after=screenshot_after if not config.save_screenshots else None,
                    execution_time_ms=(time.time() - step_start) * 1000,
                    success=success,
                )

                # Save screenshots if configured
                if config.save_screenshots and config.output_dir:
                    step.screenshot_before_path = await self._save_screenshot(
                        screenshot,
                        config.output_dir / f"step_{step_index:03d}_before.{self.settings.screenshot_format}",
                    )
                    step.screenshot_after_path = await self._save_screenshot(
                        screenshot_after,
                        config.output_dir / f"step_{step_index:03d}_after.{self.settings.screenshot_format}",
                    )

                trajectory.add_step(step)

                # Call progress callback if provided
                if progress_callback:
                    action_desc = f"{result.action.action_type.value}"
                    if result.action.x is not None and result.action.y is not None:
                        action_desc += f" at ({result.action.x}, {result.action.y})"
                    if result.action.text:
                        action_desc += f": {result.action.text[:50]}"
                    progress_callback(step_index, result.thought.reasoning, action_desc)

                logger.info(
                    f"Step {step_index}: {result.action.action_type.value} "
                    f"({result.thought.reasoning[:50]}...)"
                )

            except Exception as e:
                logger.error(f"Exploration step {step_index} failed: {e}")
                step = UITARSStep(
                    step_index=step_index,
                    thought=result.thought if "result" in dir() else UITARSStep(
                        step_index=step_index,
                        thought=type("Thought", (), {"reasoning": str(e)})(),
                        action=type("Action", (), {"action_type": UITARSActionType.WAIT})(),
                    ).thought,
                    action=result.action if "result" in dir() else UITARSStep(
                        step_index=step_index,
                        thought=type("Thought", (), {"reasoning": str(e)})(),
                        action=type("Action", (), {"action_type": UITARSActionType.WAIT})(),
                    ).action,
                    success=False,
                    error=str(e),
                )
                trajectory.add_step(step)

                # Call progress callback for failed steps too
                if progress_callback:
                    progress_callback(step_index, f"Error: {e}", "failed")

        if trajectory.final_status == "incomplete":
            trajectory.complete("max_steps_reached")

        logger.info(
            f"Exploration complete: {trajectory.successful_steps}/{trajectory.total_steps} steps"
        )
        return trajectory

    async def _capture_screenshot_array(
        self,
        region: BoundingBox | None = None,
    ) -> np.ndarray[Any, Any]:
        """Capture screenshot as numpy array.

        Args:
            region: Optional region to capture

        Returns:
            RGB numpy array
        """
        if region:
            monitor = {
                "left": region.x,
                "top": region.y,
                "width": region.width,
                "height": region.height,
            }
        else:
            # Primary monitor
            monitor = self._screen_capture.monitors[1]

        screenshot = self._screen_capture.grab(monitor)

        # Convert to numpy RGB (mss captures BGRA)
        img = np.array(screenshot)
        # BGRA to RGB
        return img[:, :, [2, 1, 0]]

    async def _execute_action(self, action: Any) -> bool:
        """Execute a UI-TARS action via pyautogui.

        Args:
            action: UITARSAction to execute

        Returns:
            True if action succeeded
        """
        try:
            return await self._execute_pyautogui_action(
                action.action_type.value,
                x=action.x,
                y=action.y,
                text=action.text,
                scroll_direction=action.scroll_direction,
                scroll_amount=action.scroll_amount,
                keys=action.keys,
                duration=action.duration,
                end_x=action.end_x,
                end_y=action.end_y,
            )
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return False

    async def _execute_pyautogui_action(
        self,
        action_type: str,
        x: int | None = None,
        y: int | None = None,
        text: str | None = None,
        scroll_direction: str | None = None,
        scroll_amount: int | None = None,
        keys: list[str] | None = None,
        duration: float | None = None,
        end_x: int | None = None,
        end_y: int | None = None,
    ) -> bool:
        """Execute action via pyautogui.

        Args:
            action_type: Type of action (click, type, scroll, etc.)
            x, y: Coordinates for click/type actions
            text: Text for type action
            scroll_direction: Direction for scroll
            scroll_amount: Amount to scroll
            keys: Keys for hotkey
            duration: Duration for wait
            end_x, end_y: End coordinates for drag

        Returns:
            True if successful
        """
        try:
            import pyautogui

            pyautogui.FAILSAFE = True

            if action_type == "click" and x is not None and y is not None:
                pyautogui.click(x, y)
            elif action_type == "double_click" and x is not None and y is not None:
                pyautogui.doubleClick(x, y)
            elif action_type == "right_click" and x is not None and y is not None:
                pyautogui.rightClick(x, y)
            elif action_type == "type":
                if x is not None and y is not None:
                    pyautogui.click(x, y)
                if text:
                    pyautogui.typewrite(text, interval=0.02)
            elif action_type == "scroll":
                scroll_amt = scroll_amount or 100
                if scroll_direction == "up":
                    pyautogui.scroll(scroll_amt)
                elif scroll_direction == "down":
                    pyautogui.scroll(-scroll_amt)
                elif scroll_direction == "left":
                    pyautogui.hscroll(-scroll_amt)
                elif scroll_direction == "right":
                    pyautogui.hscroll(scroll_amt)
            elif action_type == "hover" and x is not None and y is not None:
                pyautogui.moveTo(x, y)
            elif action_type == "drag" and x is not None and y is not None and end_x is not None and end_y is not None:
                pyautogui.moveTo(x, y)
                pyautogui.drag(end_x - x, end_y - y)
            elif action_type == "hotkey" and keys:
                pyautogui.hotkey(*keys)
            elif action_type == "wait":
                await asyncio.sleep(duration or 1.0)
            elif action_type == "done":
                pass  # No action needed
            else:
                logger.warning(f"Unknown action type: {action_type}")
                return False

            return True

        except Exception as e:
            logger.error(f"pyautogui action failed: {e}")
            return False

    def _build_exploration_prompt(
        self,
        goal: str,
        step_index: int,
        recent_steps: list[UITARSStep],
    ) -> str:
        """Build the exploration prompt for UI-TARS.

        Args:
            goal: Exploration goal
            step_index: Current step number
            recent_steps: Recent steps for context

        Returns:
            Prompt string
        """
        prompt_parts = [
            f"Task: {goal}",
            f"Step: {step_index + 1}",
        ]

        if recent_steps:
            prompt_parts.append("Recent actions:")
            for step in recent_steps:
                prompt_parts.append(
                    f"- {step.action.action_type.value}: {step.thought.reasoning[:100]}"
                )

        prompt_parts.append(
            "Analyze the screenshot and decide the next action to progress toward the goal. "
            "Output your reasoning as Thought: and the action as Action:."
        )

        return "\n".join(prompt_parts)

    def _hash_screenshot(self, screenshot: np.ndarray[Any, Any]) -> str:
        """Create a hash of a screenshot for loop detection.

        Args:
            screenshot: Screenshot array

        Returns:
            Hash string
        """
        # Downsample for faster comparison
        small = screenshot[::10, ::10]
        return hashlib.md5(small.tobytes()).hexdigest()

    async def _save_screenshot(
        self,
        screenshot: np.ndarray[Any, Any],
        path: Path,
    ) -> Path:
        """Save screenshot to file.

        Args:
            screenshot: Screenshot array
            path: Output path

        Returns:
            Path to saved file
        """
        from PIL import Image

        img = Image.fromarray(screenshot)
        img.save(path)
        return path

    def _save_temp_screenshot(self, screenshot: np.ndarray[Any, Any]) -> str | None:
        """Save screenshot to temp file and return path."""
        import tempfile

        from PIL import Image

        img = Image.fromarray(screenshot)
        fd, path = tempfile.mkstemp(suffix=f".{self.settings.screenshot_format}")
        img.save(path)
        return path
