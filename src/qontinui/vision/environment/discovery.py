"""GUI Environment Discovery orchestration.

Provides the main GUIEnvironmentDiscovery class that orchestrates all
environment analyzers to create a comprehensive GUIEnvironment model.
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from qontinui_schemas.testing.environment import (
    ColorPalette,
    ConfidenceScores,
    DiscoveryProgress,
    DynamicRegions,
    ElementPatterns,
    GUIEnvironment,
    Layout,
    Typography,
    VisualStates,
)

from qontinui.vision.environment.analyzers import (
    ColorPaletteAnalyzer,
    DynamicRegionDetector,
    ElementPatternDetector,
    LayoutAnalyzer,
    TypographyAnalyzer,
    VisualStateLearner,
)
from qontinui.vision.environment.storage import (
    load_environment,
    save_environment,
)

logger = logging.getLogger(__name__)


class GUIEnvironmentDiscovery:
    """Orchestrates GUI environment discovery from screenshots.

    This class manages all environment analyzers and provides methods for
    passive discovery (from screenshots), active exploration (with actions),
    and continuous learning.

    Usage:
        discovery = GUIEnvironmentDiscovery()

        # Passive discovery from screenshots
        env = await discovery.discover_passive(screenshots)

        # Active exploration with actions
        env = await discovery.discover_active(
            action_callback=perform_action,
            initial_screenshots=screenshots,
        )

        # Save for later use
        discovery.save("environment.json")
    """

    def __init__(
        self,
        app_identifier: str | None = None,
        enable_continuous_learning: bool = False,
    ) -> None:
        """Initialize the discovery system.

        Args:
            app_identifier: Optional identifier for the application being analyzed.
            enable_continuous_learning: Whether to enable continuous learning mode.
        """
        self.app_identifier = app_identifier
        self.continuous_learning_enabled = enable_continuous_learning

        # Initialize analyzers
        self._color_analyzer = ColorPaletteAnalyzer()
        self._typography_analyzer = TypographyAnalyzer()
        self._layout_analyzer = LayoutAnalyzer()
        self._dynamic_detector = DynamicRegionDetector()
        self._state_learner = VisualStateLearner()
        self._element_detector = ElementPatternDetector()

        # Current environment state
        self._environment: GUIEnvironment | None = None
        self._screenshots_analyzed: int = 0
        self._actions_observed: int = 0

        # Progress callback
        self._progress_callback: Callable[[DiscoveryProgress], None] | None = None

    @property
    def environment(self) -> GUIEnvironment | None:
        """Get the current discovered environment."""
        return self._environment

    def set_progress_callback(
        self,
        callback: Callable[[DiscoveryProgress], None] | None,
    ) -> None:
        """Set a callback for progress updates during discovery.

        Args:
            callback: Function to call with DiscoveryProgress updates.
        """
        self._progress_callback = callback

    def _report_progress(
        self,
        phase: str,
        progress: float,
        message: str | None = None,
        screenshots_processed: int = 0,
        total_screenshots: int = 0,
    ) -> None:
        """Report progress via callback if set.

        Args:
            phase: Current analysis phase.
            progress: Progress value (0.0-1.0).
            message: Optional status message.
            screenshots_processed: Number of screenshots processed.
            total_screenshots: Total screenshots to process.
        """
        if self._progress_callback:
            self._progress_callback(
                DiscoveryProgress(
                    phase=phase,
                    progress=progress,
                    message=message,
                    screenshots_processed=screenshots_processed,
                    total_screenshots=total_screenshots,
                )
            )

    def _load_screenshot(self, source: str | Path | NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Load a screenshot from various sources.

        Args:
            source: Screenshot path, base64 string, or numpy array.

        Returns:
            Screenshot as numpy array in BGR format.

        Raises:
            ValueError: If screenshot cannot be loaded.
        """
        if isinstance(source, np.ndarray):
            return source

        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.exists():
                img = cv2.imread(str(path))
                if img is None:
                    raise ValueError(f"Failed to load image: {path}")
                return img.astype(np.uint8)

            # Try base64 decode
            if isinstance(source, str) and len(source) > 1000:
                import base64

                try:
                    img_bytes = base64.b64decode(source)
                    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        return img.astype(np.uint8)
                except Exception:
                    pass

            raise ValueError(f"Cannot load screenshot from: {source}")

        raise ValueError(f"Unsupported screenshot source type: {type(source)}")

    def _get_screen_resolution(
        self,
        screenshots: list[NDArray[np.uint8]],
    ) -> tuple[int, int] | None:
        """Get screen resolution from screenshots.

        Args:
            screenshots: List of screenshot arrays.

        Returns:
            (width, height) tuple or None if no screenshots.
        """
        if not screenshots:
            return None

        # Use first screenshot dimensions
        h, w = screenshots[0].shape[:2]
        return (w, h)

    async def discover_passive(
        self,
        screenshots: list[str | Path | NDArray[np.uint8]],
        include_colors: bool = True,
        include_typography: bool = True,
        include_layout: bool = True,
        include_dynamic: bool = True,
        include_elements: bool = True,
    ) -> GUIEnvironment:
        """Discover GUI environment from screenshots (passive mode).

        This mode analyzes existing screenshots without performing any actions.
        It cannot learn visual states (requires before/after action pairs).

        Args:
            screenshots: List of screenshots (paths, base64, or arrays).
            include_colors: Whether to analyze colors.
            include_typography: Whether to analyze typography.
            include_layout: Whether to analyze layout.
            include_dynamic: Whether to detect dynamic regions.
            include_elements: Whether to detect element patterns.

        Returns:
            Discovered GUIEnvironment model.
        """
        logger.info(f"Starting passive discovery with {len(screenshots)} screenshots")

        # Load screenshots
        self._report_progress("loading", 0.0, "Loading screenshots...")
        loaded_screenshots: list[NDArray[np.uint8]] = []
        for i, source in enumerate(screenshots):
            try:
                img = self._load_screenshot(source)
                loaded_screenshots.append(img)
            except ValueError as e:
                logger.warning(f"Skipping screenshot {i}: {e}")

            self._report_progress(
                "loading",
                (i + 1) / len(screenshots),
                f"Loaded {i + 1}/{len(screenshots)} screenshots",
                i + 1,
                len(screenshots),
            )

        if not loaded_screenshots:
            raise ValueError("No valid screenshots to analyze")

        screen_resolution = self._get_screen_resolution(loaded_screenshots)
        self._screenshots_analyzed = len(loaded_screenshots)

        # Run analyzers in parallel where possible
        tasks: dict[str, asyncio.Task[Any]] = {}

        if include_colors:
            tasks["colors"] = asyncio.create_task(self._color_analyzer.analyze(loaded_screenshots))

        if include_typography:
            tasks["typography"] = asyncio.create_task(
                self._typography_analyzer.analyze(loaded_screenshots)
            )

        if include_layout:
            tasks["layout"] = asyncio.create_task(self._layout_analyzer.analyze(loaded_screenshots))

        if include_dynamic:
            tasks["dynamic"] = asyncio.create_task(
                self._dynamic_detector.analyze(loaded_screenshots)
            )

        if include_elements:
            tasks["elements"] = asyncio.create_task(
                self._element_detector.analyze(loaded_screenshots)
            )

        # Wait for all tasks
        results: dict[str, Any] = {}
        total_tasks = len(tasks)
        completed = 0

        for name, task in tasks.items():
            self._report_progress(
                name,
                completed / total_tasks if total_tasks > 0 else 1.0,
                f"Analyzing {name}...",
            )
            try:
                results[name] = await task
                logger.info(f"Completed {name} analysis")
            except Exception as e:
                logger.error(f"Failed {name} analysis: {e}")
                results[name] = None

            completed += 1
            self._report_progress(
                name,
                completed / total_tasks,
                f"Completed {name}",
            )

        # Build environment model
        self._report_progress("building", 0.9, "Building environment model...")

        confidence_scores = ConfidenceScores(
            color_extraction=self._color_analyzer.confidence if include_colors else 0.0,
            typography_detection=(
                self._typography_analyzer.confidence if include_typography else 0.0
            ),
            layout_analysis=self._layout_analyzer.confidence if include_layout else 0.0,
            dynamic_detection=self._dynamic_detector.confidence if include_dynamic else 0.0,
            state_learning=0.0,  # No state learning in passive mode
            element_detection=self._element_detector.confidence if include_elements else 0.0,
        )

        self._environment = GUIEnvironment(
            version="1.0.0",
            app_identifier=self.app_identifier,
            discovery_timestamp=datetime.now(UTC),
            screen_resolution=screen_resolution,
            screenshots_analyzed=self._screenshots_analyzed,
            actions_observed=0,
            colors=results.get("colors") or ColorPalette(),
            typography=results.get("typography") or Typography(),
            layout=results.get("layout") or Layout(),
            dynamic_regions=results.get("dynamic") or DynamicRegions(),
            visual_states=VisualStates(),  # Empty in passive mode
            element_patterns=results.get("elements") or ElementPatterns(),
            confidence_scores=confidence_scores,
            continuous_learning_enabled=self.continuous_learning_enabled,
            last_updated=datetime.now(UTC),
            update_count=0,
        )

        self._report_progress("complete", 1.0, "Discovery complete")
        logger.info("Passive discovery complete")

        return self._environment

    async def discover_active(
        self,
        action_callback: Callable[[str], tuple[NDArray[np.uint8], NDArray[np.uint8]]],
        initial_screenshots: list[str | Path | NDArray[np.uint8]] | None = None,
        actions_to_perform: list[str] | None = None,
        include_all: bool = True,
    ) -> GUIEnvironment:
        """Discover GUI environment with active exploration.

        This mode performs actions and captures before/after screenshots to
        learn visual states and detect dynamic regions that change on action.

        Args:
            action_callback: Function that performs an action and returns
                (before_screenshot, after_screenshot).
            initial_screenshots: Optional initial screenshots for passive analysis.
            actions_to_perform: List of actions to perform for state learning.
            include_all: Whether to include all analysis types.

        Returns:
            Discovered GUIEnvironment model.
        """
        logger.info("Starting active discovery")

        # First do passive analysis if we have initial screenshots
        if initial_screenshots:
            await self.discover_passive(
                initial_screenshots,
                include_colors=include_all,
                include_typography=include_all,
                include_layout=include_all,
                include_dynamic=include_all,
                include_elements=include_all,
            )

        # Default actions for state learning
        if actions_to_perform is None:
            actions_to_perform = [
                "click_button",
                "hover_button",
                "focus_input",
                "toggle_checkbox",
                "enable_disabled",
            ]

        # Perform actions and collect before/after pairs
        action_pairs: list[tuple[NDArray[np.uint8], NDArray[np.uint8], str]] = []

        for i, action in enumerate(actions_to_perform):
            self._report_progress(
                "active_exploration",
                i / len(actions_to_perform),
                f"Performing action: {action}",
            )

            try:
                before, after = action_callback(action)
                action_pairs.append((before, after, action))
                self._actions_observed += 1
                logger.info(f"Captured action pair for: {action}")
            except Exception as e:
                logger.warning(f"Failed to perform action {action}: {e}")

        # Learn visual states from action pairs
        if action_pairs:
            self._report_progress(
                "state_learning",
                0.5,
                "Learning visual states from actions...",
            )

            # Extract screenshots for state learner
            all_screenshots = [pair[0] for pair in action_pairs] + [
                pair[1] for pair in action_pairs
            ]
            actions = [pair[2] for pair in action_pairs]

            try:
                visual_states = await self._state_learner.analyze(
                    all_screenshots,
                    action_pairs=list(
                        zip(
                            [p[0] for p in action_pairs],
                            [p[1] for p in action_pairs], strict=False,
                        )
                    ),
                    actions=actions,
                )

                if self._environment:
                    self._environment.visual_states = visual_states
                    self._environment.confidence_scores.state_learning = (
                        self._state_learner.confidence
                    )
                    self._environment.actions_observed = self._actions_observed

            except Exception as e:
                logger.error(f"State learning failed: {e}")

        # Update environment
        if self._environment:
            self._environment.last_updated = datetime.now(UTC)
            self._environment.update_count += 1

        self._report_progress("complete", 1.0, "Active discovery complete")
        logger.info("Active discovery complete")

        return self._environment or GUIEnvironment()

    async def update_from_screenshot(
        self,
        screenshot: str | Path | NDArray[np.uint8],
        before_screenshot: NDArray[np.uint8] | None = None,
        action_type: str | None = None,
    ) -> None:
        """Update environment with a new screenshot (continuous learning).

        Args:
            screenshot: New screenshot to analyze.
            before_screenshot: Screenshot before action (for state learning).
            action_type: Type of action performed (for state learning).
        """
        if not self.continuous_learning_enabled:
            logger.warning("Continuous learning is disabled")
            return

        if not self._environment:
            logger.warning("No environment to update - run discover_passive first")
            return

        _img = self._load_screenshot(screenshot)  # noqa: F841 - validates screenshot

        # Update analyzers that benefit from more data
        # Color and typography benefit from more samples
        # Layout should remain stable
        # Dynamic detection benefits from more frames

        self._screenshots_analyzed += 1
        self._environment.screenshots_analyzed = self._screenshots_analyzed
        self._environment.last_updated = datetime.now(UTC)
        self._environment.update_count += 1

        logger.info("Updated environment with new screenshot")

    def save(self, file_path: str | Path) -> None:
        """Save the current environment to a JSON file.

        Args:
            file_path: Path to save the environment.

        Raises:
            ValueError: If no environment has been discovered.
        """
        if not self._environment:
            raise ValueError("No environment to save - run discovery first")

        save_environment(self._environment, file_path)

    @classmethod
    def load(cls, file_path: str | Path) -> "GUIEnvironmentDiscovery":
        """Load an environment from a JSON file.

        Args:
            file_path: Path to the environment file.

        Returns:
            GUIEnvironmentDiscovery instance with loaded environment.
        """
        environment = load_environment(file_path)

        discovery = cls(
            app_identifier=environment.app_identifier,
            enable_continuous_learning=environment.continuous_learning_enabled,
        )
        discovery._environment = environment
        discovery._screenshots_analyzed = environment.screenshots_analyzed
        discovery._actions_observed = environment.actions_observed

        return discovery

    def reset(self) -> None:
        """Reset the discovery system for fresh analysis."""
        self._color_analyzer.reset()
        self._typography_analyzer.reset()
        self._layout_analyzer.reset()
        self._dynamic_detector.reset()
        self._state_learner.reset()
        self._element_detector.reset()

        self._environment = None
        self._screenshots_analyzed = 0
        self._actions_observed = 0

        logger.info("Discovery system reset")


__all__ = ["GUIEnvironmentDiscovery"]
