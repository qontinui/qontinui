"""Example integration of self-healing locators with qontinui Actions.

This module demonstrates how to integrate the self-healing locator system
with the existing qontinui Actions class for reduced test brittleness.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..actions.action_result import ActionResult, ActionResultBuilder
from ..model.element import Pattern, Region
from .healing import HealingConfig, HealingManager
from .strategies import ScreenContext


class SelfHealingActions:
    """Actions wrapper with self-healing locators.

    Wraps the standard Actions class to provide automatic healing when
    pattern matching fails. Can be used as a drop-in replacement for Actions.

    Example:
        >>> from qontinui.locators import HealingConfig
        >>> config = HealingConfig(auto_heal=True, update_on_heal=True)
        >>> actions = SelfHealingActions(config)
        >>>
        >>> # Use like normal Actions - healing happens automatically
        >>> result = actions.find(pattern)
        >>> if result.success:
        ...     print(f"Found! Healed: {result.metadata.get('healed', False)}")
    """

    def __init__(self, config: HealingConfig | None = None) -> None:
        """Initialize self-healing actions.

        Args:
            config: Healing configuration (default: HealingConfig())
        """
        self.healing_manager = HealingManager(config or HealingConfig())

        # Import here to avoid circular dependency
        from ..actions.pure import PureActions

        self.pure_actions = PureActions()

    def find(self, pattern: Pattern, region: Region | None = None) -> ActionResult:
        """Find pattern with self-healing.

        First tries standard template matching. If that fails, tries alternative
        strategies based on healing configuration.

        Args:
            pattern: Pattern to find
            region: Optional search region

        Returns:
            ActionResult with matches and healing metadata
        """
        # Capture screenshot
        screenshot = self._capture_screenshot()

        # Create screen context
        context = ScreenContext(
            screenshot=screenshot,
            timestamp=time.time(),
            metadata={"search_region": region} if region else {},
        )

        # Find with healing
        result = self.healing_manager.find_with_healing(
            target=pattern,
            context=context,
        )

        # Convert to ActionResult
        if result.found and result.match_result:
            match_location = result.match_result.to_location()

            # Create ActionResult with match
            action_result = (
                ActionResultBuilder()
                .with_success(True)
                .with_matches([match_location])  # type: ignore[attr-defined]
                .build()
            )

            # Add healing metadata
            action_result.metadata = {  # type: ignore[attr-defined]
                "healed": result.metadata.get("healed", False),
                "healing_strategy": result.successful_strategy,
                "confidence": result.confidence,
                "attempts": len(result.attempts),
            }

            return action_result  # type: ignore[no-any-return]
        else:
            # No match found
            return (
                ActionResultBuilder()
                .with_success(False)
                .with_output_text(f"Pattern {pattern.name} not found after healing attempts")
                .build()
            )

    def _capture_screenshot(self) -> np.ndarray:
        """Capture screenshot using pure actions.

        Returns:
            Screenshot as numpy array (BGR format)
        """
        # Use pure actions to capture screenshot
        screenshot_result = self.pure_actions.screenshot()  # type: ignore[attr-defined]

        # Convert to numpy array if needed
        if hasattr(screenshot_result, "get_mat_bgr"):
            return screenshot_result.get_mat_bgr()  # type: ignore[no-any-return]
        elif isinstance(screenshot_result, np.ndarray):
            return screenshot_result
        else:
            # Fallback: create empty screenshot
            return np.zeros((1080, 1920, 3), dtype=np.uint8)

    def get_healing_stats(self) -> dict[str, Any]:
        """Get healing statistics.

        Returns:
            Dict with healing stats
        """
        return self.healing_manager.get_healing_stats()


# Example usage functions


def example_basic_usage() -> None:
    """Example: Basic usage with self-healing."""
    from ..model.element import Pattern

    # Create pattern (would normally load from file)
    pattern = Pattern.from_file("path/to/button.png")

    # Create actions with healing
    config = HealingConfig(auto_heal=True)
    actions = SelfHealingActions(config)

    # Find pattern - healing happens automatically
    result = actions.find(pattern)

    if result.success:
        if result.metadata.get("healed"):  # type: ignore[attr-defined]
            print(f"Found via healing using {result.metadata['healing_strategy']}")  # type: ignore[attr-defined]
        else:
            print("Found via primary strategy")
    else:
        print("Not found even after healing attempts")


def example_aggressive_healing() -> None:
    """Example: Aggressive healing with pattern updates."""
    from ..model.element import Pattern

    # Create pattern
    pattern = Pattern.from_file("path/to/element.png")

    # Create actions with aggressive healing
    # - Updates patterns on successful heal
    # - Lower confidence threshold
    actions = SelfHealingActions(
        HealingConfig(auto_heal=True, update_on_heal=True, confidence_threshold=0.6)
    )

    # Find pattern
    result = actions.find(pattern)

    if result.success and result.metadata.get("healed"):  # type: ignore[attr-defined]
        print("Pattern healed and updated for future use")


def example_conservative_healing() -> None:
    """Example: Conservative healing without updates."""
    from ..model.element import Pattern

    pattern = Pattern.from_file("path/to/element.png")

    # Conservative healing: higher threshold, no updates
    actions = SelfHealingActions(
        HealingConfig(
            auto_heal=True,
            update_on_heal=False,
            confidence_threshold=0.85,
            fallback_strategies=["visual", "text"],
        )
    )

    result = actions.find(pattern)

    if result.success:
        print(f"Found with confidence {result.metadata['confidence']}")  # type: ignore[attr-defined]


def example_custom_strategies() -> None:
    """Example: Custom strategy configuration."""
    from ..model.element import Pattern

    pattern = Pattern.from_file("path/to/element.png")

    # Only use visual and text strategies
    config = HealingConfig(
        auto_heal=True,
        fallback_strategies=["visual", "text"],
    )

    actions = SelfHealingActions(config)
    result = actions.find(pattern)

    print(f"Strategies tried: {len(result.metadata.get('attempts', []))}")  # type: ignore[attr-defined]


def example_healing_stats() -> None:
    """Example: Getting healing statistics."""
    from ..model.element import Pattern

    actions = SelfHealingActions()

    # Perform multiple finds
    patterns = [Pattern.from_file(f"pattern_{i}.png") for i in range(5)]

    for pattern in patterns:
        actions.find(pattern)

    # Get statistics
    stats = actions.get_healing_stats()

    print(f"Total attempts: {stats['total_attempts']}")
    print(f"Healing rate: {stats['healing_rate']:.2%}")
    print(f"Patterns updated: {stats['patterns_updated']}")


def example_with_existing_actions() -> None:
    """Example: Integrating with existing Actions class."""
    from ..actions import Actions
    from ..model.element import Pattern

    # Standard actions
    standard_actions = Actions()

    # Self-healing actions
    healing_actions = SelfHealingActions()

    pattern = Pattern.from_file("path/to/element.png")

    # Try standard first
    result = standard_actions.find(pattern)

    if not result.success:
        # Fall back to healing
        print("Standard find failed, trying with healing...")
        result = healing_actions.find(pattern)

        if result.success:
            print(f"Healed! Strategy: {result.metadata['healing_strategy']}")  # type: ignore[attr-defined]


if __name__ == "__main__":
    # Run examples
    print("=== Basic Usage ===")
    example_basic_usage()

    print("\n=== Aggressive Healing ===")
    example_aggressive_healing()

    print("\n=== Conservative Healing ===")
    example_conservative_healing()

    print("\n=== Custom Strategies ===")
    example_custom_strategies()

    print("\n=== Healing Stats ===")
    example_healing_stats()

    print("\n=== Integration with Existing Actions ===")
    example_with_existing_actions()
