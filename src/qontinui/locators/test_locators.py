"""Tests for self-healing locator system.

These tests demonstrate the functionality and can be run with pytest.
"""

import time

import numpy as np
import pytest

from ..model.element import Pattern, Region
from .healing import HealingConfig, HealingManager
from .multi_strategy import MultiStrategyLocator
from .strategies import (
    ColorRegionStrategy,
    MatchResult,
    RelativePositionStrategy,
    ScreenContext,
    SemanticTextStrategy,
    StructuralStrategy,
    VisualPatternStrategy,
)


@pytest.fixture
def sample_screenshot():
    """Create sample screenshot for testing."""
    # Create 1920x1080 BGR image
    screenshot = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Add some colored regions for testing
    screenshot[100:200, 100:200] = [255, 0, 0]  # Blue region
    screenshot[300:400, 300:400] = [0, 255, 0]  # Green region
    screenshot[500:600, 500:600] = [0, 0, 255]  # Red region

    return screenshot


@pytest.fixture
def sample_context(sample_screenshot):
    """Create sample screen context."""
    return ScreenContext(screenshot=sample_screenshot, timestamp=time.time())


@pytest.fixture
def sample_pattern():
    """Create sample pattern for testing."""
    # Create small pattern (50x50 blue square)
    pixel_data = np.zeros((50, 50, 3), dtype=np.uint8)
    pixel_data[:, :] = [255, 0, 0]  # Blue

    mask = np.ones((50, 50), dtype=np.float32)

    return Pattern(
        id="test_pattern_1",
        name="blue_square",
        pixel_data=pixel_data,
        mask=mask,
        x=0,
        y=0,
        width=50,
        height=50,
    )


class TestLocatorStrategies:
    """Test individual locator strategies."""

    def test_visual_pattern_strategy_can_handle(self, sample_pattern):
        """Test VisualPatternStrategy.can_handle()."""
        strategy = VisualPatternStrategy()

        assert strategy.can_handle(sample_pattern) is True
        assert strategy.can_handle({"text": "button"}) is False
        assert strategy.can_handle("invalid") is False

    def test_visual_pattern_strategy_name(self):
        """Test VisualPatternStrategy.get_name()."""
        strategy = VisualPatternStrategy()
        assert strategy.get_name() == "VisualPattern"

    def test_semantic_text_strategy_can_handle(self):
        """Test SemanticTextStrategy.can_handle()."""
        strategy = SemanticTextStrategy()

        assert strategy.can_handle({"text": "Login"}) is True
        assert strategy.can_handle({"color_range": [(0, 0, 0), (255, 255, 255)]}) is False

    def test_relative_position_strategy_can_handle(self, sample_pattern):
        """Test RelativePositionStrategy.can_handle()."""
        strategy = RelativePositionStrategy()

        config = {"anchor_pattern": sample_pattern, "offset_x": 100, "offset_y": 50}

        assert strategy.can_handle(config) is True
        assert strategy.can_handle(sample_pattern) is False

    def test_color_region_strategy_can_handle(self):
        """Test ColorRegionStrategy.can_handle()."""
        strategy = ColorRegionStrategy()

        config = {"color_range": ((0, 100, 100), (10, 255, 255)), "min_area": 100}

        assert strategy.can_handle(config) is True
        assert strategy.can_handle({"text": "button"}) is False

    def test_structural_strategy_can_handle(self):
        """Test StructuralStrategy.can_handle()."""
        strategy = StructuralStrategy()

        config = {"element_type": "button", "min_width": 50, "max_width": 200}

        assert strategy.can_handle(config) is True
        assert strategy.can_handle({"text": "button"}) is False

    def test_color_region_strategy_find(self, sample_context):
        """Test ColorRegionStrategy.find() with known color."""
        strategy = ColorRegionStrategy()

        # Search for blue region (BGR: [255, 0, 0] = HSV: [120, 255, 255])
        config = {
            "color_range": ((115, 200, 200), (125, 255, 255)),  # Blue in HSV
            "min_area": 100,
        }

        result = strategy.find(config, sample_context, min_confidence=0.5)

        # Should find the blue region we created
        assert result is not None
        assert isinstance(result, MatchResult)
        assert result.confidence > 0.0


class TestMultiStrategyLocator:
    """Test multi-strategy locator."""

    def test_create_default(self):
        """Test creating locator with default strategies."""
        locator = MultiStrategyLocator.create_default()

        assert len(locator.strategies) == 5
        assert locator.strategies[0].get_name() == "VisualPattern"

    def test_create_visual_only(self):
        """Test creating locator with only visual strategy."""
        locator = MultiStrategyLocator.create_visual_only()

        assert len(locator.strategies) == 1
        assert locator.strategies[0].get_name() == "VisualPattern"

    def test_create_with_strategies(self):
        """Test creating locator with specific strategies."""
        locator = MultiStrategyLocator.create_with_strategies("visual", "text", "color")

        assert len(locator.strategies) == 3
        assert locator.strategies[0].get_name() == "VisualPattern"
        assert locator.strategies[1].get_name() == "SemanticText"
        assert locator.strategies[2].get_name() == "ColorRegion"

    def test_create_with_invalid_strategy(self):
        """Test creating locator with invalid strategy name."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            MultiStrategyLocator.create_with_strategies("invalid_strategy")

    def test_add_strategy(self):
        """Test adding strategy to locator."""
        locator = MultiStrategyLocator(strategies=[])
        assert len(locator.strategies) == 0

        locator.add_strategy(VisualPatternStrategy())
        assert len(locator.strategies) == 1

    def test_remove_strategy(self):
        """Test removing strategy from locator."""
        locator = MultiStrategyLocator.create_default()
        initial_count = len(locator.strategies)

        removed = locator.remove_strategy("SemanticText")

        assert removed is True
        assert len(locator.strategies) == initial_count - 1

    def test_clear_strategies(self):
        """Test clearing all strategies."""
        locator = MultiStrategyLocator.create_default()
        locator.clear_strategies()

        assert len(locator.strategies) == 0

    def test_get_strategy_stats(self):
        """Test getting strategy statistics."""
        locator = MultiStrategyLocator.create_default()

        # Stats should be empty initially
        stats = locator.get_strategy_stats()
        assert len(stats) == 0

    def test_find_with_no_strategies(self, sample_pattern, sample_context):
        """Test find with no strategies returns not found."""
        locator = MultiStrategyLocator(strategies=[])

        result = locator.find(sample_pattern, sample_context)

        assert result.found is False
        assert result.successful_strategy is None


class TestHealingManager:
    """Test healing manager."""

    def test_create_with_defaults(self):
        """Test creating healing manager with defaults."""
        manager = HealingManager.create_with_defaults()

        assert manager.config.auto_heal is True
        assert manager.config.confidence_threshold == 0.7

    def test_create_aggressive(self):
        """Test creating aggressive healing manager."""
        manager = HealingManager.create_aggressive()

        assert manager.config.auto_heal is True
        assert manager.config.update_on_heal is True
        assert manager.config.confidence_threshold == 0.6

    def test_create_conservative(self):
        """Test creating conservative healing manager."""
        manager = HealingManager.create_conservative()

        assert manager.config.auto_heal is True
        assert manager.config.update_on_heal is False
        assert manager.config.confidence_threshold == 0.85

    def test_healing_config_validation(self):
        """Test HealingConfig validation."""
        # Valid config
        config = HealingConfig(confidence_threshold=0.8, max_healing_attempts=5)
        assert config.confidence_threshold == 0.8

        # Invalid confidence threshold
        with pytest.raises(ValueError, match="confidence_threshold"):
            HealingConfig(confidence_threshold=1.5)

        # Invalid max_healing_attempts
        with pytest.raises(ValueError, match="max_healing_attempts"):
            HealingConfig(max_healing_attempts=0)

    def test_get_healing_stats_empty(self):
        """Test getting healing stats when no attempts made."""
        manager = HealingManager()

        stats = manager.get_healing_stats()

        assert stats["total_attempts"] == 0
        assert stats["healing_successes"] == 0
        assert stats["healing_rate"] == 0.0

    def test_clear_history(self):
        """Test clearing healing history."""
        manager = HealingManager()

        # History should be empty initially
        assert len(manager.get_healing_history()) == 0

        # Clear should work even when empty
        manager.clear_history()
        assert len(manager.get_healing_history()) == 0

    def test_get_strategy_preferences_unknown_pattern(self):
        """Test getting preferences for unknown pattern."""
        manager = HealingManager()

        preferences = manager.get_strategy_preferences("unknown_pattern")

        assert len(preferences) == 0

    def test_register_update_callback(self):
        """Test registering pattern update callback."""
        manager = HealingManager()

        callback_called = False

        def callback(pattern, new_pixel_data):
            nonlocal callback_called
            callback_called = True

        manager.register_update_callback("pattern_1", callback)

        # Callback registered but not called yet
        assert callback_called is False

    def test_string_representation(self):
        """Test __str__ and __repr__."""
        manager = HealingManager()

        str_repr = str(manager)
        assert "HealingManager" in str_repr
        assert "auto_heal=True" in str_repr

        repr_repr = repr(manager)
        assert "HealingManager" in repr_repr


class TestMatchResult:
    """Test MatchResult functionality."""

    def test_to_location(self):
        """Test converting MatchResult to Location."""
        region = Region(x=100, y=200, width=50, height=30)
        result = MatchResult(
            region=region,
            confidence=0.95,
            strategy_name="VisualPattern",
        )

        location = result.to_location()

        # Location should be at center of region
        assert location.x == 125  # 100 + 50/2
        assert location.y == 215  # 200 + 30/2
        assert location.region == region


class TestScreenContext:
    """Test ScreenContext."""

    def test_screen_context_creation(self, sample_screenshot):
        """Test creating screen context."""
        timestamp = time.time()

        context = ScreenContext(
            screenshot=sample_screenshot,
            timestamp=timestamp,
            metadata={"test": "value"},
        )

        assert context.screenshot.shape == sample_screenshot.shape
        assert context.timestamp == timestamp
        assert context.metadata["test"] == "value"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
