"""Tests for background removal functionality"""

import numpy as np
import pytest
from discovery.background_removal import (
    BackgroundRemovalAnalyzer,
    BackgroundRemovalConfig,
    remove_backgrounds_simple,
)


def create_test_screenshot_with_static_icon(
    width: int = 100,
    height: int = 100,
    icon_color: tuple = (255, 255, 255),
    bg_color: tuple = (50, 50, 50),
) -> np.ndarray:
    """Create a test screenshot with a static icon on a background"""
    screenshot = np.full((height, width, 3), bg_color, dtype=np.uint8)

    # Add a static "icon" in the center
    icon_size = 20
    center_x, center_y = width // 2, height // 2
    screenshot[
        center_y - icon_size // 2 : center_y + icon_size // 2,
        center_x - icon_size // 2 : center_x + icon_size // 2,
    ] = icon_color

    return screenshot


def test_background_removal_config():
    """Test that config can be created with defaults"""
    config = BackgroundRemovalConfig()
    assert config.use_temporal_variance is True
    assert config.use_edge_density is True
    assert config.use_uniformity is True
    assert config.variance_threshold == 20.0


def test_analyzer_initialization():
    """Test that analyzer can be initialized"""
    analyzer = BackgroundRemovalAnalyzer()
    assert analyzer.config is not None

    custom_config = BackgroundRemovalConfig(variance_threshold=30.0)
    analyzer_custom = BackgroundRemovalAnalyzer(custom_config)
    assert analyzer_custom.config.variance_threshold == 30.0


def test_remove_backgrounds_empty_list():
    """Test that empty screenshot list returns empty result"""
    analyzer = BackgroundRemovalAnalyzer()
    masked, stats = analyzer.remove_backgrounds([])
    assert len(masked) == 0
    assert stats == {}


def test_remove_backgrounds_single_screenshot():
    """Test background removal with single screenshot"""
    analyzer = BackgroundRemovalAnalyzer()
    screenshot = create_test_screenshot_with_static_icon()

    masked, stats = analyzer.remove_backgrounds([screenshot])

    assert len(masked) == 1
    assert masked[0].shape[2] == 4  # RGBA
    assert "foreground_pixels" in stats
    assert "background_pixels" in stats


def test_remove_backgrounds_multiple_screenshots():
    """Test background removal with multiple screenshots"""
    analyzer = BackgroundRemovalAnalyzer()

    # Create 3 screenshots with same icon but different backgrounds
    screenshots = [
        create_test_screenshot_with_static_icon(bg_color=(50, 50, 50)),
        create_test_screenshot_with_static_icon(bg_color=(60, 60, 60)),
        create_test_screenshot_with_static_icon(bg_color=(70, 70, 70)),
    ]

    masked, stats = analyzer.remove_backgrounds(screenshots)

    assert len(masked) == 3
    assert all(img.shape[2] == 4 for img in masked)  # All RGBA
    assert stats["num_screenshots"] == 3
    assert stats["foreground_percentage"] > 0
    assert stats["background_percentage"] > 0


def test_temporal_variance_detection():
    """Test that temporal variance correctly identifies changing pixels"""
    config = BackgroundRemovalConfig(
        use_temporal_variance=True,
        use_edge_density=False,
        use_uniformity=False,
        variance_threshold=5.0,
    )
    analyzer = BackgroundRemovalAnalyzer(config)

    # Create screenshots where background changes but icon stays same
    screenshots = [
        create_test_screenshot_with_static_icon(bg_color=(50, 50, 50)),
        create_test_screenshot_with_static_icon(bg_color=(100, 100, 100)),
        create_test_screenshot_with_static_icon(bg_color=(150, 150, 150)),
    ]

    masked, stats = analyzer.remove_backgrounds(screenshots, debug=True)

    # The icon should be preserved (not transparent)
    # Check center pixel of icon
    center_x, center_y = 50, 50
    icon_alpha = masked[0][center_y, center_x, 3]
    assert icon_alpha == 255  # Icon should be opaque


def test_simple_convenience_function():
    """Test the simple convenience function"""
    screenshots = [
        create_test_screenshot_with_static_icon(),
        create_test_screenshot_with_static_icon(bg_color=(60, 60, 60)),
    ]

    masked = remove_backgrounds_simple(screenshots)

    assert len(masked) == 2
    assert all(img.shape[2] == 4 for img in masked)


def test_statistics_calculation():
    """Test that statistics are calculated correctly"""
    analyzer = BackgroundRemovalAnalyzer()
    screenshot = create_test_screenshot_with_static_icon(width=100, height=100)

    masked, stats = analyzer.remove_backgrounds([screenshot])

    assert stats["total_pixels"] == 100 * 100
    assert stats["foreground_pixels"] + stats["background_pixels"] == stats["total_pixels"]
    assert 0 <= stats["foreground_percentage"] <= 100
    assert 0 <= stats["background_percentage"] <= 100
    assert abs(stats["foreground_percentage"] + stats["background_percentage"] - 100) < 0.01


def test_visualize_mask():
    """Test mask visualization"""
    analyzer = BackgroundRemovalAnalyzer()
    screenshot = create_test_screenshot_with_static_icon()

    # Create a simple background mask
    background_mask = np.zeros((100, 100), dtype=np.uint8)
    background_mask[0:40, 0:40] = 255  # Mark corner as background

    vis = analyzer.visualize_mask(screenshot, background_mask)

    assert vis.shape == screenshot.shape
    assert len(vis.shape) == 3  # Should be BGR


def test_grayscale_input():
    """Test that grayscale images are handled correctly"""
    analyzer = BackgroundRemovalAnalyzer()

    # Create grayscale screenshot
    gray_screenshot = np.full((100, 100), 128, dtype=np.uint8)
    gray_screenshot[40:60, 40:60] = 255  # Bright square in center

    masked, stats = analyzer.remove_backgrounds([gray_screenshot])

    assert len(masked) == 1
    assert masked[0].shape[2] == 4  # Should be converted to RGBA


def test_morphological_cleanup():
    """Test that morphological cleanup can be disabled"""
    config_with = BackgroundRemovalConfig(apply_morphology=True)
    config_without = BackgroundRemovalConfig(apply_morphology=False)

    analyzer_with = BackgroundRemovalAnalyzer(config_with)
    analyzer_without = BackgroundRemovalAnalyzer(config_without)

    screenshot = create_test_screenshot_with_static_icon()

    masked_with, _ = analyzer_with.remove_backgrounds([screenshot])
    masked_without, _ = analyzer_without.remove_backgrounds([screenshot])

    # Both should produce results
    assert len(masked_with) == 1
    assert len(masked_without) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
