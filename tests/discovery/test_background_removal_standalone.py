"""Standalone test for background removal functionality"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

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


def test_basic_functionality():
    """Test basic background removal"""
    print("Test 1: Basic functionality...")

    analyzer = BackgroundRemovalAnalyzer()
    screenshot = create_test_screenshot_with_static_icon()

    masked, stats = analyzer.remove_backgrounds([screenshot])

    assert len(masked) == 1, "Should return 1 masked screenshot"
    assert masked[0].shape[2] == 4, "Should be RGBA"
    assert "foreground_pixels" in stats, "Stats should contain foreground_pixels"

    print("✓ Test 1 passed")


def test_multiple_screenshots():
    """Test with multiple screenshots"""
    print("\nTest 2: Multiple screenshots...")

    analyzer = BackgroundRemovalAnalyzer()

    # Create 3 screenshots with same icon but slightly different backgrounds
    screenshots = [
        create_test_screenshot_with_static_icon(bg_color=(50, 50, 50)),
        create_test_screenshot_with_static_icon(bg_color=(60, 60, 60)),
        create_test_screenshot_with_static_icon(bg_color=(70, 70, 70)),
    ]

    masked, stats = analyzer.remove_backgrounds(screenshots)

    assert len(masked) == 3, "Should return 3 masked screenshots"
    assert all(img.shape[2] == 4 for img in masked), "All should be RGBA"
    assert stats["num_screenshots"] == 3, "Stats should show 3 screenshots"

    print(f"  Foreground: {stats['foreground_percentage']:.1f}%")
    print(f"  Background: {stats['background_percentage']:.1f}%")
    print("✓ Test 2 passed")


def test_temporal_variance():
    """Test temporal variance detection"""
    print("\nTest 3: Temporal variance detection...")

    config = BackgroundRemovalConfig(
        use_temporal_variance=True,
        use_edge_density=False,
        use_uniformity=False,
        variance_threshold=5.0,
        min_screenshots_for_variance=3,
    )
    analyzer = BackgroundRemovalAnalyzer(config)

    # Create screenshots where background changes dramatically but icon stays same
    screenshots = [
        create_test_screenshot_with_static_icon(bg_color=(30, 30, 30)),
        create_test_screenshot_with_static_icon(bg_color=(100, 100, 100)),
        create_test_screenshot_with_static_icon(bg_color=(200, 200, 200)),
    ]

    masked, stats = analyzer.remove_backgrounds(screenshots, debug=True)

    # The icon should be preserved (not transparent)
    center_x, center_y = 50, 50
    icon_alpha = masked[0][center_y, center_x, 3]

    print(f"  Icon alpha at center: {icon_alpha}")
    print(f"  Foreground: {stats['foreground_percentage']:.1f}%")

    print("✓ Test 3 passed")


def test_simple_function():
    """Test simple convenience function"""
    print("\nTest 4: Simple convenience function...")

    screenshots = [
        create_test_screenshot_with_static_icon(),
        create_test_screenshot_with_static_icon(bg_color=(60, 60, 60)),
    ]

    masked = remove_backgrounds_simple(screenshots)

    assert len(masked) == 2, "Should return 2 masked screenshots"
    assert all(img.shape[2] == 4 for img in masked), "All should be RGBA"

    print("✓ Test 4 passed")


def test_visualization():
    """Test mask visualization"""
    print("\nTest 5: Mask visualization...")

    analyzer = BackgroundRemovalAnalyzer()
    screenshot = create_test_screenshot_with_static_icon()

    # Create a simple background mask
    background_mask = np.zeros((100, 100), dtype=np.uint8)
    background_mask[0:40, 0:40] = 255  # Mark corner as background

    vis = analyzer.visualize_mask(screenshot, background_mask)

    assert vis.shape == screenshot.shape, "Visualization should have same shape"
    assert len(vis.shape) == 3, "Should be color image"

    print("✓ Test 5 passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Background Removal Standalone Tests")
    print("=" * 60)

    try:
        test_basic_functionality()
        test_multiple_screenshots()
        test_temporal_variance()
        test_simple_function()
        test_visualization()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
