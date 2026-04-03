"""Test masked pattern matching functionality.

This test verifies that image masks work correctly to match only selected pixels,
ignoring masked-out regions even when they differ between template and screenshot.
"""

import os

import numpy as np
import pytest

from qontinui import Find, Image
from qontinui.model.element.pattern import Pattern

# Decorator for skipping tests that require a working X display
# This uses a lambda to defer evaluation until test time (after fixtures run)
skip_without_display = pytest.mark.skipif(
    lambda: os.environ.get("DISPLAY", "") == ":99",
    reason="Requires X display (run with Xvfb or in GUI environment)",
)


class TestMaskedMatching:
    """Test suite for masked pattern matching."""

    @skip_without_display
    def test_masked_pattern_ignores_masked_regions(self):
        """Test that masks correctly ignore differences in masked-out regions.

        Creates a template with a red square + green bar, but masks out the green bar.
        Screenshot has red square + BLUE bar (different from template).
        Should match because mask ignores the bar region.
        """
        # Create template: red square with green bar at top
        template_data = np.zeros((50, 50, 3), dtype=np.uint8)
        template_data[10:40, 10:40] = [255, 0, 0]  # Red square in center
        template_data[0:10, 0:50] = [0, 255, 0]  # Green bar at top

        # Create mask that only includes the red square (ignores green bar)
        mask = np.zeros((50, 50), dtype=np.float32)
        mask[10:40, 10:40] = 1.0  # Only match the red square

        # Create Pattern with mask
        template_pattern = Pattern(
            id="test-pattern",
            name="masked-pattern",
            pixel_data=template_data,
            mask=mask,
            width=50,
            height=50,
        )

        # Create screenshot with red square but DIFFERENT top bar (blue instead of green)
        screenshot_data = np.zeros((200, 200, 3), dtype=np.uint8)
        screenshot_data[50:100, 50:100] = [255, 0, 0]  # Red square at (50, 50)
        screenshot_data[40:50, 50:100] = [
            0,
            0,
            255,
        ]  # BLUE bar (different from template)
        screenshot = Image.from_numpy(screenshot_data, name="screenshot")

        # Execute find with mask - should match despite different top bars
        results = Find(template_pattern).similarity(0.85).screenshot(screenshot).execute()

        # Verify match was found
        assert len(results.matches) > 0, "Mask should allow match despite different top bars"
        assert results.matches.first.similarity >= 0.85, "Match confidence should be high"

    @skip_without_display
    def test_unmasked_pattern_fails_with_differences(self):
        """Test that without mask, differences in any region cause match to fail.

        Same setup as above but with all-ones mask (equivalent to no mask).
        Should NOT match because top bars differ.
        """
        # Create template: red square with green bar at top
        template_data = np.zeros((50, 50, 3), dtype=np.uint8)
        template_data[10:40, 10:40] = [255, 0, 0]  # Red square
        template_data[0:10, 0:50] = [0, 255, 0]  # Green bar

        # Create Pattern WITHOUT mask (all ones = no masking)
        template_pattern = Pattern(
            id="test-pattern",
            name="no-mask-pattern",
            pixel_data=template_data,
            mask=np.ones((50, 50), dtype=np.float32),
            width=50,
            height=50,
        )

        # Create screenshot with red square but BLUE bar
        screenshot_data = np.zeros((200, 200, 3), dtype=np.uint8)
        screenshot_data[50:100, 50:100] = [255, 0, 0]  # Red square
        screenshot_data[40:50, 50:100] = [0, 0, 255]  # BLUE bar
        screenshot = Image.from_numpy(screenshot_data, name="screenshot")

        # Execute find without mask - should NOT match due to different top bars
        results = Find(template_pattern).similarity(0.85).screenshot(screenshot).execute()

        # Verify match was NOT found
        assert len(results.matches) == 0, "Should not match when top bars differ without mask"

    @skip_without_display
    def test_mask_with_perfect_match(self):
        """Test that masked pattern matching works with perfect matches."""
        # Create simple template
        template_data = np.zeros((30, 30, 3), dtype=np.uint8)
        template_data[5:25, 5:25] = [255, 128, 0]  # Orange square

        # Create partial mask
        mask = np.zeros((30, 30), dtype=np.float32)
        mask[5:25, 5:25] = 1.0

        template_pattern = Pattern(
            id="orange-pattern",
            name="orange-square",
            pixel_data=template_data,
            mask=mask,
            width=30,
            height=30,
        )

        # Create screenshot with exact match
        screenshot_data = np.zeros((100, 100, 3), dtype=np.uint8)
        screenshot_data[20:50, 30:60] = template_data  # Perfect match at (30, 20)
        screenshot = Image.from_numpy(screenshot_data)

        # Find with mask
        results = Find(template_pattern).similarity(0.95).screenshot(screenshot).execute()

        # Should find perfect match
        assert len(results.matches) > 0, "Should find perfect match with mask"
        assert results.matches.first.similarity >= 0.95, (
            "Perfect match should have very high confidence"
        )

    @skip_without_display
    def test_image_converts_to_pattern_with_full_mask(self):
        """Test that when Image is passed to Find, it gets full mask (no masking)."""
        # Create simple image
        image_data = np.zeros((40, 40, 3), dtype=np.uint8)
        image_data[10:30, 10:30] = [100, 200, 150]  # Teal square
        test_image = Image.from_numpy(image_data, name="test-image")

        # Create matching screenshot
        screenshot_data = np.zeros((150, 150, 3), dtype=np.uint8)
        screenshot_data[50:90, 60:100] = image_data  # Exact match
        screenshot = Image.from_numpy(screenshot_data)

        # Find using Image (should auto-convert to Pattern with full mask)
        results = Find(test_image).similarity(0.90).screenshot(screenshot).execute()

        # Should find match (Image gets converted to Pattern with all-ones mask)
        assert len(results.matches) > 0, "Image should work in Find (converts to Pattern)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
