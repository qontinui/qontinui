import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock cv2 to avoid DLL issues
mock_cv2 = MagicMock()
sys.modules["cv2"] = mock_cv2

from qontinui.state_management.builders.state_machine_builder import (
    ImageMatchingStateMachineBuilder,
    TrackedImage,
)


class TestClusteringLogic:
    @pytest.fixture
    def builder(self, tmp_path):
        with patch("cv2.imread"):
            builder = ImageMatchingStateMachineBuilder(screenshots_dir=tmp_path)
            # Default threshold might be 0.8, we want to test if lowering it fixes things
            # but for this test we'll start with default behavior simulation
            return builder

    def test_identical_menu_bar_clustering(self, builder):
        """
        Simulate User Scenario:
        - 3 screens (Screen 1, Screen 2, Screen 3)
        - Each has a "Menu Bar" at the top (visually identical)
        - Expected: 1 State containing the "Menu Bar" image, present on Screens [1, 2, 3]
        - Current Bug: 3 separate states/images, one for each screen.
        """

        # 1. Setup Data
        # Menu bar is a 100x20 strip of random noise (high variance)
        menu_bar_texture = np.random.randint(0, 255, (20, 100, 3), dtype=np.uint8)

        # Create 3 tracked images (one from each screen) representing the SAME visual element
        img1 = TrackedImage(
            id="menu_s1",
            name="MenuBar",
            source_screenshot_id="screen1",
            source_bbox={"x": 0, "y": 0, "width": 100, "height": 20},
            image_data=menu_bar_texture,
            screens_found={"screen1"},
            extraction_category="static",
        )
        img2 = TrackedImage(
            id="menu_s2",
            name="MenuBar",
            source_screenshot_id="screen2",
            source_bbox={"x": 0, "y": 0, "width": 100, "height": 20},
            image_data=menu_bar_texture,
            screens_found={"screen2"},
            extraction_category="static",
        )
        img3 = TrackedImage(
            id="menu_s3",
            name="MenuBar",
            source_screenshot_id="screen3",
            source_bbox={"x": 0, "y": 0, "width": 100, "height": 20},
            image_data=menu_bar_texture,
            screens_found={"screen3"},
            extraction_category="static",
        )

        builder.tracked_images = [img1, img2, img3]

        # 2. Mock CV2 matching to return HIGH similarity (perfect match)
        with (
            patch("cv2.matchTemplate") as mock_match,
            patch("cv2.minMaxLoc") as mock_loc,
            patch("cv2.resize") as mock_resize,
        ):

            # matchTemplate returns a matrix where the max value is the score
            mock_match.return_value = np.array([[0.99]])  # 0.99 similarity
            # minMaxLoc returns (minVal, maxVal, minLoc, maxLoc)
            mock_loc.return_value = (0, 0.99, (0, 0), (0, 0))
            # resize just returns the image
            mock_resize.side_effect = lambda img, size: img

            # 3. Run Deduplication
            builder.deduplicate_tracked_images()

        # 4. Assertions
        # Expectation: ALL 3 merged into 1 unique TrackedImage
        assert (
            len(builder.tracked_images) == 1
        ), f"Expected 1 merged menu bar, but found {len(builder.tracked_images)}"

        merged_menu = builder.tracked_images[0]
        assert merged_menu.screens_found == {"screen1", "screen2", "screen3"}

        # 5. Run Clustering
        states_config = builder.cluster_into_states()

        # Expectation:
        # Since the Menu Bar appears on ALL screens, it should form a state for {screen1, screen2, screen3}
        # If there were other elements unique to screens, they would form other states.
        # But here we only have the menu bar.
        # So we expect 1 State, which appears on {"screen1", "screen2", "screen3"}

        assert len(states_config) == 1
        state = states_config[0]
        assert set(state["screensFound"]) == {"screen1", "screen2", "screen3"}
        # Updated behavior: We now create one stateImage PER SCREEN to ensure correct bounding boxes per screen.
        # So for 3 screens, we expect 3 stateImages in the merged state.
        assert (
            len(state["stateImages"]) == 3
        ), "Should have 3 stateImages (one per screen) for correct bbox tracking"

        # Verify screensFound covers all
        screens_found = set()
        for img in state["stateImages"]:
            screens_found.update(img["screensFound"])
        assert screens_found == {"screen1", "screen2", "screen3"}
        assert state["stateImages"][0]["name"] == "MenuBar"

    def test_slightly_different_sizes_clustering(self, builder):
        """
        Simulate slight bbox variations (1px off) causing resize artifacts or mismatches.
        """
        # Base texture 100x20
        texture_base = np.random.randint(0, 255, (20, 100, 3), dtype=np.uint8)
        # Slightly larger texture 101x21 (simulating padding/1px error)
        # We'll just fake it by having different shapes in the object

        img1 = TrackedImage(
            id="s1",
            name="Btn",
            source_screenshot_id="screen1",
            source_bbox={"x": 0, "y": 0, "width": 100, "height": 20},
            image_data=texture_base,
            screens_found={"screen1"},
        )
        img2 = TrackedImage(
            id="s2",
            name="Btn",
            source_screenshot_id="screen2",
            source_bbox={"x": 0, "y": 0, "width": 101, "height": 21},  # 1px larger
            image_data=np.random.randint(0, 255, (21, 101, 3), dtype=np.uint8),  # Different array
            screens_found={"screen2"},
        )

        builder.tracked_images = [img1, img2]

        with (
            patch("cv2.matchTemplate") as mock_match,
            patch("cv2.minMaxLoc") as mock_loc,
            patch("cv2.resize") as mock_resize,
        ):

            # Simulate a "Good Enough" match (e.g. 0.75) which might FAIL if threshold is 0.8
            mock_match.return_value = np.array([[0.75]])
            mock_loc.return_value = (0, 0.75, (0, 0), (0, 0))
            mock_resize.side_effect = lambda img, size: img

            # Set builder threshold to 0.8 initially (default)
            builder.similarity_threshold = 0.8

            builder.deduplicate_tracked_images()

            # Should FAIL validation (remain 2 images) because 0.75 < 0.8
            if len(builder.tracked_images) == 2:
                # PROVE it fails with high threshold
                pass
            else:
                pytest.fail("Expected deduplication to FAIL with threshold 0.8 and match 0.75")

            # Reset
            builder.tracked_images = [img1, img2]

            # Now Lower Threshold manually to simulate the fix
            builder.similarity_threshold = 0.7
            builder.deduplicate_tracked_images()

            # Should SUCCEED now
            assert len(builder.tracked_images) == 1, "Should merge with lower threshold"
