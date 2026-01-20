import sys
from unittest.mock import MagicMock, patch

# Mock cv2 to avoid DLL issues
mock_cv2 = MagicMock()
sys.modules["cv2"] = mock_cv2

import numpy as np
import pytest

from qontinui.state_management.builders.state_machine_builder import (
    ImageMatchingStateMachineBuilder,
    TrackedImage,
)


class TestImageMatchingStateMachineBuilder:
    @pytest.fixture
    def builder(self, tmp_path):
        with patch("cv2.imread"):
            return ImageMatchingStateMachineBuilder(screenshots_dir=tmp_path)

    @patch("cv2.imread")
    def test_load_screenshots(self, mock_imread, builder, tmp_path):
        # Create a dummy image file
        img_path = tmp_path / "screen1.png"
        img_path.write_text("dummy image data")

        mock_img = MagicMock()
        mock_img.shape = (1080, 1920, 3)
        mock_imread.return_value = mock_img

        builder.load_screenshots(["screen1"])

        assert "screen1" in builder.screenshots
        assert builder.screenshots["screen1"] == mock_img

    @patch("cv2.imread")
    def test_extract_and_track_images(self, mock_imread, builder):
        # Setup mock screenshots
        # Use random noise to ensure high variance
        textured_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        builder.screenshots["screen1"] = textured_img

        elements_by_screenshot = {
            "screen1": [
                {
                    "id": "elem1",
                    "name": "Button",
                    "bbox": {"x": 10, "y": 10, "width": 20, "height": 20},
                    "element_type": "button",
                }
            ]
        }

        builder.extract_and_track_images(elements_by_screenshot)
        assert len(builder.tracked_images) == 1
        assert builder.tracked_images[0].name == "Button"
        assert builder.tracked_images[0].source_screenshot_id == "screen1"

    def test_reproduce_duplication_issue(self, builder):
        """
        Setup a scenario that reproduces the image duplication issue.
        In the Pure Comparison Model, this means extracting elements from two screens
        and verifying they are merged by visual identity.
        """
        # Texture for both to pass extraction variance filter
        textured_data = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)

        builder.screenshots["screen1"] = np.ones((100, 100, 3), dtype=np.uint8) * 128
        builder.screenshots["screen2"] = np.ones((100, 100, 3), dtype=np.uint8) * 128

        # Embed EXACT same texture in both screenshots
        builder.screenshots["screen1"][10:30, 10:30] = textured_data
        builder.screenshots["screen2"][10:30, 10:30] = textured_data

        elements = {
            "screen1": [
                {
                    "id": "btn1",
                    "name": "ButtonS1",
                    "bbox": {"x": 10, "y": 10, "width": 20, "height": 20},
                }
            ],
            "screen2": [
                {
                    "id": "btn2",
                    "name": "ButtonS2",
                    "bbox": {"x": 10, "y": 10, "width": 20, "height": 20},
                }
            ],
        }

        with (
            patch("cv2.matchTemplate") as mock_match,
            patch("cv2.minMaxLoc") as mock_loc,
            patch("cv2.resize") as mock_resize,
        ):
            # Mock perfect visual match
            mock_match.return_value = np.array([[1.0]])
            mock_loc.return_value = (0, 1.0, (0, 0), (0, 0))
            mock_resize.side_effect = lambda img, size: img

            builder.extract_and_track_images(elements)
            assert len(builder.tracked_images) == 2

            # Pure comparison deduplication
            builder.deduplicate_tracked_images()

        # Should now be 1 tracked image because they were visually identical
        assert len(builder.tracked_images) == 1
        assert "screen1" in builder.tracked_images[0].screens_found
        assert "screen2" in builder.tracked_images[0].screens_found

    def test_verify_deduplication(self, builder):
        """
        Verify that deduplicate_tracked_images consolidates identical images.
        """
        textured_data = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)

        btn1 = TrackedImage(
            id="btn1",
            name="Button1",
            source_screenshot_id="screen1",
            source_bbox={"x": 0, "y": 0, "width": 20, "height": 20},
            image_data=textured_data,
            screens_found={"screen1"},
        )

        btn2 = TrackedImage(
            id="btn2",
            name="Button2",
            source_screenshot_id="screen2",
            source_bbox={"x": 50, "y": 50, "width": 20, "height": 20},
            image_data=textured_data,
            screens_found={"screen2"},
        )

        builder.tracked_images = [btn1, btn2]

        with (
            patch("cv2.matchTemplate") as mock_match,
            patch("cv2.minMaxLoc") as mock_loc,
            patch("cv2.resize") as mock_resize,
        ):
            mock_match.return_value = np.array([[1.0]])
            mock_loc.return_value = (0, 1.0, (0, 0), (0, 0))
            mock_resize.side_effect = lambda img, size: img

            builder.deduplicate_tracked_images()

        assert len(builder.tracked_images) == 1
        assert set(builder.tracked_images[0].screens_found) == {"screen1", "screen2"}

    def test_extract_skips_low_variance(self, builder):
        """Verify that uniform regions are skipped during extraction."""
        # Mock screenshot - all same color (0 variance)
        builder.screenshots["screen1"] = np.ones((100, 100, 3), dtype=np.uint8) * 200

        elements = {
            "screen1": [{"id": "background", "bbox": {"x": 10, "y": 10, "width": 50, "height": 50}}]
        }

        builder.extract_and_track_images(elements)
        assert len(builder.tracked_images) == 0

    def test_robust_representative_selection(self, builder):
        """Verify that the highest-variance image is picked as representative."""
        # High variance data
        high_v_data = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
        # Low variance data
        low_v_data = np.ones((20, 20, 3), dtype=np.uint8) * 100
        low_v_data[0, 0] = 101

        e1 = TrackedImage(
            id="real_button",
            name="RealButton",
            source_screenshot_id="screen1",
            source_bbox={"x": 0, "y": 0, "width": 20, "height": 20},
            image_data=high_v_data,
            screens_found={"screen1"},
        )

        e2 = TrackedImage(
            id="faded_button",
            name="FadedButton",
            source_screenshot_id="screen2",
            source_bbox={"x": 0, "y": 0, "width": 20, "height": 20},
            image_data=low_v_data,
            screens_found={"screen2"},
        )

        builder.tracked_images = [e1, e2]

        with (
            patch("cv2.matchTemplate") as mock_match,
            patch("cv2.minMaxLoc") as mock_loc,
            patch("cv2.resize") as mock_resize,
        ):
            mock_match.return_value = np.array([[0.95]])
            mock_loc.return_value = (0, 0.95, (0, 0), (0, 0))
            mock_resize.side_effect = lambda img, size: img

            builder.deduplicate_tracked_images()

        assert len(builder.tracked_images) == 1
        # Should pick high variance e1
        assert builder.tracked_images[0].id == "real_button"
