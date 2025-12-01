"""Tests for screen segmentation."""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from qontinui.perception.segmentation import ScreenSegmenter


class TestScreenSegmenter:
    """Test ScreenSegmenter class."""

    @pytest.fixture
    def segmenter(self):
        """Create ScreenSegmenter instance without SAM."""
        return ScreenSegmenter(use_sam=False)

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        # Create a simple test image with rectangles
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        # Draw some rectangles to segment
        cv2.rectangle(img, (50, 50), (150, 100), (0, 0, 0), -1)
        cv2.rectangle(img, (200, 150), (400, 250), (0, 0, 0), -1)
        cv2.rectangle(img, (100, 300), (300, 400), (0, 0, 0), -1)
        return img

    def test_init_without_sam(self):
        """Test initialization without SAM."""
        segmenter = ScreenSegmenter(use_sam=False)
        assert not segmenter.use_sam
        assert segmenter.sam is None
        assert segmenter.mask_generator is None

    @pytest.mark.skip(
        reason="SAM imports are conditional and occur inside __init__, making them difficult to patch. Requires actual segment_anything library."
    )
    @patch("qontinui.perception.segmentation.sam_model_registry")
    @patch("qontinui.perception.segmentation.SamAutomaticMaskGenerator")
    def test_init_with_sam_mock(self, mock_mask_gen, mock_registry):
        """Test initialization with SAM (mocked)."""
        mock_model = MagicMock()
        mock_registry.__getitem__.return_value = mock_model

        segmenter = ScreenSegmenter(use_sam=True)

        # SAM initialization should be attempted but may fail
        # This tests the initialization path
        assert segmenter is not None

    def test_segment_screen_opencv(self, segmenter, sample_image):
        """Test screen segmentation using OpenCV."""
        segments = segmenter.segment_screen(sample_image)

        assert isinstance(segments, list)
        assert len(segments) > 0

        for segment in segments:
            assert "id" in segment
            assert "bbox" in segment
            assert "image" in segment
            assert "area" in segment

            # Check bbox format
            bbox = segment["bbox"]
            assert len(bbox) == 4
            assert all(isinstance(v, int) for v in bbox)

            # Check image is numpy array
            assert isinstance(segment["image"], np.ndarray)

    def test_crop_image(self, segmenter):
        """Test image cropping."""
        img = np.ones((100, 100, 3), dtype=np.uint8)

        # Normal crop
        cropped = segmenter._crop_image(img, (10, 10, 20, 20))
        assert cropped.shape == (20, 20, 3)

        # Crop at edge
        cropped = segmenter._crop_image(img, (90, 90, 20, 20))
        assert cropped.shape[0] <= 20
        assert cropped.shape[1] <= 20

        # Crop outside bounds
        cropped = segmenter._crop_image(img, (110, 110, 20, 20))
        assert cropped.size == 0 or cropped.shape[0] == 0

    def test_detect_text_regions(self, segmenter, sample_image):
        """Test text region detection."""
        # Add some text-like regions
        img = sample_image.copy()
        cv2.rectangle(img, (50, 450), (300, 470), (0, 0, 0), -1)  # Text-like aspect ratio

        text_regions = segmenter.detect_text_regions(img)

        assert isinstance(text_regions, list)

        for region in text_regions:
            assert "id" in region
            assert "bbox" in region
            assert "image" in region
            assert "aspect_ratio" in region

            # Text regions should have wider aspect ratio
            if region["aspect_ratio"] > 0:
                assert region["aspect_ratio"] > 1.5

    def test_detect_buttons(self, segmenter, sample_image):
        """Test button detection."""
        buttons = segmenter.detect_buttons(sample_image)

        assert isinstance(buttons, list)

        for button in buttons:
            assert "id" in button
            assert "bbox" in button
            assert "image" in button
            assert "aspect_ratio" in button

            # Check button constraints
            x, y, w, h = button["bbox"]
            assert 20 < w < 300
            assert 15 < h < 100

    def test_segment_with_empty_image(self, segmenter):
        """Test segmentation with empty image."""
        empty_img = np.zeros((100, 100, 3), dtype=np.uint8)
        segments = segmenter.segment_screen(empty_img)

        assert isinstance(segments, list)
        # May or may not find segments in empty image

    def test_segment_with_complex_image(self, segmenter):
        """Test segmentation with complex image."""
        # Create complex image with multiple elements
        img = np.ones((600, 800, 3), dtype=np.uint8) * 255

        # Add various shapes
        cv2.circle(img, (100, 100), 50, (0, 0, 0), -1)
        cv2.ellipse(img, (400, 100), (80, 40), 0, 0, 360, (0, 0, 0), -1)
        cv2.rectangle(img, (50, 200), (200, 300), (0, 0, 0), -1)

        # Add lines
        cv2.line(img, (0, 400), (800, 400), (0, 0, 0), 2)
        cv2.line(img, (400, 0), (400, 600), (0, 0, 0), 2)

        segments = segmenter.segment_screen(img)

        assert isinstance(segments, list)
        assert len(segments) > 0

        # Check segments are sorted by area
        areas = [s["area"] for s in segments]
        assert areas == sorted(areas, reverse=True)
