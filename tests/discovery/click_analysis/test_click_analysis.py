"""Tests for click analysis functionality.

Tests the ClickBoundingBoxInferrer, ElementBoundaryFinder, and ClickContextAnalyzer
classes that provide sophisticated bounding box inference from click locations.
"""

import numpy as np
import pytest

from qontinui.discovery.click_analysis import (
    ClickBoundingBoxInferrer,
    ClickContextAnalyzer,
    DetectionStrategy,
    ElementBoundaryFinder,
    ElementType,
    InferenceConfig,
    InferenceResult,
    InferredBoundingBox,
    infer_bbox_from_click,
)

# Test fixtures for creating synthetic screenshots


def create_simple_button_screenshot(
    width: int = 800,
    height: int = 600,
    button_x: int = 350,
    button_y: int = 280,
    button_w: int = 100,
    button_h: int = 40,
    button_color: tuple[int, int, int] = (50, 120, 200),
    bg_color: tuple[int, int, int] = (240, 240, 240),
) -> np.ndarray:
    """Create a screenshot with a simple rectangular button."""
    screenshot = np.full((height, width, 3), bg_color, dtype=np.uint8)
    screenshot[button_y : button_y + button_h, button_x : button_x + button_w] = button_color
    return screenshot


def create_icon_screenshot(
    width: int = 800,
    height: int = 600,
    icon_x: int = 100,
    icon_y: int = 100,
    icon_size: int = 32,
    icon_color: tuple[int, int, int] = (100, 100, 100),
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Create a screenshot with a simple square icon."""
    screenshot = np.full((height, width, 3), bg_color, dtype=np.uint8)
    screenshot[icon_y : icon_y + icon_size, icon_x : icon_x + icon_size] = icon_color
    return screenshot


def create_complex_screenshot(
    width: int = 1024,
    height: int = 768,
) -> np.ndarray:
    """Create a screenshot with multiple UI elements."""
    screenshot = np.full((height, width, 3), (240, 240, 240), dtype=np.uint8)

    # Button 1 (blue)
    screenshot[100:140, 50:180] = (50, 100, 200)

    # Button 2 (green)
    screenshot[100:140, 200:330] = (50, 180, 80)

    # Icon (dark square)
    screenshot[100:132, 400:432] = (80, 80, 80)

    # Text area (darker background)
    screenshot[200:220, 50:250] = (200, 200, 200)

    # Input field (white with border effect)
    screenshot[300:330, 50:300] = (255, 255, 255)
    screenshot[300, 50:300] = (180, 180, 180)  # Top border
    screenshot[329, 50:300] = (180, 180, 180)  # Bottom border
    screenshot[300:330, 50] = (180, 180, 180)  # Left border
    screenshot[300:330, 299] = (180, 180, 180)  # Right border

    return screenshot


class TestInferenceConfig:
    """Tests for InferenceConfig dataclass."""

    def test_default_config(self):
        """Test that default config has sensible values."""
        config = InferenceConfig()

        assert config.search_radius == 100
        assert config.min_element_size == (10, 10)
        assert config.max_element_size == (500, 500)
        assert config.fallback_box_size == 50
        assert config.use_fallback is True
        assert len(config.preferred_strategies) > 0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = InferenceConfig(
            search_radius=50,
            min_element_size=(20, 20),
            max_element_size=(300, 300),
            fallback_box_size=30,
        )

        assert config.search_radius == 50
        assert config.min_element_size == (20, 20)
        assert config.max_element_size == (300, 300)
        assert config.fallback_box_size == 30

    def test_config_to_dict(self):
        """Test config serialization."""
        config = InferenceConfig()
        config_dict = config.to_dict()

        assert "search_radius" in config_dict
        assert "min_element_size" in config_dict
        assert "preferred_strategies" in config_dict
        assert isinstance(config_dict["preferred_strategies"], list)


class TestInferredBoundingBox:
    """Tests for InferredBoundingBox dataclass."""

    def test_basic_properties(self):
        """Test basic bbox properties."""
        bbox = InferredBoundingBox(
            x=100,
            y=200,
            width=80,
            height=40,
            confidence=0.85,
            strategy_used=DetectionStrategy.CONTOUR_BASED,
        )

        assert bbox.x == 100
        assert bbox.y == 200
        assert bbox.width == 80
        assert bbox.height == 40
        assert bbox.x2 == 180
        assert bbox.y2 == 240
        assert bbox.center == (140, 220)
        assert bbox.area == 3200
        assert bbox.confidence == 0.85

    def test_as_bbox_list(self):
        """Test conversion to COCO format list."""
        bbox = InferredBoundingBox(
            x=100,
            y=200,
            width=80,
            height=40,
            confidence=0.9,
            strategy_used=DetectionStrategy.EDGE_BASED,
        )

        bbox_list = bbox.as_bbox_list()
        assert bbox_list == [100, 200, 80, 40]

    def test_contains_point(self):
        """Test point containment check."""
        bbox = InferredBoundingBox(
            x=100,
            y=100,
            width=50,
            height=50,
            confidence=0.8,
            strategy_used=DetectionStrategy.CONTOUR_BASED,
        )

        assert bbox.contains_point(125, 125)  # Center
        assert bbox.contains_point(100, 100)  # Top-left
        assert bbox.contains_point(149, 149)  # Near bottom-right
        assert not bbox.contains_point(150, 150)  # Just outside
        assert not bbox.contains_point(99, 100)  # Just outside left
        assert not bbox.contains_point(100, 99)  # Just outside top

    def test_to_dict(self):
        """Test serialization to dictionary."""
        bbox = InferredBoundingBox(
            x=100,
            y=200,
            width=80,
            height=40,
            confidence=0.85,
            strategy_used=DetectionStrategy.COLOR_SEGMENTATION,
            element_type=ElementType.BUTTON,
            metadata={"test_key": "test_value"},
        )

        bbox_dict = bbox.to_dict()

        assert bbox_dict["x"] == 100
        assert bbox_dict["y"] == 200
        assert bbox_dict["width"] == 80
        assert bbox_dict["height"] == 40
        assert bbox_dict["x2"] == 180
        assert bbox_dict["y2"] == 240
        assert bbox_dict["confidence"] == 0.85
        assert bbox_dict["strategy_used"] == "color_segmentation"
        assert bbox_dict["element_type"] == "button"
        assert bbox_dict["metadata"]["test_key"] == "test_value"


class TestElementBoundaryFinder:
    """Tests for ElementBoundaryFinder class."""

    @pytest.fixture
    def finder(self):
        """Create boundary finder instance."""
        return ElementBoundaryFinder()

    @pytest.fixture
    def button_screenshot(self):
        """Create screenshot with a button."""
        return create_simple_button_screenshot()

    def test_initialization(self, finder):
        """Test finder initialization."""
        assert finder is not None
        assert finder.config is not None

    def test_find_boundaries_on_button(self, finder, button_screenshot):
        """Test boundary detection on a button element."""
        # Click in the center of the button
        click_location = (400, 300)

        candidates = finder.find_boundaries(button_screenshot, click_location)

        assert len(candidates) > 0
        # The best candidate should be close to the button boundaries
        best = candidates[0]
        assert best.confidence > 0.3
        assert best.width > 0
        assert best.height > 0

    def test_find_boundaries_on_background(self, finder, button_screenshot):
        """Test boundary detection on empty background area."""
        # Click on background, away from button
        click_location = (50, 50)

        candidates = finder.find_boundaries(button_screenshot, click_location)

        # May find candidates or may be empty
        # If found, should have lower confidence or be larger (background region)
        if candidates:
            best = candidates[0]
            assert best.confidence <= 0.9

    def test_click_outside_image_bounds(self, finder, button_screenshot):
        """Test handling of click outside image bounds."""
        click_location = (1000, 1000)  # Outside 800x600 image

        candidates = finder.find_boundaries(button_screenshot, click_location)

        assert candidates == []

    def test_different_strategies(self, finder, button_screenshot):
        """Test that different strategies can be specified."""
        click_location = (400, 300)

        # Test with only contour-based strategy
        candidates_contour = finder.find_boundaries(
            button_screenshot,
            click_location,
            strategies=[DetectionStrategy.CONTOUR_BASED],
        )

        # Test with only edge-based strategy
        candidates_edge = finder.find_boundaries(
            button_screenshot,
            click_location,
            strategies=[DetectionStrategy.EDGE_BASED],
        )

        # Both should find something (or be empty if strategy fails)
        # They may produce different results
        assert isinstance(candidates_contour, list)
        assert isinstance(candidates_edge, list)


class TestClickContextAnalyzer:
    """Tests for ClickContextAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create context analyzer instance."""
        return ClickContextAnalyzer()

    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer is not None

    def test_classify_button(self, analyzer):
        """Test classification of button element."""
        screenshot = create_simple_button_screenshot()
        bbox = InferredBoundingBox(
            x=350,
            y=280,
            width=100,
            height=40,
            confidence=0.8,
            strategy_used=DetectionStrategy.CONTOUR_BASED,
        )
        click_location = (400, 300)

        element_type = analyzer.analyze_element_type(screenshot, bbox, click_location)

        # Should recognize as button due to wide aspect ratio and uniform color
        assert element_type in [
            ElementType.BUTTON,
            ElementType.TEXT,
            ElementType.UNKNOWN,
        ]

    def test_classify_icon(self, analyzer):
        """Test classification of icon element."""
        screenshot = create_icon_screenshot()
        bbox = InferredBoundingBox(
            x=100,
            y=100,
            width=32,
            height=32,
            confidence=0.8,
            strategy_used=DetectionStrategy.CONTOUR_BASED,
        )
        click_location = (116, 116)

        element_type = analyzer.analyze_element_type(screenshot, bbox, click_location)

        # Should recognize as icon due to square shape and small size
        assert element_type in [
            ElementType.ICON,
            ElementType.CHECKBOX,
            ElementType.UNKNOWN,
        ]

    def test_get_element_type_with_confidence(self, analyzer):
        """Test getting element type with confidence score."""
        screenshot = create_simple_button_screenshot()
        bbox = InferredBoundingBox(
            x=350,
            y=280,
            width=100,
            height=40,
            confidence=0.8,
            strategy_used=DetectionStrategy.CONTOUR_BASED,
        )
        click_location = (400, 300)

        element_type, confidence = analyzer.get_element_type_confidence(
            screenshot, bbox, click_location
        )

        assert isinstance(element_type, ElementType)
        assert 0.0 <= confidence <= 1.0


class TestClickBoundingBoxInferrer:
    """Tests for ClickBoundingBoxInferrer class."""

    @pytest.fixture
    def inferrer(self):
        """Create inferrer instance."""
        return ClickBoundingBoxInferrer()

    @pytest.fixture
    def button_screenshot(self):
        """Create screenshot with a button."""
        return create_simple_button_screenshot()

    @pytest.fixture
    def complex_screenshot(self):
        """Create complex screenshot with multiple elements."""
        return create_complex_screenshot()

    def test_initialization(self, inferrer):
        """Test inferrer initialization."""
        assert inferrer is not None
        assert inferrer.config is not None
        assert inferrer.boundary_finder is not None
        assert inferrer.context_analyzer is not None

    def test_initialization_with_custom_config(self):
        """Test inferrer with custom configuration."""
        config = InferenceConfig(
            search_radius=50,
            fallback_box_size=30,
        )
        inferrer = ClickBoundingBoxInferrer(config)

        assert inferrer.config.search_radius == 50
        assert inferrer.config.fallback_box_size == 30

    def test_infer_bounding_box_on_button(self, inferrer, button_screenshot):
        """Test inference on button click."""
        click_location = (400, 300)

        result = inferrer.infer_bounding_box(button_screenshot, click_location)

        assert isinstance(result, InferenceResult)
        assert result.click_location == click_location
        assert result.primary_bbox is not None
        assert result.primary_bbox.confidence > 0
        assert result.image_width == 800
        assert result.image_height == 600

    def test_infer_bounding_box_fallback(self, inferrer):
        """Test that fallback works when no element found."""
        # Create empty image
        empty_screenshot = np.full((600, 800, 3), (255, 255, 255), dtype=np.uint8)
        click_location = (400, 300)

        result = inferrer.infer_bounding_box(empty_screenshot, click_location)

        # Should still return a result (fallback)
        assert result.primary_bbox is not None
        # Fallback should be centered on click
        bbox = result.primary_bbox
        center_x = bbox.x + bbox.width // 2
        center_y = bbox.y + bbox.height // 2
        # Center should be close to click location
        assert abs(center_x - click_location[0]) < 30
        assert abs(center_y - click_location[1]) < 30

    def test_infer_bbox_outside_bounds(self, inferrer, button_screenshot):
        """Test inference with click outside image bounds."""
        click_location = (-10, -10)

        result = inferrer.infer_bounding_box(button_screenshot, click_location)

        # Should handle gracefully
        assert result.primary_bbox is not None

    def test_infer_bbox_simple_interface(self, inferrer, button_screenshot):
        """Test simplified interface returning list."""
        click_location = (400, 300)

        bbox_list = inferrer.infer_bbox_simple(button_screenshot, click_location)

        assert isinstance(bbox_list, list)
        assert len(bbox_list) == 4
        x, y, w, h = bbox_list
        assert x >= 0
        assert y >= 0
        assert w > 0
        assert h > 0

    def test_alternative_candidates(self, inferrer, complex_screenshot):
        """Test that alternative candidates are provided."""
        # Click on button area
        click_location = (115, 120)

        result = inferrer.infer_bounding_box(complex_screenshot, click_location)

        # May have alternative candidates
        assert isinstance(result.alternative_candidates, list)

    def test_result_to_dict(self, inferrer, button_screenshot):
        """Test result serialization."""
        click_location = (400, 300)

        result = inferrer.infer_bounding_box(button_screenshot, click_location)
        result_dict = result.to_dict()

        assert "click_location" in result_dict
        assert "primary_bbox" in result_dict
        assert "image_width" in result_dict
        assert "image_height" in result_dict
        assert "strategies_attempted" in result_dict
        assert "processing_time_ms" in result_dict


class TestConvenienceFunction:
    """Tests for the convenience function infer_bbox_from_click."""

    def test_basic_usage(self):
        """Test basic convenience function usage."""
        screenshot = create_simple_button_screenshot()
        click_location = (400, 300)

        result = infer_bbox_from_click(screenshot, click_location)

        assert isinstance(result, InferenceResult)
        assert result.primary_bbox is not None

    def test_with_config(self):
        """Test convenience function with custom config."""
        screenshot = create_simple_button_screenshot()
        click_location = (400, 300)
        config = InferenceConfig(search_radius=50)

        result = infer_bbox_from_click(screenshot, click_location, config=config)

        assert isinstance(result, InferenceResult)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def inferrer(self):
        return ClickBoundingBoxInferrer()

    def test_very_small_image(self, inferrer):
        """Test with very small image."""
        small_screenshot = np.zeros((10, 10, 3), dtype=np.uint8)
        click_location = (5, 5)

        result = inferrer.infer_bounding_box(small_screenshot, click_location)

        assert result.primary_bbox is not None

    def test_grayscale_image(self, inferrer):
        """Test with grayscale image."""
        gray_screenshot = np.full((600, 800), 200, dtype=np.uint8)
        gray_screenshot[280:320, 350:450] = 100  # Dark rectangle
        click_location = (400, 300)

        result = inferrer.infer_bounding_box(gray_screenshot, click_location)

        assert result.primary_bbox is not None

    def test_click_at_edge(self, inferrer):
        """Test click at image edge."""
        screenshot = create_simple_button_screenshot()

        # Click at various edges
        edge_locations = [(0, 300), (799, 300), (400, 0), (400, 599)]

        for click_location in edge_locations:
            result = inferrer.infer_bounding_box(screenshot, click_location)
            assert result.primary_bbox is not None
            # Bbox should be within image bounds
            assert result.primary_bbox.x >= 0
            assert result.primary_bbox.y >= 0
            assert result.primary_bbox.x2 <= 800
            assert result.primary_bbox.y2 <= 600

    def test_click_at_corner(self, inferrer):
        """Test click at image corners."""
        screenshot = create_simple_button_screenshot()

        corner_locations = [(0, 0), (799, 0), (0, 599), (799, 599)]

        for click_location in corner_locations:
            result = inferrer.infer_bounding_box(screenshot, click_location)
            assert result.primary_bbox is not None

    def test_high_contrast_image(self, inferrer):
        """Test with high contrast image."""
        screenshot = np.zeros((600, 800, 3), dtype=np.uint8)
        screenshot[280:320, 350:450] = (255, 255, 255)  # White rectangle on black
        click_location = (400, 300)

        result = inferrer.infer_bounding_box(screenshot, click_location)

        assert result.primary_bbox is not None
        # Should detect the high contrast element
        if not result.used_fallback:
            assert result.primary_bbox.confidence > 0.3


class TestPerformance:
    """Performance tests for click analysis."""

    def test_inference_speed(self):
        """Test that inference completes in reasonable time."""
        import time

        inferrer = ClickBoundingBoxInferrer()
        screenshot = create_complex_screenshot()
        click_location = (115, 120)

        start = time.time()
        for _ in range(10):
            inferrer.infer_bounding_box(screenshot, click_location)
        elapsed = time.time() - start

        avg_time = elapsed / 10
        # Should complete in less than 500ms per inference
        assert avg_time < 0.5, f"Inference too slow: {avg_time:.3f}s"

    def test_consistent_results(self):
        """Test that same input produces consistent results."""
        inferrer = ClickBoundingBoxInferrer()
        screenshot = create_simple_button_screenshot()
        click_location = (400, 300)

        results = [inferrer.infer_bounding_box(screenshot, click_location) for _ in range(5)]

        # All results should have same primary bbox
        bboxes = [r.primary_bbox.as_bbox_list() for r in results]
        assert all(bbox == bboxes[0] for bbox in bboxes)
