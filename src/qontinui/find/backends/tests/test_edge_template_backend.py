"""Tests for edge-aware template matching and coarse-to-fine matching."""

import numpy as np

from ..edge_template_backend import EdgeTemplateMatchBackend


def _make_button(
    width: int = 120,
    height: int = 40,
    bg_color: tuple[int, int, int] = (240, 240, 240),
    border_color: tuple[int, int, int] = (100, 100, 100),
    border_width: int = 2,
) -> np.ndarray:
    """Create a synthetic button image (BGR)."""
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    # Draw border
    img[:border_width, :] = border_color
    img[-border_width:, :] = border_color
    img[:, :border_width] = border_color
    img[:, -border_width:] = border_color
    return img


def _make_screenshot(
    button: np.ndarray,
    canvas_size: tuple[int, int] = (600, 800),
    canvas_color: tuple[int, int, int] = (255, 255, 255),
    position: tuple[int, int] = (200, 150),
) -> np.ndarray:
    """Place a button into a larger canvas (BGR)."""
    h, w = canvas_size
    canvas = np.full((h, w, 3), canvas_color, dtype=np.uint8)
    bh, bw = button.shape[:2]
    py, px = position
    canvas[py : py + bh, px : px + bw] = button
    return canvas


class TestEdgeTemplateMatchBackend:
    """Tests for EdgeTemplateMatchBackend."""

    def test_finds_exact_match(self):
        """Edge matching should find a template placed in a screenshot."""
        button = _make_button()
        screenshot = _make_screenshot(button)

        backend = EdgeTemplateMatchBackend()
        results = backend.find(
            needle=button,
            haystack=screenshot,
            config={"min_confidence": 0.7},
        )

        assert len(results) >= 1
        best = results[0]
        assert best.confidence >= 0.7
        assert best.backend_name == "edge_template"
        # position=(200, 150) means py=200, px=150 → x=150, y=200
        assert abs(best.x - 150) < 10
        assert abs(best.y - 200) < 10

    def test_finds_across_theme_change(self):
        """Edge matching should find a button even when colors are inverted."""
        # Light theme button
        light_button = _make_button(
            bg_color=(240, 240, 240),
            border_color=(80, 80, 80),
        )
        # Dark theme button — same shape, inverted colors
        dark_button = _make_button(
            bg_color=(30, 30, 30),
            border_color=(180, 180, 180),
        )
        # Place dark button in dark screenshot
        dark_screenshot = _make_screenshot(
            dark_button,
            canvas_color=(20, 20, 20),
        )

        backend = EdgeTemplateMatchBackend()
        # Search for the LIGHT button template in the DARK screenshot
        results = backend.find(
            needle=light_button,
            haystack=dark_screenshot,
            config={"min_confidence": 0.6},
        )

        assert len(results) >= 1
        best = results[0]
        assert best.confidence >= 0.6
        # position=(200, 150) means py=200, px=150 → x=150, y=200
        assert abs(best.x - 150) < 10
        assert abs(best.y - 200) < 10

    def test_no_false_positive(self):
        """Edge matching should return empty when template is not in image."""
        button = _make_button()
        # Empty canvas with no button
        empty = np.full((600, 800, 3), (200, 200, 200), dtype=np.uint8)

        backend = EdgeTemplateMatchBackend()
        results = backend.find(
            needle=button,
            haystack=empty,
            config={"min_confidence": 0.7},
        )

        assert len(results) == 0

    def test_search_region(self):
        """Should restrict search to specified region."""
        button = _make_button()
        screenshot = _make_screenshot(button, position=(200, 150))

        backend = EdgeTemplateMatchBackend()
        # Search in region that does NOT contain the button
        results = backend.find(
            needle=button,
            haystack=screenshot,
            config={
                "min_confidence": 0.7,
                "search_region": (500, 400, 200, 200),
            },
        )

        assert len(results) == 0

    def test_backend_properties(self):
        """Backend should report correct name, cost, and supported types."""
        backend = EdgeTemplateMatchBackend()
        assert backend.name == "edge_template"
        assert backend.estimated_cost_ms() == 40.0
        assert backend.supports("template") is True
        assert backend.supports("text") is False
        assert backend.is_available() is True

    def test_template_too_large_for_haystack(self):
        """Should return empty when template is larger than haystack."""
        big_template = _make_button(width=500, height=500)
        small_haystack = np.full((100, 100, 3), (200, 200, 200), dtype=np.uint8)

        backend = EdgeTemplateMatchBackend()
        results = backend.find(
            needle=big_template,
            haystack=small_haystack,
            config={"min_confidence": 0.7},
        )

        assert len(results) == 0

    def test_handles_pattern_object(self):
        """Should extract pixel_data from Pattern-like objects."""

        class FakePattern:
            def __init__(self, data: np.ndarray):
                self.pixel_data = data

        button = _make_button()
        screenshot = _make_screenshot(button)

        backend = EdgeTemplateMatchBackend()
        results = backend.find(
            needle=FakePattern(button),
            haystack=screenshot,
            config={"min_confidence": 0.7},
        )

        assert len(results) >= 1


class TestCascadeEdgeTemplateOrdering:
    """Test that edge_template is registered in the cascade."""

    def test_cascade_includes_edge_template(self):
        """CascadeDetector should include edge_template backend."""
        from ..cascade import CascadeDetector

        detector = CascadeDetector()
        names = [b.name for b in detector.backends]
        assert "edge_template" in names

    def test_edge_template_ordered_before_feature(self):
        """edge_template (40ms) should come before feature (100ms) in cascade."""
        from ..cascade import CascadeDetector

        detector = CascadeDetector()
        names = [b.name for b in detector.backends]

        if "edge_template" in names and "feature" in names:
            assert names.index("edge_template") < names.index("feature")


class TestCoarseToFineMatching:
    """Tests for the coarse-to-fine strategy in TemplateEngine."""

    def test_coarse_to_fine_same_result_as_single_scale(self):
        """Coarse-to-fine should find the same location as single-scale."""
        from qontinui.vision.verification.detection.template import TemplateEngine

        button = _make_button()
        # Large screenshot to trigger coarse-to-fine (>= 1920x1080)
        screenshot = _make_screenshot(
            button,
            canvas_size=(1080, 1920),
            position=(500, 300),
        )

        engine = TemplateEngine()

        # Single-scale (baseline)
        single = engine._find_single_scale(screenshot, button, 0.8)

        # Coarse-to-fine
        coarse = engine._find_coarse_to_fine(screenshot, button, 0.8)

        # Both should find the button at roughly the same location
        if single:
            assert len(coarse) >= 1
            # Centers should be within 5 pixels
            sc = single[0].center
            cc = coarse[0].center
            assert abs(sc[0] - cc[0]) < 5
            assert abs(sc[1] - cc[1]) < 5

    def test_coarse_to_fine_disabled_by_default(self):
        """Without config, coarse-to-fine should not activate."""
        from qontinui.vision.verification.detection.template import TemplateEngine

        engine = TemplateEngine()
        big_image = np.zeros((2160, 3840, 3), dtype=np.uint8)
        assert engine._should_use_coarse_to_fine(big_image) is False

    def test_merge_regions(self):
        """_merge_regions should combine overlapping boxes."""
        from qontinui.vision.verification.detection.template import TemplateEngine

        regions = [
            (0, 0, 100, 100),
            (50, 50, 100, 100),  # overlaps with first
            (500, 500, 50, 50),  # separate
        ]
        merged = TemplateEngine._merge_regions(regions)
        assert len(merged) == 2
        # First merged region should encompass both overlapping boxes
        big = [r for r in merged if r[2] > 100]
        assert len(big) == 1
        assert big[0][0] == 0  # x
        assert big[0][1] == 0  # y
        assert big[0][2] == 150  # width (0 to 150)
        assert big[0][3] == 150  # height (0 to 150)

    def test_coarse_to_fine_activates_with_config(self):
        """When config enables coarse-to-fine, it should activate for large images."""
        from qontinui.vision.verification.config import VisionConfig
        from qontinui.vision.verification.detection.template import TemplateEngine

        config = VisionConfig()
        config.detection.coarse_to_fine = True
        engine = TemplateEngine(config=config)

        big_image = np.zeros((2160, 3840, 3), dtype=np.uint8)
        assert engine._should_use_coarse_to_fine(big_image) is True

    def test_coarse_to_fine_skips_small_images_with_config(self):
        """Even with config enabled, small images should skip coarse-to-fine."""
        from qontinui.vision.verification.config import VisionConfig
        from qontinui.vision.verification.detection.template import TemplateEngine

        config = VisionConfig()
        config.detection.coarse_to_fine = True
        engine = TemplateEngine(config=config)

        small_image = np.zeros((720, 1280, 3), dtype=np.uint8)
        assert engine._should_use_coarse_to_fine(small_image) is False

    def test_edge_template_disabled_via_constructor(self):
        """EdgeTemplateMatchBackend should report unavailable when disabled."""
        backend = EdgeTemplateMatchBackend(enabled=False)
        assert backend.is_available() is False
