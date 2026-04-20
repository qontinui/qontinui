"""Tests for composable detection annotators.

Verifies:
- Each annotator produces a valid image of the same shape
- BoundingBoxAnnotator applies correct match/non-match colours
- LabelAnnotator draws text labels
- ConfidenceBarAnnotator draws confidence bars
- RegionAnnotator draws semi-transparent fills
- Chaining annotators works correctly
"""

import sys
from unittest.mock import MagicMock


class _MockFinder:
    """Auto-mock missing external packages so the import chain works."""

    _MOCK_PREFIXES = (
        "qontinui_schemas",
        "pyautogui",
        "screeninfo",
        "mss",
        "pygetwindow",
        "pynput",
        "Xlib",
    )

    def find_module(self, fullname, path=None):
        for prefix in self._MOCK_PREFIXES:
            if fullname == prefix or fullname.startswith(prefix + "."):
                return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        mod = MagicMock()
        sys.modules[fullname] = mod
        return mod


sys.meta_path.insert(0, _MockFinder())

import numpy as np

from qontinui.find.annotators import (
    BoundingBoxAnnotator,
    ConfidenceBarAnnotator,
    LabelAnnotator,
    RegionAnnotator,
)
from qontinui.find.detections import Detections

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_scene(w: int = 640, h: int = 480) -> np.ndarray:
    """Create a blank BGR scene."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_detections(n: int = 3) -> Detections:
    """Create n test detections spread across the scene."""
    xyxy = np.array(
        [[50 + i * 150, 50 + i * 80, 130 + i * 150, 110 + i * 80] for i in range(n)],
        dtype=np.int_,
    )
    confidence = np.linspace(0.3, 0.95, n)
    backend_name = np.array([f"backend_{i}" for i in range(n)], dtype=object)
    return Detections(xyxy=xyxy, confidence=confidence, backend_name=backend_name)


# ---------------------------------------------------------------------------
# Tests: BoundingBoxAnnotator
# ---------------------------------------------------------------------------


class TestBoundingBoxAnnotator:
    def test_returns_same_shape(self):
        scene = _make_scene()
        dets = _make_detections()
        ann = BoundingBoxAnnotator()
        result = ann.annotate(scene, dets)
        assert result.shape == scene.shape

    def test_modifies_scene_in_place(self):
        scene = _make_scene()
        dets = _make_detections()
        ann = BoundingBoxAnnotator()
        result = ann.annotate(scene, dets)
        assert result is scene

    def test_draws_pixels(self):
        scene = _make_scene()
        original_sum = scene.sum()
        ann = BoundingBoxAnnotator()
        ann.annotate(scene, _make_detections())
        assert scene.sum() > original_sum  # Pixels were drawn

    def test_confidence_threshold_colours(self):
        scene = _make_scene(400, 400)
        # One detection with low confidence, one with high
        xyxy = np.array([[10, 10, 100, 100], [200, 200, 300, 300]], dtype=np.int_)
        conf = np.array([0.3, 0.9])
        names = np.array(["a", "b"], dtype=object)
        dets = Detections(xyxy=xyxy, confidence=conf, backend_name=names)

        ann = BoundingBoxAnnotator(
            color_match=(0, 255, 0),
            color_non_match=(0, 0, 255),
            confidence_threshold=0.5,
        )
        ann.annotate(scene, dets)

        # Check a pixel on the low-confidence box border — should be red (BGR)
        # Top edge of first box at y=10, between x=10..100
        red_pixel = scene[10, 50]
        assert red_pixel[2] > 0  # R channel present (non-match)

        # Top edge of second box at y=200, between x=200..300
        green_pixel = scene[200, 250]
        assert green_pixel[1] > 0  # G channel present (match)

    def test_empty_detections(self):
        scene = _make_scene()
        original = scene.copy()
        ann = BoundingBoxAnnotator()
        ann.annotate(scene, Detections.empty())
        np.testing.assert_array_equal(scene, original)


# ---------------------------------------------------------------------------
# Tests: LabelAnnotator
# ---------------------------------------------------------------------------


class TestLabelAnnotator:
    def test_returns_same_shape(self):
        scene = _make_scene()
        dets = _make_detections()
        ann = LabelAnnotator()
        result = ann.annotate(scene, dets)
        assert result.shape == scene.shape

    def test_draws_pixels(self):
        scene = _make_scene()
        original_sum = scene.sum()
        ann = LabelAnnotator()
        ann.annotate(scene, _make_detections())
        assert scene.sum() > original_sum

    def test_custom_label_fn(self):
        scene = _make_scene()
        ann = LabelAnnotator(label_fn=lambda d, i: "custom")
        result = ann.annotate(scene, _make_detections(1))
        assert result.shape == scene.shape

    def test_empty_detections(self):
        scene = _make_scene()
        original = scene.copy()
        ann = LabelAnnotator()
        ann.annotate(scene, Detections.empty())
        np.testing.assert_array_equal(scene, original)


# ---------------------------------------------------------------------------
# Tests: ConfidenceBarAnnotator
# ---------------------------------------------------------------------------


class TestConfidenceBarAnnotator:
    def test_returns_same_shape(self):
        scene = _make_scene()
        dets = _make_detections()
        ann = ConfidenceBarAnnotator()
        result = ann.annotate(scene, dets)
        assert result.shape == scene.shape

    def test_draws_pixels(self):
        # Use a non-black scene so the bar is visible
        scene = np.full((480, 640, 3), 128, dtype=np.uint8)
        ann = ConfidenceBarAnnotator()
        original_sum = scene.sum()
        ann.annotate(scene, _make_detections())
        # Pixels should change (bar drawn)
        assert scene.sum() != original_sum

    def test_empty_detections(self):
        scene = _make_scene()
        original = scene.copy()
        ann = ConfidenceBarAnnotator()
        ann.annotate(scene, Detections.empty())
        np.testing.assert_array_equal(scene, original)


# ---------------------------------------------------------------------------
# Tests: RegionAnnotator
# ---------------------------------------------------------------------------


class TestRegionAnnotator:
    def test_returns_same_shape(self):
        scene = _make_scene()
        dets = _make_detections()
        ann = RegionAnnotator()
        result = ann.annotate(scene, dets)
        assert result.shape == scene.shape

    def test_draws_filled_regions(self):
        scene = _make_scene()
        ann = RegionAnnotator(color=(255, 200, 0), opacity=0.5)
        ann.annotate(scene, _make_detections())
        assert scene.sum() > 0

    def test_border_drawn(self):
        scene = _make_scene()
        ann = RegionAnnotator(border_thickness=2)
        ann.annotate(scene, _make_detections())
        assert scene.sum() > 0

    def test_empty_detections(self):
        scene = _make_scene()
        original = scene.copy()
        ann = RegionAnnotator()
        ann.annotate(scene, Detections.empty())
        np.testing.assert_array_equal(scene, original)


# ---------------------------------------------------------------------------
# Tests: Chaining
# ---------------------------------------------------------------------------


class TestChaining:
    def test_chain_all_annotators(self):
        scene = _make_scene()
        dets = _make_detections()

        annotators = [
            RegionAnnotator(opacity=0.2),
            BoundingBoxAnnotator(),
            LabelAnnotator(),
            ConfidenceBarAnnotator(),
        ]

        for ann in annotators:
            scene = ann.annotate(scene, dets)

        assert scene.shape == (480, 640, 3)
        assert scene.sum() > 0  # All annotators drew something

    def test_chain_with_empty_detections(self):
        scene = _make_scene()
        original = scene.copy()
        dets = Detections.empty()

        for ann in [BoundingBoxAnnotator(), LabelAnnotator(), ConfidenceBarAnnotator()]:
            scene = ann.annotate(scene, dets)

        np.testing.assert_array_equal(scene, original)
