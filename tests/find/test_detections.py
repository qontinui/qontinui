"""Tests for the Detections universal container.

Verifies:
- Construction from DetectionResult list
- Empty container
- Filtering by confidence, boolean mask, slice, index
- Merge and NMS
- Normalization
- Round-trip to/from DetectionResult
- Properties (area, center, width, height)
"""

import sys
from unittest.mock import MagicMock


class _MockFinder:
    """Auto-mock missing external packages so the import chain works."""

    _MOCK_PREFIXES = (
        "qontinui_schemas",
        "cv2",
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
import pytest

from qontinui.find.backends.base import DetectionResult
from qontinui.find.detections import Detections


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_results(n: int = 3) -> list[DetectionResult]:
    """Create n DetectionResult objects for testing."""
    results = []
    for i in range(n):
        results.append(
            DetectionResult(
                x=i * 100,
                y=i * 50,
                width=80,
                height=40,
                confidence=0.5 + i * 0.2,
                backend_name=f"backend_{i}",
                label=f"label_{i}" if i % 2 == 0 else None,
                metadata={"index": i},
            )
        )
    return results


# ---------------------------------------------------------------------------
# Tests: Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_empty(self):
        d = Detections.empty()
        assert len(d) == 0
        assert d.is_empty()
        assert d.xyxy.shape == (0, 4)
        assert d.confidence.shape == (0,)

    def test_from_detection_results(self):
        results = _make_results(3)
        d = Detections.from_detection_results(results)
        assert len(d) == 3
        assert not d.is_empty()
        # Check xyxy conversion (x, y, x+w, y+h)
        np.testing.assert_array_equal(d.xyxy[0], [0, 0, 80, 40])
        np.testing.assert_array_equal(d.xyxy[1], [100, 50, 180, 90])

    def test_from_empty_results(self):
        d = Detections.from_detection_results([])
        assert d.is_empty()

    def test_from_results_with_normalization(self):
        results = _make_results(2)
        d = Detections.from_detection_results(
            results, screen_width=1920, screen_height=1080
        )
        assert d.normalized_xyxy is not None
        assert d.normalized_xyxy.shape == (2, 4)
        # First box: x1=0/1920, y1=0/1080
        assert d.normalized_xyxy[0, 0] == pytest.approx(0.0)
        assert d.normalized_xyxy[0, 1] == pytest.approx(0.0)

    def test_validation_errors(self):
        with pytest.raises(ValueError, match="xyxy must have shape"):
            Detections(
                xyxy=np.array([1, 2, 3, 4]),
                confidence=np.array([0.9]),
                backend_name=np.array(["test"], dtype=object),
            )
        with pytest.raises(ValueError, match="confidence length"):
            Detections(
                xyxy=np.array([[1, 2, 3, 4]]),
                confidence=np.array([0.9, 0.8]),
                backend_name=np.array(["test"], dtype=object),
            )


# ---------------------------------------------------------------------------
# Tests: Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_area(self):
        d = Detections.from_detection_results(_make_results(2))
        areas = d.area
        assert len(areas) == 2
        # 80 * 40 = 3200 for both
        np.testing.assert_array_equal(areas, [3200, 3200])

    def test_center(self):
        d = Detections.from_detection_results(_make_results(1))
        # box = [0, 0, 80, 40], center = [40, 20]
        np.testing.assert_array_almost_equal(d.center[0], [40.0, 20.0])

    def test_width_height(self):
        d = Detections.from_detection_results(_make_results(1))
        assert d.width[0] == 80
        assert d.height[0] == 40


# ---------------------------------------------------------------------------
# Tests: Filtering
# ---------------------------------------------------------------------------


class TestFiltering:
    def test_boolean_mask(self):
        d = Detections.from_detection_results(_make_results(3))
        # Confidences: 0.5, 0.7, 0.9
        high = d[d.confidence > 0.6]
        assert len(high) == 2

    def test_slice(self):
        d = Detections.from_detection_results(_make_results(3))
        first_two = d[:2]
        assert len(first_two) == 2

    def test_integer_index(self):
        d = Detections.from_detection_results(_make_results(3))
        single = d[0]
        assert len(single) == 1

    def test_index_list(self):
        d = Detections.from_detection_results(_make_results(3))
        selected = d[[0, 2]]
        assert len(selected) == 2

    def test_sort_by_confidence(self):
        d = Detections.from_detection_results(_make_results(3))
        sorted_d = d.sort_by_confidence(descending=True)
        assert sorted_d.confidence[0] >= sorted_d.confidence[-1]


# ---------------------------------------------------------------------------
# Tests: Merge and NMS
# ---------------------------------------------------------------------------


class TestMergeAndNMS:
    def test_merge_empty(self):
        merged = Detections.merge([])
        assert merged.is_empty()

    def test_merge_two(self):
        d1 = Detections.from_detection_results(_make_results(2))
        d2 = Detections.from_detection_results(_make_results(3))
        merged = Detections.merge([d1, d2])
        assert len(merged) == 5

    def test_nms_removes_overlapping(self):
        # Two nearly identical boxes — NMS should keep only the higher-confidence one
        xyxy = np.array([[10, 10, 90, 90], [12, 12, 92, 92]], dtype=np.int_)
        conf = np.array([0.9, 0.7])
        names = np.array(["a", "b"], dtype=object)
        d = Detections(xyxy=xyxy, confidence=conf, backend_name=names)
        result = d.with_nms(iou_threshold=0.5)
        assert len(result) == 1
        assert result.confidence[0] == pytest.approx(0.9)

    def test_nms_keeps_non_overlapping(self):
        # Two boxes far apart — NMS should keep both
        xyxy = np.array([[0, 0, 50, 50], [200, 200, 250, 250]], dtype=np.int_)
        conf = np.array([0.9, 0.8])
        names = np.array(["a", "b"], dtype=object)
        d = Detections(xyxy=xyxy, confidence=conf, backend_name=names)
        result = d.with_nms(iou_threshold=0.5)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Tests: Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_to_and_from_detection_results(self):
        original = _make_results(3)
        d = Detections.from_detection_results(original, screen_width=1920, screen_height=1080)
        recovered = d.to_detection_results()
        assert len(recovered) == 3
        for orig, rec in zip(original, recovered):
            assert rec.x == orig.x
            assert rec.y == orig.y
            assert rec.width == orig.width
            assert rec.height == orig.height
            assert rec.confidence == pytest.approx(orig.confidence)
            assert rec.backend_name == orig.backend_name


# ---------------------------------------------------------------------------
# Tests: Normalization
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_normalize(self):
        d = Detections.from_detection_results(_make_results(1))
        assert d.normalized_xyxy is None
        normed = d.normalize(1920, 1080)
        assert normed.normalized_xyxy is not None
        assert normed.normalized_xyxy[0, 0] == pytest.approx(0.0)
        assert normed.normalized_xyxy[0, 2] == pytest.approx(80 / 1920)

    def test_normalize_zero_dims_noop(self):
        d = Detections.from_detection_results(_make_results(1))
        same = d.normalize(0, 0)
        assert same.normalized_xyxy is None


# ---------------------------------------------------------------------------
# Tests: Repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_nonempty(self):
        d = Detections.from_detection_results(_make_results(2))
        r = repr(d)
        assert "n=2" in r
        assert "confidence=" in r
        assert "backends=" in r

    def test_repr_empty(self):
        d = Detections.empty()
        assert "n=0" in repr(d)
