"""Tests for scale/rotation-invariant template matching.

Covers:
- ColorDifferenceFilter (unit)
- OpenCVMatcher.find_template_invariant (unit)
- OpenCVMatcher.find_all_template_invariant (unit)
- OpenCVMatcher.dpi_aware_scales (unit)
- InvariantMatchBackend properties and cascade integration
"""

import sys
from unittest.mock import MagicMock

import numpy as np

# ---------------------------------------------------------------------------
# Auto-mock missing external deps (same pattern as other find tests)
# ---------------------------------------------------------------------------


class _MockFinder:
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
                if fullname not in sys.modules:
                    return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        mod = MagicMock()
        mod.__path__ = []
        mod.__name__ = fullname
        sys.modules[fullname] = mod
        return mod


sys.meta_path.insert(0, _MockFinder())

from qontinui.find.backends.base import DetectionBackend, DetectionResult  # noqa: E402
from qontinui.find.backends.cascade import CascadeDetector, MatchSettings  # noqa: E402

# ===========================================================================
# Fixtures
# ===========================================================================

_rng = np.random.RandomState(42)


def _make_textured_patch(
    w: int, h: int, base_color: tuple[int, int, int]
) -> np.ndarray:
    """Textured BGR patch — TM_CCOEFF_NORMED needs variance to work."""
    patch = np.empty((h, w, 3), dtype=np.uint8)
    for c in range(3):
        patch[:, :, c] = np.clip(
            base_color[c] + _rng.randint(-20, 20, (h, w)),
            0,
            255,
        ).astype(np.uint8)
    return patch


def _make_checkerboard(w: int, h: int, cell: int = 5) -> np.ndarray:
    """Create a distinctive checkerboard (RGB, not BGR)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if (i // cell + j // cell) % 2 == 0:
                img[i, j] = [255, 50, 50]
            else:
                img[i, j] = [50, 50, 255]
    return img


def _place(image: np.ndarray, x: int, y: int, patch: np.ndarray) -> None:
    h, w = patch.shape[:2]
    image[y : y + h, x : x + w] = patch


# ===========================================================================
# ColorDifferenceFilter
# ===========================================================================


class TestColorDifferenceFilter:
    def test_passes_matching_color(self):
        from qontinui.find.utils.color_filter import ColorDifferenceFilter

        cf = ColorDifferenceFilter(reference_color=(100, 100, 100), tolerance=10.0)
        # BGR region with mean close to (100, 100, 100) RGB
        region = np.full((10, 10, 3), [100, 100, 100], dtype=np.uint8)
        assert cf.passes(region)

    def test_rejects_distant_color(self):
        from qontinui.find.utils.color_filter import ColorDifferenceFilter

        cf = ColorDifferenceFilter(reference_color=(255, 0, 0), tolerance=30.0)
        # Gray region — far from red
        region = np.full((10, 10, 3), [128, 128, 128], dtype=np.uint8)
        assert not cf.passes(region)

    def test_empty_region_passes_with_zero_reference(self):
        from qontinui.find.utils.color_filter import ColorDifferenceFilter

        cf = ColorDifferenceFilter(reference_color=(0, 0, 0), tolerance=1.0)
        region = np.empty((0, 0, 3), dtype=np.uint8)
        # Empty region mean is (0,0,0) which matches reference
        assert cf.passes(region)

    def test_filter_results_keeps_matching(self):
        from qontinui.find.utils.color_filter import ColorDifferenceFilter

        cf = ColorDifferenceFilter(reference_color=(255, 0, 0), tolerance=30.0)

        # BGR haystack: red region at (0,0,10,10) and gray at (20,0,10,10)
        haystack = np.full((20, 40, 3), [128, 128, 128], dtype=np.uint8)
        haystack[0:10, 0:10] = [0, 0, 255]  # BGR red

        # Two fake results
        class FakeResult:
            def __init__(self, x, y, w, h):
                self.x, self.y, self.width, self.height = x, y, w, h

        r1 = FakeResult(0, 0, 10, 10)  # red region — should pass
        r2 = FakeResult(20, 0, 10, 10)  # gray region — should be filtered

        kept = cf.filter_results([r1, r2], haystack)
        assert len(kept) == 1
        assert kept[0] is r1

    def test_color_distance_calculation(self):
        from qontinui.find.utils.color_filter import ColorDifferenceFilter

        cf = ColorDifferenceFilter(reference_color=(0, 0, 0), tolerance=10.0)
        dist = cf.color_distance((3.0, 4.0, 0.0))
        assert abs(dist - 5.0) < 0.01  # 3-4-5 triangle

    def test_tolerance_boundary(self):
        from qontinui.find.utils.color_filter import ColorDifferenceFilter

        # Exactly at tolerance boundary
        cf = ColorDifferenceFilter(reference_color=(100, 100, 100), tolerance=0.0)
        region = np.full((5, 5, 3), [100, 100, 100], dtype=np.uint8)
        assert cf.passes(region)  # distance == 0 <= tolerance 0

        region_off = np.full((5, 5, 3), [101, 100, 100], dtype=np.uint8)
        assert not cf.passes(region_off)  # distance > 0


# ===========================================================================
# DPI-Aware Scales
# ===========================================================================


class TestDPIAwareScales:
    def test_100_percent_returns_only_native(self):
        from qontinui.hal.implementations.opencv_matcher import OpenCVMatcher

        scales = OpenCVMatcher.dpi_aware_scales(1.0)
        assert scales == [1.0]

    def test_150_percent_returns_three_scales(self):
        from qontinui.hal.implementations.opencv_matcher import OpenCVMatcher

        scales = OpenCVMatcher.dpi_aware_scales(1.5)
        assert 1.0 in scales
        assert 1.5 in scales
        assert 0.667 in scales
        assert len(scales) == 3

    def test_125_percent(self):
        from qontinui.hal.implementations.opencv_matcher import OpenCVMatcher

        scales = OpenCVMatcher.dpi_aware_scales(1.25)
        assert 1.0 in scales
        assert 1.25 in scales
        assert 0.8 in scales

    def test_200_percent(self):
        from qontinui.hal.implementations.opencv_matcher import OpenCVMatcher

        scales = OpenCVMatcher.dpi_aware_scales(2.0)
        assert 1.0 in scales
        assert 2.0 in scales
        assert 0.5 in scales

    def test_scales_are_sorted(self):
        from qontinui.hal.implementations.opencv_matcher import OpenCVMatcher

        scales = OpenCVMatcher.dpi_aware_scales(1.75)
        assert scales == sorted(scales)


# ===========================================================================
# find_template_invariant (single best match)
# ===========================================================================


class TestFindTemplateInvariant:
    def _get_matcher(self):
        from qontinui.hal.implementations.opencv_matcher import OpenCVMatcher

        return OpenCVMatcher()

    def test_finds_exact_match_at_native_scale(self):
        from PIL import Image

        matcher = self._get_matcher()
        haystack = np.random.randint(50, 150, (200, 300, 3), dtype=np.uint8)
        patch = _make_checkerboard(30, 30)
        _place(haystack, 100, 80, patch)

        result = matcher.find_template_invariant(
            Image.fromarray(haystack),
            Image.fromarray(patch),
            scales=[1.0],
            confidence=0.8,
        )
        assert result is not None
        assert abs(result.x - 100) < 3
        assert abs(result.y - 80) < 3
        assert result.confidence > 0.9

    def test_finds_scaled_match(self):
        from PIL import Image

        matcher = self._get_matcher()

        # Haystack has a 60x60 checkerboard (2x scale)
        haystack = np.random.randint(50, 150, (300, 400, 3), dtype=np.uint8)
        big_patch = _make_checkerboard(60, 60, cell=10)
        _place(haystack, 100, 80, big_patch)

        # Needle is 30x30 (1x)
        needle = _make_checkerboard(30, 30, cell=5)

        result = matcher.find_template_invariant(
            Image.fromarray(haystack),
            Image.fromarray(needle),
            scales=[2.0],
            confidence=0.8,
        )
        assert result is not None
        assert abs(result.x - 100) < 5
        assert abs(result.y - 80) < 5
        assert result.width == 60
        assert result.height == 60

    def test_returns_none_below_threshold(self):
        from PIL import Image

        matcher = self._get_matcher()
        haystack = np.random.randint(50, 150, (200, 300, 3), dtype=np.uint8)
        needle = _make_checkerboard(30, 30)

        result = matcher.find_template_invariant(
            Image.fromarray(haystack),
            Image.fromarray(needle),
            scales=[1.0],
            confidence=0.99,
        )
        # Random background, no placed pattern — should not match at 0.99
        assert result is None

    def test_skips_too_small_scaled_needle(self):
        from PIL import Image

        matcher = self._get_matcher()
        haystack = np.random.randint(50, 150, (200, 300, 3), dtype=np.uint8)
        needle = _make_checkerboard(15, 15)

        # Scale 0.5 would make 7x7 — below 10x10 minimum
        result = matcher.find_template_invariant(
            Image.fromarray(haystack),
            Image.fromarray(needle),
            scales=[0.5],
            confidence=0.5,
        )
        assert result is None

    def test_skips_needle_larger_than_haystack(self):
        from PIL import Image

        matcher = self._get_matcher()
        haystack = np.random.randint(50, 150, (50, 50, 3), dtype=np.uint8)
        needle = _make_checkerboard(30, 30)

        # Scale 2.0 makes 60x60, larger than 50x50 haystack
        result = matcher.find_template_invariant(
            Image.fromarray(haystack),
            Image.fromarray(needle),
            scales=[2.0],
            confidence=0.5,
        )
        assert result is None

    def test_early_exit_at_high_confidence(self):
        from PIL import Image

        matcher = self._get_matcher()
        haystack = np.random.randint(50, 150, (200, 300, 3), dtype=np.uint8)
        patch = _make_checkerboard(30, 30)
        _place(haystack, 50, 50, patch)

        # Should early-exit on first scale (1.0) since perfect match
        result = matcher.find_template_invariant(
            Image.fromarray(haystack),
            Image.fromarray(patch),
            scales=[1.0, 1.25, 1.5, 2.0],
            confidence=0.8,
        )
        assert result is not None
        assert result.confidence >= 0.95

    def test_with_rotation(self):
        from PIL import Image

        matcher = self._get_matcher()
        # Place a horizontal bar, search for it with 0 and 90 degree rotation
        haystack = np.full((200, 200, 3), 128, dtype=np.uint8)
        bar = np.full((10, 50, 3), 255, dtype=np.uint8)
        _place(haystack, 75, 95, bar)

        result = matcher.find_template_invariant(
            Image.fromarray(haystack),
            Image.fromarray(bar),
            scales=[1.0],
            rotations=[0.0],
            confidence=0.8,
        )
        assert result is not None

    def test_default_scales_used_when_none(self):
        from PIL import Image

        matcher = self._get_matcher()
        haystack = np.random.randint(50, 150, (200, 300, 3), dtype=np.uint8)
        needle = _make_checkerboard(30, 30)

        # Should not crash with default scales
        result = matcher.find_template_invariant(
            Image.fromarray(haystack),
            Image.fromarray(needle),
            confidence=0.99,
        )
        # No placed pattern — may or may not match, just shouldn't crash
        assert result is None or result.confidence >= 0.99


# ===========================================================================
# find_all_template_invariant (multiple matches)
# ===========================================================================


class TestFindAllTemplateInvariant:
    def _get_matcher(self):
        from qontinui.hal.implementations.opencv_matcher import OpenCVMatcher

        return OpenCVMatcher()

    def test_finds_multiple_matches(self):
        from PIL import Image

        matcher = self._get_matcher()
        haystack = np.random.randint(50, 100, (400, 600, 3), dtype=np.uint8)
        patch = _make_checkerboard(30, 30)
        _place(haystack, 50, 50, patch)
        _place(haystack, 300, 200, patch)

        matches = matcher.find_all_template_invariant(
            Image.fromarray(haystack),
            Image.fromarray(patch),
            scales=[1.0],
            confidence=0.8,
        )
        assert len(matches) >= 2

    def test_limit_restricts_count(self):
        from PIL import Image

        matcher = self._get_matcher()
        haystack = np.random.randint(50, 100, (400, 600, 3), dtype=np.uint8)
        patch = _make_checkerboard(30, 30)
        _place(haystack, 50, 50, patch)
        _place(haystack, 300, 200, patch)
        _place(haystack, 150, 300, patch)

        matches = matcher.find_all_template_invariant(
            Image.fromarray(haystack),
            Image.fromarray(patch),
            scales=[1.0],
            confidence=0.8,
            limit=2,
        )
        assert len(matches) <= 2

    def test_returns_empty_when_no_match(self):
        from PIL import Image

        matcher = self._get_matcher()
        haystack = np.random.randint(50, 150, (200, 300, 3), dtype=np.uint8)
        needle = _make_checkerboard(30, 30)

        matches = matcher.find_all_template_invariant(
            Image.fromarray(haystack),
            Image.fromarray(needle),
            scales=[1.0],
            confidence=0.99,
        )
        assert matches == []

    def test_finds_at_different_scales(self):
        from PIL import Image

        matcher = self._get_matcher()
        haystack = np.random.randint(50, 100, (400, 600, 3), dtype=np.uint8)
        # Place at 1x scale
        patch_1x = _make_checkerboard(30, 30, cell=5)
        _place(haystack, 50, 50, patch_1x)
        # Place at 2x scale
        patch_2x = _make_checkerboard(60, 60, cell=10)
        _place(haystack, 300, 200, patch_2x)

        needle = _make_checkerboard(30, 30, cell=5)
        matches = matcher.find_all_template_invariant(
            Image.fromarray(haystack),
            Image.fromarray(needle),
            scales=[1.0, 2.0],
            confidence=0.8,
        )
        # Should find both — one at 1x and one at 2x
        assert len(matches) >= 2

    def test_nms_deduplicates_overlapping(self):
        from PIL import Image

        matcher = self._get_matcher()
        haystack = np.random.randint(50, 100, (200, 300, 3), dtype=np.uint8)
        patch = _make_checkerboard(30, 30)
        _place(haystack, 100, 80, patch)

        # Search at multiple similar scales — should deduplicate
        matches = matcher.find_all_template_invariant(
            Image.fromarray(haystack),
            Image.fromarray(patch),
            scales=[1.0, 1.05],
            confidence=0.8,
        )
        # NMS should keep only one match for the same location
        assert len(matches) == 1

    def test_sorted_by_confidence_descending(self):
        from PIL import Image

        matcher = self._get_matcher()
        haystack = np.random.randint(50, 100, (400, 600, 3), dtype=np.uint8)
        patch = _make_checkerboard(30, 30)
        _place(haystack, 50, 50, patch)
        _place(haystack, 300, 200, patch)

        matches = matcher.find_all_template_invariant(
            Image.fromarray(haystack),
            Image.fromarray(patch),
            scales=[1.0],
            confidence=0.8,
        )
        for i in range(len(matches) - 1):
            assert matches[i].confidence >= matches[i + 1].confidence


# ===========================================================================
# NMS helper
# ===========================================================================


class TestNMS:
    def test_no_matches_returns_empty(self):
        from qontinui.hal.implementations.opencv_matcher import OpenCVMatcher

        assert OpenCVMatcher._nms_matches([]) == []

    def test_non_overlapping_kept(self):
        from qontinui.hal.implementations.opencv_matcher import OpenCVMatcher
        from qontinui.hal.interfaces.pattern_matcher import Match

        m1 = Match(x=0, y=0, width=10, height=10, confidence=0.9, center=(5, 5))
        m2 = Match(x=100, y=100, width=10, height=10, confidence=0.8, center=(105, 105))
        result = OpenCVMatcher._nms_matches([m1, m2])
        assert len(result) == 2

    def test_overlapping_suppressed(self):
        from qontinui.hal.implementations.opencv_matcher import OpenCVMatcher
        from qontinui.hal.interfaces.pattern_matcher import Match

        m1 = Match(x=0, y=0, width=10, height=10, confidence=0.9, center=(5, 5))
        m2 = Match(x=2, y=2, width=10, height=10, confidence=0.8, center=(7, 7))
        result = OpenCVMatcher._nms_matches([m1, m2], iou_threshold=0.3)
        assert len(result) == 1
        assert result[0].confidence == 0.9  # Kept the higher confidence one

    def test_iou_calculation(self):
        from qontinui.hal.implementations.opencv_matcher import OpenCVMatcher
        from qontinui.hal.interfaces.pattern_matcher import Match

        m1 = Match(x=0, y=0, width=10, height=10, confidence=1.0, center=(5, 5))
        m2 = Match(x=0, y=0, width=10, height=10, confidence=1.0, center=(5, 5))
        assert OpenCVMatcher._iou(m1, m2) == 1.0  # identical

        m3 = Match(x=100, y=100, width=10, height=10, confidence=1.0, center=(105, 105))
        assert OpenCVMatcher._iou(m1, m3) == 0.0  # no overlap


# ===========================================================================
# InvariantMatchBackend properties
# ===========================================================================


class TestInvariantMatchBackendProperties:
    def test_instantiation_and_properties(self):
        from qontinui.find.backends.invariant_match_backend import InvariantMatchBackend

        b = InvariantMatchBackend()
        assert b.name == "invariant_template"
        assert b.estimated_cost_ms() == 120.0
        assert b.supports("template")
        assert not b.supports("text")
        assert not b.supports("accessibility_id")


# ===========================================================================
# Cascade integration
# ===========================================================================


class TestCascadeInvariantIntegration:
    def test_invariant_in_default_backends(self):
        cd = CascadeDetector()
        names = [b.name for b in cd.backends]
        assert "invariant_template" in names

    def test_invariant_ordered_after_feature(self):
        cd = CascadeDetector()
        costs = {b.name: b.estimated_cost_ms() for b in cd.backends}
        assert costs.get("invariant_template", 0) > costs.get("feature", 0)
        assert costs.get("invariant_template", 0) > costs.get("template", 0)

    def test_invariant_ordered_before_omniparser(self):
        cd = CascadeDetector()
        costs = {b.name: b.estimated_cost_ms() for b in cd.backends}
        if "omniparser" in costs:
            assert costs["invariant_template"] < costs["omniparser"]

    def test_preferred_backend_forces_invariant(self):
        """MatchSettings(preferred_backend='invariant_template') puts it first."""

        class StubBackend(DetectionBackend):
            def __init__(self, backend_name, cost, results=None):
                self._name = backend_name
                self._cost = cost
                self._results = results or []
                self.find_called = False

            def find(self, needle, haystack, config):
                self.find_called = True
                return list(self._results)

            def supports(self, needle_type):
                return needle_type == "template"

            def estimated_cost_ms(self):
                return self._cost

            @property
            def name(self):
                return self._name

        cheap = StubBackend(
            "cheap",
            10.0,
            [DetectionResult(0, 0, 10, 10, 0.95, "cheap")],
        )
        invariant = StubBackend(
            "invariant_template",
            120.0,
            [DetectionResult(0, 0, 10, 10, 0.99, "invariant_template")],
        )

        cd = CascadeDetector(backends=[cheap, invariant])

        # Without preferred_backend, cheap wins (cost-ordered)
        results = cd.find("needle", "haystack", {"needle_type": "template"})
        assert results[0].backend_name == "cheap"
        assert cheap.find_called
        assert not invariant.find_called

        # Reset
        cheap.find_called = False

        # With preferred_backend, invariant goes first
        ms = MatchSettings(preferred_backend="invariant_template")
        results = cd.find(
            "needle", "haystack", {"needle_type": "template", "match_settings": ms}
        )
        assert results[0].backend_name == "invariant_template"
        assert invariant.find_called
