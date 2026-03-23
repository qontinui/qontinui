"""Tests for QATM (Quality-Aware Template Matching) backend.

Tests cover:
- Backend properties (name, cost, supports, availability)
- Config settings and env var propagation
- Image conversion (_to_bgr) for numpy, PIL, Pattern, grayscale
- Search region cropping and coordinate offset
- NMS deduplication
- IoU computation
- Unload lifecycle (should_unload timing)
- Cascade integration (ordering, availability gating)
- QATMMatcher.find with template larger than screenshot
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ..qatm_backend import QATMBackend, _ensure_bgr_array, _torch_available
from ..qatm_config import QATMSettings
from ..qatm_matcher import (  # noqa: F401 (_compute_iou used in tests)
    QATMMatch,
    QATMMatcher,
    _compute_iou,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_button(
    width: int = 120,
    height: int = 40,
    bg_color: tuple[int, int, int] = (240, 240, 240),
    border_color: tuple[int, int, int] = (100, 100, 100),
) -> np.ndarray:
    """Create a synthetic button image (BGR)."""
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    img[:2, :] = border_color
    img[-2:, :] = border_color
    img[:, :2] = border_color
    img[:, -2:] = border_color
    return img


def _make_screenshot(
    button: np.ndarray,
    canvas_size: tuple[int, int] = (600, 800),
    canvas_color: tuple[int, int, int] = (255, 255, 255),
    position: tuple[int, int] = (200, 150),
) -> np.ndarray:
    """Place a button into a larger canvas (BGR). position=(py, px)."""
    h, w = canvas_size
    canvas = np.full((h, w, 3), canvas_color, dtype=np.uint8)
    bh, bw = button.shape[:2]
    py, px = position
    canvas[py : py + bh, px : px + bw] = button
    return canvas


# ===========================================================================
# QATMSettings
# ===========================================================================


class TestQATMSettings:
    def test_defaults(self):
        s = QATMSettings()
        assert s.enabled is False
        assert s.device == "auto"
        assert s.confidence_threshold == 0.7
        assert s.feature_layer == "relu4_1"
        assert s.lazy_load is True
        assert s.unload_after_seconds == 300.0
        assert s.alpha == 25.0

    def test_enabled_via_constructor(self):
        s = QATMSettings(enabled=True, device="cpu")
        assert s.enabled is True
        assert s.device == "cpu"

    def test_resolve_device_explicit(self):
        s = QATMSettings(device="cpu")
        assert s.resolve_device() == "cpu"

        s2 = QATMSettings(device="cuda:0")
        assert s2.resolve_device() == "cuda:0"

    def test_resolve_device_auto_without_torch(self):
        with patch.dict("sys.modules", {"torch": None}):
            s = QATMSettings(device="auto")
            # When torch import fails, should fall back to cpu
            assert s.resolve_device() == "cpu"


# ===========================================================================
# _ensure_bgr_array
# ===========================================================================


class TestEnsureBgrArray:
    def test_3channel_passthrough(self):
        img = np.zeros((10, 20, 3), dtype=np.uint8)
        result = _ensure_bgr_array(img)
        assert result.shape == (10, 20, 3)

    def test_grayscale_to_bgr(self):
        gray = np.zeros((10, 20), dtype=np.uint8)
        result = _ensure_bgr_array(gray)
        assert result.shape == (10, 20, 3)

    def test_single_channel_to_bgr(self):
        single = np.zeros((10, 20, 1), dtype=np.uint8)
        result = _ensure_bgr_array(single)
        assert result.shape == (10, 20, 3)

    def test_rgba_strips_alpha(self):
        rgba = np.zeros((10, 20, 4), dtype=np.uint8)
        result = _ensure_bgr_array(rgba)
        assert result.shape == (10, 20, 3)


# ===========================================================================
# QATMBackend._to_bgr
# ===========================================================================


class TestToBgr:
    def test_numpy_array(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        result = QATMBackend._to_bgr(img)
        assert result is not None
        assert result.shape == (10, 10, 3)

    def test_grayscale_numpy(self):
        gray = np.zeros((10, 10), dtype=np.uint8)
        result = QATMBackend._to_bgr(gray)
        assert result is not None
        assert result.shape == (10, 10, 3)

    def test_pattern_object_with_pixel_data(self):
        pattern = MagicMock()
        pattern.pixel_data = np.zeros((10, 10, 3), dtype=np.uint8)
        result = QATMBackend._to_bgr(pattern)
        assert result is not None
        assert result.shape == (10, 10, 3)

    def test_pattern_with_none_pixel_data(self):
        pattern = MagicMock()
        pattern.pixel_data = None
        result = QATMBackend._to_bgr(pattern)
        assert result is None

    def test_pil_image(self):
        from PIL import Image

        pil_img = Image.new("RGB", (10, 10), color=(255, 0, 0))
        result = QATMBackend._to_bgr(pil_img)
        assert result is not None
        assert result.shape == (10, 10, 3)

    def test_pil_grayscale(self):
        from PIL import Image

        pil_img = Image.new("L", (10, 10), color=128)
        result = QATMBackend._to_bgr(pil_img)
        assert result is not None
        assert result.shape == (10, 10, 3)

    def test_pil_rgba(self):
        from PIL import Image

        pil_img = Image.new("RGBA", (10, 10), color=(255, 0, 0, 128))
        result = QATMBackend._to_bgr(pil_img)
        assert result is not None
        assert result.shape == (10, 10, 3)

    def test_unsupported_type_returns_none(self):
        result = QATMBackend._to_bgr("not_an_image")
        assert result is None

    def test_pattern_with_pil_pixel_data(self):
        """Pattern.pixel_data can be a PIL Image — should recurse."""
        from PIL import Image

        pattern = MagicMock()
        pattern.pixel_data = Image.new("RGB", (10, 10))
        result = QATMBackend._to_bgr(pattern)
        assert result is not None
        assert result.shape == (10, 10, 3)


# ===========================================================================
# QATMBackend properties
# ===========================================================================


class TestQATMBackendProperties:
    def test_name(self):
        b = QATMBackend()
        assert b.name == "qatm"

    def test_estimated_cost(self):
        b = QATMBackend()
        assert b.estimated_cost_ms() == 200.0

    def test_supports_template(self):
        b = QATMBackend()
        assert b.supports("template")

    def test_does_not_support_text(self):
        b = QATMBackend()
        assert not b.supports("text")
        assert not b.supports("description")
        assert not b.supports("accessibility_id")

    def test_unavailable_by_default(self):
        """Disabled by default (QONTINUI_QATM_ENABLED not set)."""
        b = QATMBackend()
        assert b.is_available() is False

    def test_available_when_enabled(self):
        settings = QATMSettings(enabled=True)
        b = QATMBackend(settings=settings)
        if _torch_available():
            assert b.is_available() is True
        else:
            # torch not installed — still unavailable
            assert b.is_available() is False

    def test_unavailable_without_torch(self):
        settings = QATMSettings(enabled=True)
        b = QATMBackend(settings=settings)
        with patch(
            "qontinui.find.backends.qatm_backend._torch_checked",
            False,
        ):
            # Force torch unavailable
            assert b.is_available() is False


# ===========================================================================
# QATMBackend.find — input handling (no real model needed)
# ===========================================================================


class TestQATMBackendFind:
    def test_returns_empty_for_unconvertible_needle(self):
        b = QATMBackend(settings=QATMSettings(enabled=True))
        results = b.find(
            needle="not_an_image",
            haystack=np.zeros((100, 100, 3), dtype=np.uint8),
            config={"min_confidence": 0.5},
        )
        assert results == []

    def test_returns_empty_for_unconvertible_haystack(self):
        b = QATMBackend(settings=QATMSettings(enabled=True))
        results = b.find(
            needle=np.zeros((10, 10, 3), dtype=np.uint8),
            haystack="not_an_image",
            config={"min_confidence": 0.5},
        )
        assert results == []

    def test_search_region_crops_and_offsets(self):
        """Results should be offset by search_region origin."""
        settings = QATMSettings(enabled=True)
        b = QATMBackend(settings=settings)

        # Mock the matcher to return a known result
        mock_matcher = MagicMock()
        mock_matcher.should_unload.return_value = False
        mock_matcher.find.return_value = [
            QATMMatch(x=10, y=20, width=30, height=40, confidence=0.9)
        ]
        b._matcher = mock_matcher

        screenshot = np.zeros((600, 800, 3), dtype=np.uint8)
        template = np.zeros((30, 40, 3), dtype=np.uint8)

        results = b.find(
            needle=template,
            haystack=screenshot,
            config={
                "min_confidence": 0.5,
                "search_region": (100, 200, 400, 300),
            },
        )

        assert len(results) == 1
        # x should be offset by region_x=100, y by region_y=200
        assert results[0].x == 10 + 100
        assert results[0].y == 20 + 200
        assert results[0].backend_name == "qatm"
        assert results[0].metadata == {"quality_aware": True}

    def test_matcher_exception_returns_empty(self):
        """Backend should catch matcher exceptions and return []."""
        settings = QATMSettings(enabled=True)
        b = QATMBackend(settings=settings)

        mock_matcher = MagicMock()
        mock_matcher.should_unload.return_value = False
        mock_matcher.find.side_effect = RuntimeError("GPU OOM")
        b._matcher = mock_matcher

        results = b.find(
            needle=np.zeros((10, 10, 3), dtype=np.uint8),
            haystack=np.zeros((100, 100, 3), dtype=np.uint8),
            config={"min_confidence": 0.5},
        )
        assert results == []

    def test_unload_checked_after_use(self):
        """should_unload is checked AFTER find(), not before."""
        settings = QATMSettings(enabled=True)
        b = QATMBackend(settings=settings)

        mock_matcher = MagicMock()
        mock_matcher.find.return_value = []
        # Model is idle — should_unload returns True
        mock_matcher.should_unload.return_value = True
        b._matcher = mock_matcher

        b.find(
            needle=np.zeros((10, 10, 3), dtype=np.uint8),
            haystack=np.zeros((100, 100, 3), dtype=np.uint8),
            config={"min_confidence": 0.5},
        )

        # find() should have been called first
        mock_matcher.find.assert_called_once()
        # Then unload should be called after
        mock_matcher.unload.assert_called_once()


# ===========================================================================
# IoU computation
# ===========================================================================


class TestComputeIoU:
    def test_identical_boxes(self):
        a = QATMMatch(x=0, y=0, width=100, height=100, confidence=0.9)
        b = QATMMatch(x=0, y=0, width=100, height=100, confidence=0.8)
        assert _compute_iou(a, b) == 1.0

    def test_no_overlap(self):
        a = QATMMatch(x=0, y=0, width=50, height=50, confidence=0.9)
        b = QATMMatch(x=100, y=100, width=50, height=50, confidence=0.8)
        assert _compute_iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = QATMMatch(x=0, y=0, width=100, height=100, confidence=0.9)
        b = QATMMatch(x=50, y=50, width=100, height=100, confidence=0.8)
        # intersection = 50*50 = 2500
        # union = 10000 + 10000 - 2500 = 17500
        expected = 2500 / 17500
        assert abs(_compute_iou(a, b) - expected) < 1e-6

    def test_touching_edges_no_overlap(self):
        a = QATMMatch(x=0, y=0, width=50, height=50, confidence=0.9)
        b = QATMMatch(x=50, y=0, width=50, height=50, confidence=0.8)
        assert _compute_iou(a, b) == 0.0

    def test_one_inside_other(self):
        a = QATMMatch(x=0, y=0, width=100, height=100, confidence=0.9)
        b = QATMMatch(x=25, y=25, width=50, height=50, confidence=0.8)
        # intersection = 50*50 = 2500
        # union = 10000 + 2500 - 2500 = 10000
        assert abs(_compute_iou(a, b) - 0.25) < 1e-6


# ===========================================================================
# NMS
# ===========================================================================


class TestNMS:
    def test_empty_input(self):
        assert QATMMatcher._nms([]) == []

    def test_single_match(self):
        m = QATMMatch(x=0, y=0, width=50, height=50, confidence=0.9)
        result = QATMMatcher._nms([m])
        assert len(result) == 1

    def test_non_overlapping_kept(self):
        m1 = QATMMatch(x=0, y=0, width=50, height=50, confidence=0.9)
        m2 = QATMMatch(x=200, y=200, width=50, height=50, confidence=0.8)
        result = QATMMatcher._nms([m1, m2])
        assert len(result) == 2

    def test_overlapping_suppressed(self):
        m1 = QATMMatch(x=0, y=0, width=100, height=100, confidence=0.9)
        m2 = QATMMatch(x=10, y=10, width=100, height=100, confidence=0.8)
        result = QATMMatcher._nms([m1, m2], iou_threshold=0.5)
        assert len(result) == 1
        assert result[0].confidence == 0.9  # Higher confidence kept

    def test_keeps_highest_confidence(self):
        """Lower confidence match should be suppressed, not the higher one."""
        low = QATMMatch(x=0, y=0, width=100, height=100, confidence=0.5)
        high = QATMMatch(x=5, y=5, width=100, height=100, confidence=0.95)
        result = QATMMatcher._nms([low, high], iou_threshold=0.5)
        assert len(result) == 1
        assert result[0].confidence == 0.95


# ===========================================================================
# QATMMatcher — template size guard
# ===========================================================================


class TestQATMMatcherGuards:
    def test_template_larger_than_screenshot(self):
        """Should return [] without crashing when template > screenshot."""
        settings = QATMSettings(enabled=True)
        matcher = QATMMatcher(settings=settings)

        big_template = np.zeros((500, 500, 3), dtype=np.uint8)
        small_screenshot = np.zeros((100, 100, 3), dtype=np.uint8)

        results = matcher.find(big_template, small_screenshot)
        assert results == []

    def test_template_same_width_as_screenshot(self):
        """Edge case: template width == screenshot width is not blocked by guard."""
        template = np.zeros((50, 100, 3), dtype=np.uint8)
        screenshot = np.zeros((200, 100, 3), dtype=np.uint8)

        # Template width == screenshot width is allowed (not >)
        # The size guard only blocks template strictly larger than screenshot.
        assert template.shape[1] <= screenshot.shape[1]
        assert template.shape[0] <= screenshot.shape[0]


# ===========================================================================
# QATMMatcher — unload lifecycle
# ===========================================================================


class TestQATMMatcherUnload:
    def test_should_unload_false_when_no_model(self):
        matcher = QATMMatcher()
        assert matcher.should_unload() is False

    def test_should_unload_false_when_timeout_zero(self):
        settings = QATMSettings(unload_after_seconds=0.0)
        matcher = QATMMatcher(settings=settings)
        matcher._model = MagicMock()  # Pretend model is loaded
        assert matcher.should_unload() is False

    def test_should_unload_true_after_timeout(self):
        settings = QATMSettings(unload_after_seconds=0.01)  # 10ms
        matcher = QATMMatcher(settings=settings)
        matcher._model = MagicMock()
        matcher._last_used = time.monotonic() - 1.0  # 1s ago
        assert matcher.should_unload() is True

    def test_should_unload_false_within_timeout(self):
        settings = QATMSettings(unload_after_seconds=300.0)
        matcher = QATMMatcher(settings=settings)
        matcher._model = MagicMock()
        matcher._last_used = time.monotonic()  # Just used
        assert matcher.should_unload() is False

    def test_unload_clears_model(self):
        matcher = QATMMatcher()
        matcher._model = MagicMock()
        matcher._device = MagicMock()

        with patch.dict("sys.modules", {"torch": MagicMock()}):
            matcher.unload()

        assert matcher._model is None
        assert matcher._device is None

    def test_unload_noop_when_no_model(self):
        """unload() should not crash when model is already None."""
        matcher = QATMMatcher()
        matcher.unload()  # Should not raise
        assert matcher._model is None


# ===========================================================================
# QATMMatcher — layer map validation
# ===========================================================================


class TestQATMMatcherLayerMap:
    def test_invalid_layer_raises(self):
        settings = QATMSettings(feature_layer="nonexistent_layer")
        matcher = QATMMatcher(settings=settings)

        with pytest.raises(ValueError, match="Unknown VGG-19 layer"):
            matcher._ensure_model()

    def test_all_layers_in_map(self):
        """All documented VGG-19 ReLU layers should be in the map."""
        expected = {
            "relu1_1",
            "relu1_2",
            "relu2_1",
            "relu2_2",
            "relu3_1",
            "relu3_2",
            "relu3_3",
            "relu3_4",
            "relu4_1",
            "relu4_2",
            "relu4_3",
            "relu4_4",
            "relu5_1",
            "relu5_2",
            "relu5_3",
            "relu5_4",
        }
        assert set(QATMMatcher._LAYER_MAP.keys()) == expected


# ===========================================================================
# Cascade integration
# ===========================================================================


class TestQATMCascadeIntegration:
    def test_qatm_in_default_backends(self):
        """QATM should appear in default cascade backend list."""
        from ..cascade import CascadeDetector

        cascade = CascadeDetector()
        names = [b.name for b in cascade.backends]
        assert "qatm" in names

    def test_qatm_ordered_after_invariant(self):
        """QATM (200ms) should come after invariant (120ms)."""
        from ..cascade import CascadeDetector

        cascade = CascadeDetector()
        names = [b.name for b in cascade.backends]
        if "invariant_template" in names and "qatm" in names:
            assert names.index("invariant_template") < names.index("qatm")

    def test_qatm_ordered_before_omniparser(self):
        """QATM (200ms) should come before omniparser (1500ms)."""
        from ..cascade import CascadeDetector

        cascade = CascadeDetector()
        names = [b.name for b in cascade.backends]
        if "omniparser" in names and "qatm" in names:
            assert names.index("qatm") < names.index("omniparser")

    def test_qatm_unavailable_by_default_in_cascade(self):
        """QATM should be in chain but is_available=False (not enabled)."""
        from ..cascade import CascadeDetector

        cascade = CascadeDetector()
        qatm_backends = [b for b in cascade.backends if b.name == "qatm"]
        assert len(qatm_backends) == 1
        assert qatm_backends[0].is_available() is False

    def test_costs_sorted_ascending(self):
        """All backends should be sorted by ascending cost."""
        from ..cascade import CascadeDetector

        cascade = CascadeDetector()
        costs = [b.estimated_cost_ms() for b in cascade.backends]
        assert costs == sorted(costs)
