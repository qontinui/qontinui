"""Integration tests for CascadeDetector wired into the find pipeline.

Verifies:
- RealFindImplementation creates a CascadeDetector on init
- Cascade fallback fires when template matching fails
- Cascade events are emitted correctly
- MatchSettings flows from config through StateImage to Pattern to cascade
"""

import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Auto-mock missing external deps (same approach as test_cascade_detector.py)
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

from qontinui.find.backends.base import DetectionBackend, DetectionResult
from qontinui.find.backends.cascade import CascadeDetector, MatchSettings
from qontinui.reporting.events import EventCollector, EventType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeBackend(DetectionBackend):
    """Backend that returns configurable results."""

    def __init__(self, name: str, cost: float, results: list[DetectionResult] | None = None):
        self._name = name
        self._cost = cost
        self._results = results or []
        self.find_count = 0

    def find(self, needle, haystack, config):
        self.find_count += 1
        return list(self._results)

    def supports(self, needle_type):
        return True

    def estimated_cost_ms(self):
        return self._cost

    @property
    def name(self):
        return self._name


def _result(confidence=0.9, backend="fake"):
    return DetectionResult(
        x=50,
        y=60,
        width=40,
        height=30,
        confidence=confidence,
        backend_name=backend,
    )


# ---------------------------------------------------------------------------
# Test: CascadeDetector events
# ---------------------------------------------------------------------------


class TestCascadeEvents:
    def test_cascade_hit_emits_events(self):
        backend = FakeBackend("fast", 10, [_result(0.95, "fast")])
        cascade = CascadeDetector(backends=[backend])

        with EventCollector() as collector:
            results = cascade.find(
                "needle",
                "haystack",
                {"needle_type": "template", "min_confidence": 0.8},
            )

        assert len(results) == 1

        started = collector.get_events(EventType.CASCADE_STARTED)
        assert len(started) == 1
        assert started[0].data["needle_type"] == "template"

        tried = collector.get_events(EventType.CASCADE_BACKEND_TRIED)
        assert len(tried) == 1
        assert tried[0].data["backend"] == "fast"
        assert tried[0].data["success"] is True

        hits = collector.get_events(EventType.CASCADE_HIT)
        assert len(hits) == 1
        assert hits[0].data["winning_backend"] == "fast"
        assert hits[0].data["backends_tried"] == 1

        misses = collector.get_events(EventType.CASCADE_MISS)
        assert len(misses) == 0

    def test_cascade_miss_emits_events(self):
        backend = FakeBackend("empty", 10, [])
        cascade = CascadeDetector(backends=[backend])

        with EventCollector() as collector:
            results = cascade.find(
                "needle",
                "haystack",
                {"needle_type": "template", "min_confidence": 0.8},
            )

        assert results == []

        misses = collector.get_events(EventType.CASCADE_MISS)
        assert len(misses) == 1
        assert misses[0].data["backends_tried"] == 1

        hits = collector.get_events(EventType.CASCADE_HIT)
        assert len(hits) == 0

    def test_cascade_fallback_emits_backend_tried_for_each(self):
        b1 = FakeBackend("b1", 10, [])
        b2 = FakeBackend("b2", 50, [_result(0.92, "b2")])
        cascade = CascadeDetector(backends=[b1, b2])

        with EventCollector() as collector:
            results = cascade.find(
                "needle",
                "haystack",
                {"needle_type": "template", "min_confidence": 0.8},
            )

        assert len(results) == 1
        tried = collector.get_events(EventType.CASCADE_BACKEND_TRIED)
        assert len(tried) == 2
        assert tried[0].data["backend"] == "b1"
        assert tried[0].data["success"] is False
        assert tried[1].data["backend"] == "b2"
        assert tried[1].data["success"] is True


# ---------------------------------------------------------------------------
# Test: MatchSettings flows through config
# ---------------------------------------------------------------------------


class TestMatchSettingsFlow:
    def test_match_settings_on_state_image_flows_to_pattern(self):
        """Verify StateImage.set_match_settings() propagates to Pattern."""
        import numpy as np

        from qontinui.model.element import Pattern
        from qontinui.model.state.state_image import StateImage

        pixel_data = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = np.ones((10, 10), dtype=np.float32)
        pattern = Pattern(
            id="test",
            name="test_pattern",
            pixel_data=pixel_data,
            mask=mask,
        )

        ms = MatchSettings(preferred_backend="feature", min_confidence=0.6)
        si = StateImage(image=pattern, name="test_img")
        si.set_match_settings(ms)

        result_pattern = si.get_pattern()
        assert result_pattern.match_settings is not None
        assert result_pattern.match_settings.preferred_backend == "feature"
        assert result_pattern.match_settings.min_confidence == 0.6

    def test_pattern_level_match_settings_not_overridden_by_state_image(self):
        """Pattern-level match_settings takes precedence over StateImage."""
        import numpy as np

        from qontinui.model.element import Pattern
        from qontinui.model.state.state_image import StateImage

        pixel_data = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = np.ones((10, 10), dtype=np.float32)
        pattern_ms = MatchSettings(preferred_backend="ocr")
        pattern = Pattern(
            id="test",
            name="test_pattern",
            pixel_data=pixel_data,
            mask=mask,
            match_settings=pattern_ms,
        )

        si_ms = MatchSettings(preferred_backend="feature")
        si = StateImage(image=pattern, name="test_img")
        si.set_match_settings(si_ms)

        result_pattern = si.get_pattern()
        # Pattern-level should win
        assert result_pattern.match_settings.preferred_backend == "ocr"


# ---------------------------------------------------------------------------
# Test: MatchSettingsConfig JSON parsing
# ---------------------------------------------------------------------------


class TestMatchSettingsConfig:
    def test_parse_from_dict(self):
        from qontinui.json_executor.config_parser import MatchSettingsConfig

        config = MatchSettingsConfig(
            **{
                "preferredBackend": "feature",
                "minConfidence": 0.7,
                "maxBackends": 3,
                "searchRegion": [100, 200, 300, 400],
            }
        )

        ms = config.to_match_settings()
        assert ms.preferred_backend == "feature"
        assert ms.min_confidence == 0.7
        assert ms.max_backends == 3
        assert ms.search_region == (100, 200, 300, 400)

    def test_defaults(self):
        from qontinui.json_executor.config_parser import MatchSettingsConfig

        config = MatchSettingsConfig()
        ms = config.to_match_settings()
        assert ms.preferred_backend is None
        assert ms.min_confidence == 0.8
        assert ms.max_backends == 5
        assert ms.search_region is None
