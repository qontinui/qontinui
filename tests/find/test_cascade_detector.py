"""Tests for the CascadeDetector and detection backends.

Verifies:
- Graduated fallback: first backend fails → second is tried
- Short-circuit: first backend succeeds → second is NOT called
- supports() filtering: backends that don't support needle_type are skipped
- Cost-based ordering: backends are called cheapest-first
- MatchSettings per-target overrides (preferred_backend, min_confidence, max_backends)
- Unavailable backends are skipped
"""

import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Auto-mock any missing external dependency so the heavy qontinui __init__
# import chain doesn't break.  We only need find.backends.{base,cascade}.
# ---------------------------------------------------------------------------


class _MockFinder:
    """Meta-path finder that intercepts imports of missing packages and
    provides an auto-attribute MagicMock module for them."""

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
        mod.__path__ = []  # make it look like a package
        mod.__name__ = fullname
        sys.modules[fullname] = mod
        return mod


sys.meta_path.insert(0, _MockFinder())

from qontinui.find.backends.base import DetectionBackend, DetectionResult  # noqa: E402
from qontinui.find.backends.cascade import CascadeDetector, MatchSettings  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class StubBackend(DetectionBackend):
    """Configurable stub backend for testing."""

    def __init__(
        self,
        backend_name: str,
        cost: float,
        supported_types: list[str],
        results: list[DetectionResult] | None = None,
        available: bool = True,
        raises: Exception | None = None,
    ):
        self._name = backend_name
        self._cost = cost
        self._supported = supported_types
        self._results = results or []
        self._available = available
        self._raises = raises
        self.find_called = False
        self.find_call_count = 0

    def find(self, needle, haystack, config):
        self.find_called = True
        self.find_call_count += 1
        if self._raises:
            raise self._raises
        return list(self._results)

    def supports(self, needle_type):
        return needle_type in self._supported

    def estimated_cost_ms(self):
        return self._cost

    @property
    def name(self):
        return self._name

    def is_available(self):
        return self._available


def _make_result(confidence: float = 0.9, backend: str = "test") -> DetectionResult:
    return DetectionResult(
        x=100,
        y=200,
        width=50,
        height=30,
        confidence=confidence,
        backend_name=backend,
    )


# ---------------------------------------------------------------------------
# Test: cost-based ordering
# ---------------------------------------------------------------------------


class TestCostBasedOrdering:
    def test_backends_sorted_by_cost(self):
        expensive = StubBackend("expensive", cost=500, supported_types=["template"])
        cheap = StubBackend("cheap", cost=10, supported_types=["template"])
        mid = StubBackend("mid", cost=100, supported_types=["template"])

        cascade = CascadeDetector(backends=[expensive, cheap, mid])
        names = [b.name for b in cascade.backends]

        assert names == ["cheap", "mid", "expensive"]

    def test_cheapest_tried_first(self):
        cheap = StubBackend(
            "cheap",
            cost=10,
            supported_types=["template"],
            results=[_make_result(0.95, "cheap")],
        )
        expensive = StubBackend(
            "expensive",
            cost=500,
            supported_types=["template"],
            results=[_make_result(0.99, "expensive")],
        )

        cascade = CascadeDetector(backends=[expensive, cheap])
        results = cascade.find(None, None, {"needle_type": "template"})

        assert len(results) == 1
        assert results[0].backend_name == "cheap"
        assert cheap.find_called
        assert not expensive.find_called


# ---------------------------------------------------------------------------
# Test: graduated fallback
# ---------------------------------------------------------------------------


class TestGraduatedFallback:
    def test_fallback_on_empty_results(self):
        first = StubBackend("first", cost=10, supported_types=["template"], results=[])
        second = StubBackend(
            "second",
            cost=100,
            supported_types=["template"],
            results=[_make_result(0.9, "second")],
        )

        cascade = CascadeDetector(backends=[first, second])
        results = cascade.find(None, None, {"needle_type": "template"})

        assert first.find_called
        assert second.find_called
        assert len(results) == 1
        assert results[0].backend_name == "second"

    def test_fallback_on_low_confidence(self):
        first = StubBackend(
            "first",
            cost=10,
            supported_types=["template"],
            results=[_make_result(0.3, "first")],  # Below default 0.8
        )
        second = StubBackend(
            "second",
            cost=100,
            supported_types=["template"],
            results=[_make_result(0.9, "second")],
        )

        cascade = CascadeDetector(backends=[first, second])
        results = cascade.find(None, None, {"needle_type": "template"})

        assert first.find_called
        assert second.find_called
        assert results[0].backend_name == "second"

    def test_fallback_on_exception(self):
        first = StubBackend(
            "first",
            cost=10,
            supported_types=["template"],
            raises=RuntimeError("boom"),
        )
        second = StubBackend(
            "second",
            cost=100,
            supported_types=["template"],
            results=[_make_result(0.9, "second")],
        )

        cascade = CascadeDetector(backends=[first, second])
        results = cascade.find(None, None, {"needle_type": "template"})

        assert first.find_called
        assert second.find_called
        assert len(results) == 1

    def test_all_backends_fail_returns_empty(self):
        b1 = StubBackend("b1", cost=10, supported_types=["template"], results=[])
        b2 = StubBackend("b2", cost=100, supported_types=["template"], results=[])

        cascade = CascadeDetector(backends=[b1, b2])
        results = cascade.find(None, None, {"needle_type": "template"})

        assert results == []
        assert b1.find_called
        assert b2.find_called


# ---------------------------------------------------------------------------
# Test: short-circuit
# ---------------------------------------------------------------------------


class TestShortCircuit:
    def test_stops_on_first_success(self):
        first = StubBackend(
            "first",
            cost=10,
            supported_types=["template"],
            results=[_make_result(0.95, "first")],
        )
        second = StubBackend(
            "second",
            cost=100,
            supported_types=["template"],
            results=[_make_result(0.99, "second")],
        )

        cascade = CascadeDetector(backends=[first, second])
        results = cascade.find(None, None, {"needle_type": "template"})

        assert first.find_called
        assert not second.find_called
        assert len(results) == 1
        assert results[0].backend_name == "first"


# ---------------------------------------------------------------------------
# Test: supports() filtering
# ---------------------------------------------------------------------------


class TestSupportsFiltering:
    def test_skips_unsupported_backends(self):
        template_only = StubBackend(
            "template_only",
            cost=10,
            supported_types=["template"],
            results=[_make_result(0.9, "template_only")],
        )
        text_only = StubBackend(
            "text_only",
            cost=5,
            supported_types=["text"],
            results=[_make_result(0.99, "text_only")],
        )

        cascade = CascadeDetector(backends=[template_only, text_only])
        results = cascade.find(None, None, {"needle_type": "template"})

        assert template_only.find_called
        assert not text_only.find_called
        assert results[0].backend_name == "template_only"

    def test_cascade_supports_reflects_backends(self):
        b1 = StubBackend("b1", cost=10, supported_types=["template"])
        b2 = StubBackend("b2", cost=100, supported_types=["text"])

        cascade = CascadeDetector(backends=[b1, b2])
        assert cascade.supports("template")
        assert cascade.supports("text")
        assert not cascade.supports("unknown")


# ---------------------------------------------------------------------------
# Test: unavailable backends skipped
# ---------------------------------------------------------------------------


class TestUnavailableBackends:
    def test_skips_unavailable_backend(self):
        unavailable = StubBackend(
            "unavailable",
            cost=5,
            supported_types=["template"],
            results=[_make_result(0.99)],
            available=False,
        )
        available = StubBackend(
            "available",
            cost=100,
            supported_types=["template"],
            results=[_make_result(0.9, "available")],
        )

        cascade = CascadeDetector(backends=[unavailable, available])
        results = cascade.find(None, None, {"needle_type": "template"})

        assert not unavailable.find_called
        assert available.find_called
        assert results[0].backend_name == "available"


# ---------------------------------------------------------------------------
# Test: MatchSettings overrides
# ---------------------------------------------------------------------------


class TestMatchSettings:
    def test_preferred_backend_tried_first(self):
        cheap = StubBackend(
            "cheap",
            cost=10,
            supported_types=["template"],
            results=[_make_result(0.85, "cheap")],
        )
        preferred = StubBackend(
            "preferred",
            cost=500,
            supported_types=["template"],
            results=[_make_result(0.95, "preferred")],
        )

        cascade = CascadeDetector(backends=[cheap, preferred])
        settings = MatchSettings(preferred_backend="preferred")
        results = cascade.find(
            None,
            None,
            {"needle_type": "template", "match_settings": settings},
        )

        assert preferred.find_called
        # cheap should not be called because preferred succeeded
        assert not cheap.find_called
        assert results[0].backend_name == "preferred"

    def test_max_backends_limits_attempts(self):
        b1 = StubBackend("b1", cost=10, supported_types=["template"], results=[])
        b2 = StubBackend("b2", cost=20, supported_types=["template"], results=[])
        b3 = StubBackend(
            "b3",
            cost=30,
            supported_types=["template"],
            results=[_make_result(0.9, "b3")],
        )

        cascade = CascadeDetector(backends=[b1, b2, b3])
        settings = MatchSettings(max_backends=2)
        results = cascade.find(
            None,
            None,
            {"needle_type": "template", "match_settings": settings},
        )

        assert b1.find_called
        assert b2.find_called
        assert not b3.find_called
        assert results == []

    def test_min_confidence_override(self):
        backend = StubBackend(
            "backend",
            cost=10,
            supported_types=["template"],
            results=[_make_result(0.7, "backend")],
        )

        cascade = CascadeDetector(backends=[backend])

        # Default min_confidence=0.8 → should filter out 0.7
        results = cascade.find(None, None, {"needle_type": "template"})
        assert results == []

        # Lower threshold via MatchSettings
        backend.find_call_count = 0
        settings = MatchSettings(min_confidence=0.6)
        results = cascade.find(
            None,
            None,
            {"needle_type": "template", "match_settings": settings},
        )
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Test: add/remove backends
# ---------------------------------------------------------------------------


class TestBackendManagement:
    def test_add_backend_maintains_sort(self):
        cascade = CascadeDetector(backends=[])
        cascade.add_backend(StubBackend("b", cost=100, supported_types=[]))
        cascade.add_backend(StubBackend("a", cost=5, supported_types=[]))
        cascade.add_backend(StubBackend("c", cost=50, supported_types=[]))

        costs = [b.estimated_cost_ms() for b in cascade.backends]
        assert costs == [5, 50, 100]

    def test_remove_backend(self):
        b1 = StubBackend("b1", cost=10, supported_types=[])
        b2 = StubBackend("b2", cost=20, supported_types=[])

        cascade = CascadeDetector(backends=[b1, b2])
        cascade.remove_backend("b1")

        assert len(cascade.backends) == 1
        assert cascade.backends[0].name == "b2"


# ---------------------------------------------------------------------------
# Test: DetectionResult properties
# ---------------------------------------------------------------------------


class TestDetectionResult:
    def test_center(self):
        r = DetectionResult(
            x=100, y=200, width=50, height=30, confidence=0.9, backend_name="test"
        )
        assert r.center == (125, 215)

    def test_bounds(self):
        r = DetectionResult(
            x=10, y=20, width=30, height=40, confidence=0.5, backend_name="test"
        )
        assert r.bounds == (10, 20, 30, 40)

    def test_normalize(self):
        r = DetectionResult(
            x=100, y=200, width=50, height=30, confidence=0.9, backend_name="test"
        )
        rn = r.normalize(1920, 1080)
        assert rn.normalized_x == pytest.approx(100 / 1920)
        assert rn.normalized_y == pytest.approx(200 / 1080)
        assert rn.normalized_width == pytest.approx(50 / 1920)
        assert rn.normalized_height == pytest.approx(30 / 1080)
        # Original is unchanged
        assert r.normalized_x is None

    def test_normalize_zero_dimensions(self):
        r = DetectionResult(
            x=100, y=200, width=50, height=30, confidence=0.9, backend_name="test"
        )
        rn = r.normalize(0, 0)
        assert rn.normalized_x is None

    def test_normalized_bounds_none_before_normalize(self):
        r = DetectionResult(
            x=10, y=20, width=30, height=40, confidence=0.5, backend_name="test"
        )
        assert r.normalized_bounds is None

    def test_normalized_bounds_after_normalize(self):
        r = DetectionResult(
            x=100, y=200, width=50, height=30, confidence=0.9, backend_name="test"
        )
        rn = r.normalize(1000, 1000)
        nb = rn.normalized_bounds
        assert nb is not None
        assert nb == pytest.approx((0.1, 0.2, 0.05, 0.03))
