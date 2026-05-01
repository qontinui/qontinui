"""Tests for PolygonZone, LineZone, and ZoneConditionEvaluator.

Verifies:
- PolygonZone membership testing with various anchor positions
- PolygonZone properties (area, bounding_rect, current_count)
- LineZone crossing detection with tracker_id
- ZoneCondition evaluation with comparison operators
- ZoneConditionEvaluator registry and batch evaluation
"""

import sys
from unittest.mock import MagicMock


class _MockFinder:
    """Auto-mock missing external packages."""

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
import pytest

from qontinui.find.detections import Detections
from qontinui.find.line_zone import LineZone, Point
from qontinui.find.zones import PolygonZone, Position
from qontinui.state_management.zone_condition import ZoneCondition, ZoneConditionEvaluator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _square_zone(x: int = 0, y: int = 0, size: int = 200) -> PolygonZone:
    """Create a square zone."""
    return PolygonZone(
        polygon=np.array(
            [
                [x, y],
                [x + size, y],
                [x + size, y + size],
                [x, y + size],
            ]
        ),
    )


def _make_dets_at(centers: list[tuple[int, int]], box_size: int = 20) -> Detections:
    """Create detections centered at given points."""
    n = len(centers)
    xyxy = np.empty((n, 4), dtype=np.int_)
    for i, (cx, cy) in enumerate(centers):
        xyxy[i] = [
            cx - box_size // 2,
            cy - box_size // 2,
            cx + box_size // 2,
            cy + box_size // 2,
        ]
    return Detections(
        xyxy=xyxy,
        confidence=np.ones(n, dtype=np.float64),
        backend_name=np.array(["test"] * n, dtype=object),
    )


def _make_tracked_dets(
    centers: list[tuple[int, int]], tracker_ids: list[int], box_size: int = 20
) -> Detections:
    """Create detections with tracker_id in data dict."""
    dets = _make_dets_at(centers, box_size)
    dets.data["tracker_id"] = tracker_ids
    return dets


# ===========================================================================
# PolygonZone Tests
# ===========================================================================


class TestPolygonZone:
    def test_detections_inside(self):
        zone = _square_zone(0, 0, 200)
        dets = _make_dets_at([(100, 100), (50, 50)])
        mask = zone.trigger(dets)
        assert mask.all()
        assert zone.current_count == 2

    def test_detections_outside(self):
        zone = _square_zone(0, 0, 100)
        dets = _make_dets_at([(500, 500), (300, 300)])
        mask = zone.trigger(dets)
        assert not mask.any()
        assert zone.current_count == 0

    def test_mixed_inside_outside(self):
        zone = _square_zone(0, 0, 200)
        dets = _make_dets_at([(100, 100), (500, 500), (50, 50)])
        mask = zone.trigger(dets)
        assert mask[0] and not mask[1] and mask[2]
        assert zone.current_count == 2

    def test_empty_detections(self):
        zone = _square_zone()
        mask = zone.trigger(Detections.empty())
        assert len(mask) == 0
        assert zone.current_count == 0

    def test_triggering_position_center(self):
        zone = _square_zone(0, 0, 100)
        # Box center at (50, 50) -> inside
        dets = _make_dets_at([(50, 50)])
        zone_c = PolygonZone(
            polygon=np.array([[0, 0], [100, 0], [100, 100], [0, 100]]),
            triggering_position=Position.CENTER,
        )
        mask = zone_c.trigger(dets)
        assert mask[0]

    def test_triggering_position_bottom_center(self):
        zone = PolygonZone(
            polygon=np.array([[0, 90], [100, 90], [100, 200], [0, 200]]),
            triggering_position=Position.BOTTOM_CENTER,
        )
        # Box from (40, 80) to (60, 100) -> bottom_center at (50, 100) -> inside zone [90..200]
        dets = Detections(
            xyxy=np.array([[40, 80, 60, 100]], dtype=np.int_),
            confidence=np.array([1.0]),
            backend_name=np.array(["test"], dtype=object),
        )
        mask = zone.trigger(dets)
        assert mask[0]

    def test_contains_point(self):
        zone = _square_zone(0, 0, 100)
        assert zone.contains_point(50, 50)
        assert not zone.contains_point(200, 200)
        assert zone.contains_point(0, 0)  # on edge

    def test_area(self):
        zone = _square_zone(0, 0, 100)
        assert zone.area == pytest.approx(10000.0)

    def test_bounding_rect(self):
        zone = _square_zone(10, 20, 100)
        x, y, w, h = zone.bounding_rect
        # OpenCV boundingRect returns inclusive dimensions (+1)
        assert (x, y) == (10, 20)
        assert w >= 100 and h >= 100

    def test_validation_too_few_vertices(self):
        with pytest.raises(ValueError, match="at least 3"):
            PolygonZone(polygon=np.array([[0, 0], [100, 100]]))

    def test_validation_wrong_shape(self):
        with pytest.raises(ValueError, match="shape"):
            PolygonZone(polygon=np.array([0, 1, 2, 3]))

    def test_repr(self):
        zone = _square_zone()
        r = repr(zone)
        assert "vertices=4" in r
        assert "current_count=" in r

    def test_all_positions(self):
        """Every Position enum value should work without error."""
        zone = _square_zone(0, 0, 1000)
        dets = _make_dets_at([(100, 100)])
        for pos in Position:
            z = PolygonZone(
                polygon=np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]]),
                triggering_position=pos,
            )
            mask = z.trigger(dets)
            assert len(mask) == 1


# ===========================================================================
# LineZone Tests
# ===========================================================================


class TestLineZone:
    def test_no_crossing_first_frame(self):
        line = LineZone(start=Point(0, 100), end=Point(400, 100))
        dets = _make_tracked_dets([(50, 50)], [1])
        in_mask, out_mask = line.trigger(dets)
        assert not in_mask.any()
        assert not out_mask.any()

    def test_crossing_in(self):
        line = LineZone(start=Point(0, 100), end=Point(400, 100))
        # Frame 1: above the line (y=50)
        dets1 = _make_tracked_dets([(200, 50)], [1])
        line.trigger(dets1)

        # Frame 2: below the line (y=150) → crossing IN
        dets2 = _make_tracked_dets([(200, 150)], [1])
        in_mask, out_mask = line.trigger(dets2)
        assert in_mask[0]
        assert not out_mask[0]
        assert line.in_count == 1

    def test_crossing_out(self):
        line = LineZone(start=Point(0, 100), end=Point(400, 100))
        # Frame 1: below the line
        dets1 = _make_tracked_dets([(200, 150)], [1])
        line.trigger(dets1)

        # Frame 2: above the line → crossing OUT
        dets2 = _make_tracked_dets([(200, 50)], [1])
        in_mask, out_mask = line.trigger(dets2)
        assert not in_mask[0]
        assert out_mask[0]
        assert line.out_count == 1

    def test_no_crossing_same_side(self):
        line = LineZone(start=Point(0, 100), end=Point(400, 100))
        dets1 = _make_tracked_dets([(200, 50)], [1])
        line.trigger(dets1)

        dets2 = _make_tracked_dets([(200, 60)], [1])
        in_mask, out_mask = line.trigger(dets2)
        assert not in_mask.any()
        assert not out_mask.any()

    def test_multiple_trackers(self):
        line = LineZone(start=Point(0, 100), end=Point(400, 100))
        # Frame 1
        dets1 = _make_tracked_dets([(100, 50), (300, 150)], [1, 2])
        line.trigger(dets1)

        # Frame 2: both cross
        dets2 = _make_tracked_dets([(100, 150), (300, 50)], [1, 2])
        in_mask, out_mask = line.trigger(dets2)
        assert in_mask[0] and out_mask[1]
        assert line.in_count == 1
        assert line.out_count == 1

    def test_reset(self):
        line = LineZone(start=Point(0, 100), end=Point(400, 100))
        dets1 = _make_tracked_dets([(200, 50)], [1])
        line.trigger(dets1)
        dets2 = _make_tracked_dets([(200, 150)], [1])
        line.trigger(dets2)
        assert line.in_count == 1

        line.reset()
        assert line.in_count == 0
        assert line.out_count == 0

    def test_empty_detections(self):
        line = LineZone(start=Point(0, 100), end=Point(400, 100))
        in_mask, out_mask = line.trigger(Detections.empty())
        assert len(in_mask) == 0

    def test_no_tracker_id(self):
        line = LineZone(start=Point(0, 100), end=Point(400, 100))
        dets = _make_dets_at([(200, 50)])  # no tracker_id
        in_mask, out_mask = line.trigger(dets)
        assert not in_mask.any()

    def test_repr(self):
        line = LineZone(start=Point(10, 20), end=Point(30, 40))
        r = repr(line)
        assert "10" in r and "20" in r


# ===========================================================================
# ZoneCondition Tests
# ===========================================================================


class TestZoneCondition:
    def test_ge_threshold_met(self):
        zone = _square_zone(0, 0, 500)
        cond = ZoneCondition(zone=zone, operator=">=", count_threshold=2)
        dets = _make_dets_at([(100, 100), (200, 200), (300, 300)])
        assert cond.evaluate(dets)

    def test_ge_threshold_not_met(self):
        zone = _square_zone(0, 0, 500)
        cond = ZoneCondition(zone=zone, operator=">=", count_threshold=5)
        dets = _make_dets_at([(100, 100), (200, 200)])
        assert not cond.evaluate(dets)

    def test_eq_operator(self):
        zone = _square_zone(0, 0, 500)
        cond = ZoneCondition(zone=zone, operator="==", count_threshold=2)
        dets = _make_dets_at([(100, 100), (200, 200)])
        assert cond.evaluate(dets)

    def test_lt_operator(self):
        zone = _square_zone(0, 0, 500)
        cond = ZoneCondition(zone=zone, operator="<", count_threshold=3)
        dets = _make_dets_at([(100, 100), (200, 200)])
        assert cond.evaluate(dets)

    def test_unknown_operator_defaults_ge(self):
        zone = _square_zone(0, 0, 500)
        cond = ZoneCondition(zone=zone, operator="??", count_threshold=1)
        dets = _make_dets_at([(100, 100)])
        # Should default to >= and pass
        assert cond.evaluate(dets)


# ===========================================================================
# ZoneConditionEvaluator Tests
# ===========================================================================


class TestZoneConditionEvaluator:
    def test_register_and_evaluate(self):
        evaluator = ZoneConditionEvaluator()
        zone = _square_zone(0, 0, 500)
        cond = ZoneCondition(zone=zone, operator=">=", count_threshold=1)
        evaluator.register("test_zone", cond)

        dets = _make_dets_at([(100, 100)])
        assert evaluator.evaluate("test_zone", dets)

    def test_evaluate_all(self):
        evaluator = ZoneConditionEvaluator()
        zone1 = _square_zone(0, 0, 500)
        zone2 = _square_zone(0, 0, 50)  # small zone
        evaluator.register(
            "big", ZoneCondition(zone=zone1, operator=">=", count_threshold=1)
        )
        evaluator.register(
            "small", ZoneCondition(zone=zone2, operator=">=", count_threshold=1)
        )

        dets = _make_dets_at([(250, 250)])  # inside big, outside small
        results = evaluator.evaluate_all(dets)
        assert results["big"]
        assert not results["small"]

    def test_triggered(self):
        evaluator = ZoneConditionEvaluator()
        zone = _square_zone(0, 0, 500)
        evaluator.register(
            "yes", ZoneCondition(zone=zone, operator=">=", count_threshold=1)
        )
        evaluator.register(
            "no", ZoneCondition(zone=zone, operator=">=", count_threshold=99)
        )

        dets = _make_dets_at([(100, 100)])
        triggered = evaluator.triggered(dets)
        assert "yes" in triggered
        assert "no" not in triggered

    def test_unregister(self):
        evaluator = ZoneConditionEvaluator()
        zone = _square_zone(0, 0, 500)
        evaluator.register(
            "a", ZoneCondition(zone=zone, operator=">=", count_threshold=1)
        )
        evaluator.unregister("a")
        assert "a" not in evaluator.conditions

    def test_missing_condition_raises(self):
        evaluator = ZoneConditionEvaluator()
        dets = _make_dets_at([(100, 100)])
        with pytest.raises(KeyError):
            evaluator.evaluate("nonexistent", dets)
