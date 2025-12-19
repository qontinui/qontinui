"""Tests for coordinate types (ScreenPoint, VirtualPoint, MonitorPoint, MonitorInfo)."""

import pytest

from qontinui.coordinates.types import MonitorInfo, MonitorPoint, ScreenPoint, VirtualPoint


class TestScreenPoint:
    """Test ScreenPoint coordinate type."""

    def test_creation(self) -> None:
        """Test creating a ScreenPoint."""
        point = ScreenPoint(100, 200)
        assert point.x == 100
        assert point.y == 200

    def test_negative_coordinates(self) -> None:
        """Test ScreenPoint with negative coordinates (left/above monitors)."""
        point = ScreenPoint(-1920, -1080)
        assert point.x == -1920
        assert point.y == -1080

    def test_immutable(self) -> None:
        """Test that ScreenPoint is immutable."""
        point = ScreenPoint(100, 200)
        with pytest.raises(Exception):  # dataclass frozen=True raises FrozenInstanceError
            point.x = 300  # type: ignore

    def test_equality(self) -> None:
        """Test ScreenPoint equality."""
        p1 = ScreenPoint(100, 200)
        p2 = ScreenPoint(100, 200)
        p3 = ScreenPoint(200, 100)

        assert p1 == p2
        assert p1 != p3

    def test_repr(self) -> None:
        """Test ScreenPoint string representation."""
        point = ScreenPoint(100, 200)
        assert repr(point) == "ScreenPoint(x=100, y=200)"


class TestVirtualPoint:
    """Test VirtualPoint coordinate type."""

    def test_creation(self) -> None:
        """Test creating a VirtualPoint."""
        point = VirtualPoint(100, 200)
        assert point.x == 100
        assert point.y == 200

    def test_zero_coordinates(self) -> None:
        """Test VirtualPoint at origin."""
        point = VirtualPoint(0, 0)
        assert point.x == 0
        assert point.y == 0

    def test_immutable(self) -> None:
        """Test that VirtualPoint is immutable."""
        point = VirtualPoint(100, 200)
        with pytest.raises(Exception):
            point.x = 300  # type: ignore

    def test_equality(self) -> None:
        """Test VirtualPoint equality."""
        p1 = VirtualPoint(100, 200)
        p2 = VirtualPoint(100, 200)
        p3 = VirtualPoint(200, 100)

        assert p1 == p2
        assert p1 != p3

    def test_repr(self) -> None:
        """Test VirtualPoint string representation."""
        point = VirtualPoint(100, 200)
        assert repr(point) == "VirtualPoint(x=100, y=200)"


class TestMonitorPoint:
    """Test MonitorPoint coordinate type."""

    def test_creation(self) -> None:
        """Test creating a MonitorPoint."""
        point = MonitorPoint(100, 200, monitor_index=1)
        assert point.x == 100
        assert point.y == 200
        assert point.monitor_index == 1

    def test_immutable(self) -> None:
        """Test that MonitorPoint is immutable."""
        point = MonitorPoint(100, 200, monitor_index=1)
        with pytest.raises(Exception):
            point.x = 300  # type: ignore

    def test_equality(self) -> None:
        """Test MonitorPoint equality."""
        p1 = MonitorPoint(100, 200, monitor_index=1)
        p2 = MonitorPoint(100, 200, monitor_index=1)
        p3 = MonitorPoint(100, 200, monitor_index=2)

        assert p1 == p2
        assert p1 != p3  # Different monitor

    def test_repr(self) -> None:
        """Test MonitorPoint string representation."""
        point = MonitorPoint(100, 200, monitor_index=1)
        assert repr(point) == "MonitorPoint(x=100, y=200, monitor=1)"


class TestMonitorInfo:
    """Test MonitorInfo type."""

    def test_creation(self) -> None:
        """Test creating MonitorInfo."""
        monitor = MonitorInfo(index=0, x=0, y=0, width=1920, height=1080, is_primary=True)

        assert monitor.index == 0
        assert monitor.x == 0
        assert monitor.y == 0
        assert monitor.width == 1920
        assert monitor.height == 1080
        assert monitor.is_primary is True

    def test_bounds_property(self) -> None:
        """Test bounds property returns correct tuple."""
        monitor = MonitorInfo(index=0, x=100, y=200, width=1920, height=1080, is_primary=False)

        bounds = monitor.bounds
        assert bounds == (100, 200, 1920, 1080)

    def test_contains_point_inside(self) -> None:
        """Test contains_point for point inside monitor bounds."""
        monitor = MonitorInfo(index=0, x=0, y=0, width=1920, height=1080, is_primary=True)

        # Test center point
        assert monitor.contains_point(960, 540) is True

        # Test top-left corner
        assert monitor.contains_point(0, 0) is True

        # Test bottom-right corner (exclusive)
        assert monitor.contains_point(1919, 1079) is True

    def test_contains_point_outside(self) -> None:
        """Test contains_point for point outside monitor bounds."""
        monitor = MonitorInfo(index=0, x=0, y=0, width=1920, height=1080, is_primary=True)

        # Outside to the right
        assert monitor.contains_point(1920, 540) is False

        # Outside below
        assert monitor.contains_point(960, 1080) is False

        # Outside to the left (negative)
        assert monitor.contains_point(-1, 540) is False

    def test_contains_point_negative_coordinates(self) -> None:
        """Test contains_point for monitor with negative coordinates."""
        # Monitor positioned to the left of primary
        monitor = MonitorInfo(index=1, x=-1920, y=0, width=1920, height=1080, is_primary=False)

        # Point inside left monitor
        assert monitor.contains_point(-1000, 500) is True

        # Point at left edge
        assert monitor.contains_point(-1920, 500) is True

        # Point outside left monitor (too far left)
        assert monitor.contains_point(-1921, 500) is False

        # Point outside left monitor (on primary)
        assert monitor.contains_point(0, 500) is False

    def test_repr_primary(self) -> None:
        """Test repr for primary monitor."""
        monitor = MonitorInfo(index=0, x=0, y=0, width=1920, height=1080, is_primary=True)

        repr_str = repr(monitor)
        assert "index=0" in repr_str
        assert "1920" in repr_str
        assert "1080" in repr_str
        assert "(primary)" in repr_str

    def test_repr_non_primary(self) -> None:
        """Test repr for non-primary monitor."""
        monitor = MonitorInfo(index=1, x=1920, y=0, width=1920, height=1080, is_primary=False)

        repr_str = repr(monitor)
        assert "index=1" in repr_str
        assert "(primary)" not in repr_str
