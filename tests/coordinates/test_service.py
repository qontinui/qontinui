"""Tests for CoordinateService."""

from unittest.mock import patch

import pytest

from qontinui.coordinates.service import CoordinateService
from qontinui.coordinates.types import ScreenPoint, VirtualPoint


@pytest.fixture
def mock_single_monitor_mss() -> list[dict[str, int]]:
    """Mock MSS monitor data for a single monitor at origin."""
    return [
        {"left": 0, "top": 0, "width": 1920, "height": 1080},  # Virtual desktop
        {"left": 0, "top": 0, "width": 1920, "height": 1080},  # Physical monitor
    ]


@pytest.fixture
def mock_dual_monitor_standard_mss() -> list[dict[str, int]]:
    """Mock MSS monitor data for dual monitors (standard layout - secondary on right)."""
    return [
        {"left": 0, "top": 0, "width": 3840, "height": 1080},  # Virtual desktop
        {"left": 0, "top": 0, "width": 1920, "height": 1080},  # Primary (left)
        {"left": 1920, "top": 0, "width": 1920, "height": 1080},  # Secondary (right)
    ]


@pytest.fixture
def mock_dual_monitor_left_mss() -> list[dict[str, int]]:
    """Mock MSS monitor data for dual monitors (secondary on LEFT - negative X)."""
    return [
        {"left": -1920, "top": 0, "width": 3840, "height": 1080},  # Virtual desktop
        {"left": 0, "top": 0, "width": 1920, "height": 1080},  # Primary (right)
        {"left": -1920, "top": 0, "width": 1920, "height": 1080},  # Secondary (left)
    ]


@pytest.fixture
def mock_triple_monitor_complex_mss() -> list[dict[str, int]]:
    """Mock MSS monitor data for three monitors in complex layout.

    Layout:
        Left:    x=-1920, y=702,  1920x1080
        Primary: x=0,     y=0,    3840x2160
        Right:   x=3840,  y=702,  1920x1080
    """
    return [
        {"left": -1920, "top": 0, "width": 7680, "height": 2160},  # Virtual desktop
        {"left": 0, "top": 0, "width": 3840, "height": 2160},  # Primary (center)
        {"left": -1920, "top": 702, "width": 1920, "height": 1080},  # Left
        {"left": 3840, "top": 702, "width": 1920, "height": 1080},  # Right
    ]


class TestCoordinateServiceSingleton:
    """Test CoordinateService singleton behavior."""

    def test_singleton(self) -> None:
        """Test that get_instance returns the same instance."""
        s1 = CoordinateService.get_instance()
        s2 = CoordinateService.get_instance()
        assert s1 is s2

    def test_singleton_thread_safety(self) -> None:
        """Test that singleton is thread-safe."""
        import threading

        instances = []

        def get_instance() -> None:
            instances.append(CoordinateService.get_instance())

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All instances should be the same object
        assert all(inst is instances[0] for inst in instances)


class TestCoordinateServiceMatchToScreen:
    """Test match_to_screen conversions (VirtualPoint -> ScreenPoint)."""

    def test_match_to_screen_single_monitor(
        self, mock_single_monitor_mss: list[dict[str, int]]
    ) -> None:
        """Test match_to_screen with single monitor at origin."""
        with patch("mss.mss") as mock_mss:
            mock_mss.return_value.__enter__.return_value.monitors = mock_single_monitor_mss

            service = CoordinateService()
            screen_point = service.match_to_screen(100, 200)

            # Origin is (0, 0), so match coords = screen coords
            assert screen_point.x == 100
            assert screen_point.y == 200

    def test_match_to_screen_dual_standard(
        self, mock_dual_monitor_standard_mss: list[dict[str, int]]
    ) -> None:
        """Test match_to_screen with standard dual monitor layout."""
        with patch("mss.mss") as mock_mss:
            mock_mss.return_value.__enter__.return_value.monitors = mock_dual_monitor_standard_mss

            service = CoordinateService()
            screen_point = service.match_to_screen(100, 200)

            # Origin is (0, 0), so match coords = screen coords
            assert screen_point.x == 100
            assert screen_point.y == 200

    def test_match_to_screen_dual_left_monitor(
        self, mock_dual_monitor_left_mss: list[dict[str, int]]
    ) -> None:
        """Test match_to_screen with secondary monitor on LEFT (negative X).

        This is the critical test case that was broken.
        """
        with patch("mss.mss") as mock_mss:
            mock_mss.return_value.__enter__.return_value.monitors = mock_dual_monitor_left_mss

            service = CoordinateService()

            # Virtual desktop origin is at (-1920, 0)
            # Match at (65, 1372) in screenshot should be screen (-1855, 1372)
            screen_point = service.match_to_screen(65, 1372)

            assert screen_point.x == -1855  # 65 + (-1920)
            assert screen_point.y == 1372  # 1372 + 0

    def test_match_to_screen_triple_complex(
        self, mock_triple_monitor_complex_mss: list[dict[str, int]]
    ) -> None:
        """Test match_to_screen with complex three-monitor layout."""
        with patch("mss.mss") as mock_mss:
            mock_mss.return_value.__enter__.return_value.monitors = mock_triple_monitor_complex_mss

            service = CoordinateService()

            # Virtual desktop origin is at (-1920, 0)
            # Match at (100, 100) should be screen (-1820, 100)
            screen_point = service.match_to_screen(100, 100)

            assert screen_point.x == -1820  # 100 + (-1920)
            assert screen_point.y == 100  # 100 + 0


class TestCoordinateServiceScreenToMatch:
    """Test screen_to_match conversions (ScreenPoint -> VirtualPoint)."""

    def test_screen_to_match_single_monitor(
        self, mock_single_monitor_mss: list[dict[str, int]]
    ) -> None:
        """Test screen_to_match with single monitor."""
        with patch("mss.mss") as mock_mss:
            mock_mss.return_value.__enter__.return_value.monitors = mock_single_monitor_mss

            service = CoordinateService()
            virtual_point = service.screen_to_match(100, 200)

            assert virtual_point.x == 100
            assert virtual_point.y == 200

    def test_screen_to_match_dual_left_monitor(
        self, mock_dual_monitor_left_mss: list[dict[str, int]]
    ) -> None:
        """Test screen_to_match with left monitor (negative screen coords)."""
        with patch("mss.mss") as mock_mss:
            mock_mss.return_value.__enter__.return_value.monitors = mock_dual_monitor_left_mss

            service = CoordinateService()

            # Screen at (-1855, 1372) should be virtual (65, 1372)
            virtual_point = service.screen_to_match(-1855, 1372)

            assert virtual_point.x == 65  # -1855 - (-1920)
            assert virtual_point.y == 1372  # 1372 - 0


class TestCoordinateServiceRoundTrip:
    """Test round-trip conversions (should be lossless)."""

    def test_match_to_screen_round_trip_single(
        self, mock_single_monitor_mss: list[dict[str, int]]
    ) -> None:
        """Test round-trip conversion with single monitor."""
        with patch("mss.mss") as mock_mss:
            mock_mss.return_value.__enter__.return_value.monitors = mock_single_monitor_mss

            service = CoordinateService()
            original = VirtualPoint(100, 200)

            screen = service.match_to_screen(original.x, original.y)
            back = service.screen_to_match(screen.x, screen.y)

            assert back.x == original.x
            assert back.y == original.y

    def test_match_to_screen_round_trip_dual_left(
        self, mock_dual_monitor_left_mss: list[dict[str, int]]
    ) -> None:
        """Test round-trip conversion with left monitor."""
        with patch("mss.mss") as mock_mss:
            mock_mss.return_value.__enter__.return_value.monitors = mock_dual_monitor_left_mss

            service = CoordinateService()
            original = VirtualPoint(65, 1372)

            screen = service.match_to_screen(original.x, original.y)
            back = service.screen_to_match(screen.x, screen.y)

            assert back.x == original.x
            assert back.y == original.y

    def test_screen_to_match_round_trip_dual_left(
        self, mock_dual_monitor_left_mss: list[dict[str, int]]
    ) -> None:
        """Test round-trip conversion starting from screen coords."""
        with patch("mss.mss") as mock_mss:
            mock_mss.return_value.__enter__.return_value.monitors = mock_dual_monitor_left_mss

            service = CoordinateService()
            original = ScreenPoint(-1855, 1372)

            virtual = service.screen_to_match(original.x, original.y)
            back = service.match_to_screen(virtual.x, virtual.y)

            assert back.x == original.x
            assert back.y == original.y


class TestCoordinateServiceMonitorConversions:
    """Test monitor-relative coordinate conversions."""

    def test_monitor_to_screen(self, mock_dual_monitor_standard_mss: list[dict[str, int]]) -> None:
        """Test monitor_to_screen conversion."""
        with patch("mss.mss") as mock_mss:
            mock_mss.return_value.__enter__.return_value.monitors = mock_dual_monitor_standard_mss

            service = CoordinateService()

            # Point at (100, 100) on monitor 1 (right monitor at x=1920)
            screen_point = service.monitor_to_screen(100, 100, monitor_index=1)

            assert screen_point.x == 2020  # 100 + 1920
            assert screen_point.y == 100  # 100 + 0

    def test_monitor_to_screen_invalid_index(
        self, mock_single_monitor_mss: list[dict[str, int]]
    ) -> None:
        """Test monitor_to_screen with invalid monitor index."""
        with patch("mss.mss") as mock_mss:
            mock_mss.return_value.__enter__.return_value.monitors = mock_single_monitor_mss

            service = CoordinateService()

            with pytest.raises(ValueError, match="Invalid monitor index"):
                service.monitor_to_screen(100, 100, monitor_index=5)

    def test_screen_to_monitor(self, mock_dual_monitor_standard_mss: list[dict[str, int]]) -> None:
        """Test screen_to_monitor conversion."""
        with patch("mss.mss") as mock_mss:
            mock_mss.return_value.__enter__.return_value.monitors = mock_dual_monitor_standard_mss

            service = CoordinateService()

            # Screen point at (2020, 100) on monitor 1
            monitor_point = service.screen_to_monitor(2020, 100, monitor_index=1)

            assert monitor_point.x == 100  # 2020 - 1920
            assert monitor_point.y == 100  # 100 - 0
            assert monitor_point.monitor_index == 1

    def test_monitor_round_trip(self, mock_dual_monitor_standard_mss: list[dict[str, int]]) -> None:
        """Test round-trip conversion for monitor coordinates."""
        with patch("mss.mss") as mock_mss:
            mock_mss.return_value.__enter__.return_value.monitors = mock_dual_monitor_standard_mss

            service = CoordinateService()
            original_x, original_y, monitor_idx = 100, 100, 1

            screen = service.monitor_to_screen(original_x, original_y, monitor_idx)
            back = service.screen_to_monitor(screen.x, screen.y, monitor_idx)

            assert back.x == original_x
            assert back.y == original_y
            assert back.monitor_index == monitor_idx


class TestCoordinateServiceMonitorQueries:
    """Test monitor query methods."""

    def test_get_monitor_at_point(
        self, mock_dual_monitor_standard_mss: list[dict[str, int]]
    ) -> None:
        """Test getting monitor at a specific screen point."""
        with patch("mss.mss") as mock_mss:
            mock_mss.return_value.__enter__.return_value.monitors = mock_dual_monitor_standard_mss

            service = CoordinateService()

            # Point on primary monitor
            monitor_idx = service.get_monitor_at_point(500, 500)
            assert monitor_idx == 0

            # Point on secondary monitor
            monitor_idx = service.get_monitor_at_point(2000, 500)
            assert monitor_idx == 1

    def test_get_monitor_at_point_outside(
        self, mock_single_monitor_mss: list[dict[str, int]]
    ) -> None:
        """Test getting monitor for point outside all monitors."""
        with patch("mss.mss") as mock_mss:
            mock_mss.return_value.__enter__.return_value.monitors = mock_single_monitor_mss

            service = CoordinateService()

            # Point outside monitor bounds
            monitor_idx = service.get_monitor_at_point(10000, 10000)
            assert monitor_idx is None

    def test_get_monitor_count(self, mock_dual_monitor_standard_mss: list[dict[str, int]]) -> None:
        """Test getting monitor count."""
        with patch("mss.mss") as mock_mss:
            mock_mss.return_value.__enter__.return_value.monitors = mock_dual_monitor_standard_mss

            service = CoordinateService()
            count = service.get_monitor_count()

            assert count == 2


class TestCoordinateServiceRefresh:
    """Test refresh functionality."""

    def test_refresh(self, mock_single_monitor_mss: list[dict[str, int]]) -> None:
        """Test that refresh updates monitor information."""
        with patch("mss.mss") as mock_mss:
            # Start with one monitor
            mock_mss.return_value.__enter__.return_value.monitors = mock_single_monitor_mss

            service = CoordinateService()
            assert service.get_monitor_count() == 1

            # Simulate adding a monitor
            dual_monitors = [
                {"left": 0, "top": 0, "width": 3840, "height": 1080},
                {"left": 0, "top": 0, "width": 1920, "height": 1080},
                {"left": 1920, "top": 0, "width": 1920, "height": 1080},
            ]
            mock_mss.return_value.__enter__.return_value.monitors = dual_monitors

            service.refresh()
            assert service.get_monitor_count() == 2


class TestCoordinateServiceRepr:
    """Test string representation."""

    def test_repr(self, mock_single_monitor_mss: list[dict[str, int]]) -> None:
        """Test repr output."""
        with patch("mss.mss") as mock_mss:
            mock_mss.return_value.__enter__.return_value.monitors = mock_single_monitor_mss

            service = CoordinateService()
            repr_str = repr(service)

            assert "CoordinateService" in repr_str
            assert "monitors=1" in repr_str
            assert "1920x1080" in repr_str
