"""Integration tests using real MSS monitor detection.

These tests use the actual MSS library to detect real monitors on the system.
They should work on any monitor configuration (single, dual, triple, etc.).
"""

import pytest

from qontinui.coordinates.service import CoordinateService


class TestIntegrationRealMonitors:
    """Integration tests with real monitor detection."""

    def test_real_monitor_detection(self) -> None:
        """Test that we can detect real monitors on the system."""
        service = CoordinateService.get_instance()
        desktop = service.get_virtual_desktop()

        # Should detect at least one monitor
        assert len(desktop.monitors) > 0
        print(f"\nDetected {len(desktop.monitors)} monitor(s)")

        # Print monitor info for debugging
        for monitor in desktop.monitors:
            print(
                f"  Monitor {monitor.index}: "
                f"{monitor.width}x{monitor.height} "
                f"at ({monitor.x}, {monitor.y})"
                f"{' (primary)' if monitor.is_primary else ''}"
            )

    def test_primary_monitor_exists(self) -> None:
        """Test that at least one monitor is marked as primary."""
        service = CoordinateService.get_instance()
        desktop = service.get_virtual_desktop()

        # At least one monitor should be primary
        assert any(m.is_primary for m in desktop.monitors)

        primary = desktop.get_primary_monitor()
        assert primary is not None
        assert primary.is_primary is True
        print(f"\nPrimary monitor: {primary.width}x{primary.height} at ({primary.x}, {primary.y})")

    def test_virtual_desktop_bounds(self) -> None:
        """Test that virtual desktop bounds are calculated correctly."""
        service = CoordinateService.get_instance()
        desktop = service.get_virtual_desktop()

        print(
            f"\nVirtual desktop: {desktop.width}x{desktop.height} "
            f"at origin ({desktop.origin_x}, {desktop.origin_y})"
        )

        # Virtual desktop should have positive dimensions
        assert desktop.width > 0
        assert desktop.height > 0

        # Origin should match min X and Y across all monitors
        if len(desktop.monitors) > 0:
            min_x = min(m.x for m in desktop.monitors)
            min_y = min(m.y for m in desktop.monitors)
            assert desktop.origin_x == min_x
            assert desktop.origin_y == min_y

    def test_coordinate_conversions_real_system(self) -> None:
        """Test coordinate conversions work on real system."""
        service = CoordinateService.get_instance()
        desktop = service.get_virtual_desktop()

        # Get the primary monitor center point
        primary = desktop.get_primary_monitor()
        assert primary is not None

        # Calculate center of primary monitor (in screen coordinates)
        center_x = primary.x + primary.width // 2
        center_y = primary.y + primary.height // 2

        print(f"\nPrimary monitor center (screen): ({center_x}, {center_y})")

        # Convert to virtual coordinates
        virtual_point = service.screen_to_match(center_x, center_y)
        print(f"Primary monitor center (virtual): ({virtual_point.x}, {virtual_point.y})")

        # Convert back to screen coordinates
        screen_point = service.match_to_screen(virtual_point.x, virtual_point.y)
        print(f"Round-trip (screen): ({screen_point.x}, {screen_point.y})")

        # Should be lossless
        assert screen_point.x == center_x
        assert screen_point.y == center_y

    def test_monitor_detection_at_points(self) -> None:
        """Test that monitor detection works at specific points."""
        service = CoordinateService.get_instance()
        desktop = service.get_virtual_desktop()

        for monitor in desktop.monitors:
            # Test center point of each monitor
            center_x = monitor.x + monitor.width // 2
            center_y = monitor.y + monitor.height // 2

            detected_idx = service.get_monitor_at_point(center_x, center_y)
            assert (
                detected_idx == monitor.index
            ), f"Failed to detect monitor {monitor.index} at its center"

            # Test top-left corner
            detected_idx = service.get_monitor_at_point(monitor.x, monitor.y)
            assert (
                detected_idx == monitor.index
            ), f"Failed to detect monitor {monitor.index} at its top-left"

            print(f"\nMonitor {monitor.index} detection: OK")

    def test_monitor_relative_conversions(self) -> None:
        """Test monitor-relative coordinate conversions."""
        service = CoordinateService.get_instance()
        desktop = service.get_virtual_desktop()

        for monitor in desktop.monitors:
            # Convert (0, 0) on this monitor to screen coordinates
            # Should equal the monitor's position
            screen_point = service.monitor_to_screen(0, 0, monitor.index)
            assert screen_point.x == monitor.x
            assert screen_point.y == monitor.y

            # Convert back
            monitor_point = service.screen_to_monitor(monitor.x, monitor.y, monitor.index)
            assert monitor_point.x == 0
            assert monitor_point.y == 0
            assert monitor_point.monitor_index == monitor.index

            print(f"\nMonitor {monitor.index} relative conversions: OK")

    def test_service_repr(self) -> None:
        """Test that service repr provides useful information."""
        service = CoordinateService.get_instance()
        repr_str = repr(service)

        print(f"\nCoordinateService repr: {repr_str}")

        assert "CoordinateService" in repr_str
        assert "monitors=" in repr_str

    def test_refresh_maintains_accuracy(self) -> None:
        """Test that refresh() maintains accurate monitor detection."""
        service = CoordinateService.get_instance()

        # Get initial state
        initial_count = service.get_monitor_count()
        initial_desktop = service.get_virtual_desktop()

        # Refresh
        service.refresh()

        # Count should be the same (monitors didn't change)
        new_count = service.get_monitor_count()
        assert new_count == initial_count

        # Virtual desktop should be the same
        new_desktop = service.get_virtual_desktop()
        assert new_desktop.origin_x == initial_desktop.origin_x
        assert new_desktop.origin_y == initial_desktop.origin_y
        assert new_desktop.width == initial_desktop.width
        assert new_desktop.height == initial_desktop.height

        print("\nRefresh maintains accuracy: OK")


class TestIntegrationMultiMonitorScenarios:
    """Integration tests for multi-monitor scenarios (if available)."""

    @pytest.mark.skipif(
        CoordinateService.get_instance().get_monitor_count() < 2,
        reason="Requires at least 2 monitors",
    )
    def test_dual_monitor_conversions(self) -> None:
        """Test conversions on dual monitor setup."""
        service = CoordinateService.get_instance()
        desktop = service.get_virtual_desktop()

        print(f"\nTesting with {len(desktop.monitors)} monitors")

        # Test conversions on both monitors
        for monitor in desktop.monitors:
            # Test a point on this monitor
            test_x = monitor.x + 100
            test_y = monitor.y + 100

            # Convert to virtual
            virtual = service.screen_to_match(test_x, test_y)

            # Convert back
            screen = service.match_to_screen(virtual.x, virtual.y)

            assert screen.x == test_x
            assert screen.y == test_y

            print(
                f"  Monitor {monitor.index}: "
                f"screen({test_x}, {test_y}) -> "
                f"virtual({virtual.x}, {virtual.y}) -> "
                f"screen({screen.x}, {screen.y})"
            )

    @pytest.mark.skipif(
        CoordinateService.get_instance().get_monitor_count() < 2,
        reason="Requires at least 2 monitors",
    )
    def test_monitor_boundaries(self) -> None:
        """Test that monitor boundaries are detected correctly."""
        service = CoordinateService.get_instance()
        desktop = service.get_virtual_desktop()

        for i, monitor in enumerate(desktop.monitors):
            # Test edge cases
            # Top-left corner (inclusive)
            detected = service.get_monitor_at_point(monitor.x, monitor.y)
            assert detected == i, f"Top-left corner of monitor {i} not detected"

            # Bottom-right corner (exclusive in contains_point)
            detected = service.get_monitor_at_point(
                monitor.x + monitor.width - 1, monitor.y + monitor.height - 1
            )
            assert detected == i, f"Bottom-right corner (inclusive) of monitor {i} not detected"

            # Just outside right edge (should NOT be on this monitor)
            detected = service.get_monitor_at_point(
                monitor.x + monitor.width, monitor.y + monitor.height // 2
            )
            # Could be on another monitor or None
            assert (
                detected != i or detected is None
            ), f"Point outside monitor {i} incorrectly detected as on monitor {i}"

            print(f"Monitor {i} boundaries: OK")

    @pytest.mark.skipif(
        any(
            m.x < 0 or m.y < 0
            for m in CoordinateService.get_instance().get_virtual_desktop().monitors
        )
        is False,
        reason="Requires monitors with negative coordinates",
    )
    def test_negative_coordinate_handling(self) -> None:
        """Test handling of monitors with negative coordinates.

        This test only runs if the system has monitors positioned left of
        or above the primary monitor (resulting in negative coordinates).
        """
        service = CoordinateService.get_instance()
        desktop = service.get_virtual_desktop()

        # Find monitors with negative coordinates
        negative_monitors = [m for m in desktop.monitors if m.x < 0 or m.y < 0]

        print(f"\nFound {len(negative_monitors)} monitor(s) with negative coordinates")

        for monitor in negative_monitors:
            print(f"  Monitor {monitor.index}: ({monitor.x}, {monitor.y})")

            # Test a point on the negative monitor
            test_x = monitor.x + 100
            test_y = monitor.y + 100

            # Should be able to convert correctly
            virtual = service.screen_to_match(test_x, test_y)
            screen = service.match_to_screen(virtual.x, virtual.y)

            assert screen.x == test_x
            assert screen.y == test_y

            # Should detect the correct monitor
            detected = service.get_monitor_at_point(test_x, test_y)
            assert detected == monitor.index

            print(f"  Monitor {monitor.index} negative coordinate handling: OK")
